"""Megatron-style MoE MLP implementation for HuggingFace compatibility.

This module mirrors Megatron-LM's MoE structure (Router + Experts) as closely
as practical in a single-process, non-parallel HF context.

Key components:
- TopKRouter: projects tokens to expert logits and selects top-k experts per token.
- Expert: a per-expert MLP with linear_fc1 and linear_fc2 following Megatron naming.
- MoeExperts: a container of Expert modules and combine logic.
- MoeMLP: ties router + experts and returns combined outputs, following MLP API.

Notes:
- This is a simplified implementation without EP/TP/A2A dispatch. It performs
  gather/compute/scatter on CPU/GPU per expert and combines results weighted
  by router probabilities.
- The linear_fc1 and linear_fc2 names match Megatron conventions for compatibility
  with conversion and checkpoint mapping logic.
- Biases are handled inside each nn.Linear. Hence we return output and bias=None.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging

from ..configuration_megatron import MegatronConfig

logger = logging.get_logger(__name__)


def _get_activation_from_config(config: MegatronConfig):
    if hasattr(config, "activation_function"):
        if isinstance(config.activation_function, str):
            return ACT2FN[config.activation_function]
        else:
            return config.activation_function
    return F.gelu


class TopKRouter(nn.Module):
    """Top-k Router closely following Megatron-LM semantics (simplified).

    Computes expert logits via a learned gate (weight shape [num_experts, hidden])
    then routes each token to top-k experts, producing per-token probabilities
    and expert indices.

    Config knobs used:
      - num_experts (int)
      - moe_router_topk (int)
      - moe_router_pre_softmax (bool)
      - moe_router_score_function ('softmax' or 'sigmoid')
      - moe_router_dtype ('fp32'|'fp64'|None)
    """

    def __init__(self, config: MegatronConfig):
        super().__init__()
        self.config = config
        assert (
            config.num_experts is not None and config.num_experts > 0
        ), "MoE requires num_experts > 0"
        self.num_experts = config.num_experts
        self.topk = max(1, int(getattr(config, "moe_router_topk", 2)))
        self.score_function = getattr(config, "moe_router_score_function", "softmax")
        self.pre_softmax = bool(getattr(config, "moe_router_pre_softmax", False))
        self.router_dtype = getattr(config, "moe_router_dtype", None)

        # Gate weight matches Megatron's convention: [num_experts, hidden_size]
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Simple init compatible with Megatron default std
        std = getattr(self.config, "init_method_std", 0.02)
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=std)
        # Move weight to configured params dtype if desired
        params_dtype = torch.float32
        if getattr(self.config, "bf16", False):
            params_dtype = torch.bfloat16
        elif getattr(self.config, "fp16", False):
            params_dtype = torch.float16
        self.weight.data = self.weight.data.to(dtype=params_dtype)

    def gating(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits = hidden @ weight^T with optional dtype override for stability."""
        # hidden [T, H], weight [E, H] => logits [T, E]
        # Cast compute dtype if requested
        compute_dtype = hidden.dtype
        if self.router_dtype == "fp32":
            compute_dtype = torch.float32
        elif self.router_dtype == "fp64":
            compute_dtype = torch.float64
        logits = torch.matmul(
            hidden.to(dtype=compute_dtype), self.weight.t().to(dtype=compute_dtype)
        )
        return logits

    def _apply_score_function(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_function == "softmax":
            return torch.softmax(logits, dim=-1, dtype=torch.float32).to(
                dtype=logits.dtype
            )
        elif self.score_function == "sigmoid":
            # Normalize across experts to form probabilities
            probs = torch.sigmoid(logits)
            denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            return (probs / denom).to(dtype=logits.dtype)
        else:
            raise ValueError(
                f"Unsupported moe_router_score_function: {self.score_function}"
            )

    def routing(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (probs [T, k], indices [T, k]) for selected experts per token."""
        T, E = logits.shape
        k = min(self.topk, E)

        if self.pre_softmax:
            # Compute scores first, then top-k on probabilities
            scores = self._apply_score_function(logits)
            topk_scores, indices = torch.topk(scores, k=k, dim=-1)
            # Normalize selected scores to sum to 1 per token (safe)
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            probs = topk_scores / denom
        else:
            # Top-k on raw logits, then softmax over selected logits
            topk_logits, indices = torch.topk(logits, k=k, dim=-1)
            # Softmax only across selected logits for each token
            probs = torch.softmax(topk_logits, dim=-1)

        return probs.to(dtype=logits.dtype), indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden_states: [S, B, H] -> returns (probs[T, k], indices[T, k])"""
        S, B, H = hidden_states.shape
        hidden = hidden_states.reshape(S * B, H)
        logits = self.gating(hidden)
        probs, indices = self.routing(logits)
        return probs, indices


class Expert(nn.Module):
    """Single expert MLP using Megatron naming: linear_fc1, linear_fc2."""

    def __init__(self, config: MegatronConfig, input_size: int | None = None):
        super().__init__()
        self.config = config
        self.input_size = input_size if input_size is not None else config.hidden_size

        ffn_hidden_size = (
            config.moe_ffn_hidden_size
            if config.moe_ffn_hidden_size
            else config.ffn_hidden_size
        )
        out_fc1 = (
            ffn_hidden_size * 2 if getattr(config, "swiglu", False) else ffn_hidden_size
        )

        # First linear (with optional pre-LN bias as HF's LinearLayerNorm wrapper)
        self.linear_fc1 = nn.Linear(
            self.input_size,
            out_fc1,
            bias=getattr(config, "add_bias_linear", True),
        )

        # Activation setup
        self.activation_func = _get_activation_from_config(config)

        # Second linear
        self.linear_fc2 = nn.Linear(
            ffn_hidden_size,
            config.hidden_size,
            bias=getattr(config, "add_bias_linear", True),
        )

        self.dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))
        self.use_swiglu = bool(getattr(config, "swiglu", False))

    def forward(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        """hidden_tokens: [N_tok_e, H] -> [N_tok_e, H]"""
        x = self.linear_fc1(hidden_tokens)  # [N, ffn*(2 if swiglu else 1)]

        if self.use_swiglu:
            a, b = torch.chunk(x, 2, dim=-1)
            x = F.silu(a) * b
        else:
            x = self.activation_func(x)

        x = self.dropout(x)
        x = self.linear_fc2(x)  # bias included inside
        return x  # [N, H]


class MoeExperts(nn.Module):
    """Container of local experts and token combine logic."""

    def __init__(self, config: MegatronConfig, input_size: int | None = None):
        super().__init__()
        self.config = config
        self.num_experts = int(config.num_experts)
        self.experts = nn.ModuleList(
            [Expert(config, input_size=input_size) for _ in range(self.num_experts)]
        )
        # Alias to match Megatron naming in checkpoints/converters.
        # self.local_experts = self.experts

    @torch.no_grad()
    def _compute_token_expert_masks(
        self, indices_col: torch.Tensor, num_experts: int
    ) -> list[torch.Tensor]:
        """Build list of boolean masks per expert e for a single top-k column of indices [T]."""
        # For each expert e, select tokens where indices_col == e
        masks = []
        for e in range(num_experts):
            masks.append(indices_col == e)
        return masks

    def forward(
        self, hidden_flat: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Combine expert outputs:
        - hidden_flat: [T, H]
        - probs: [T, k]
        - indices: [T, k]
        Returns:
        - output_flat: [T, H]
        """
        T, H = hidden_flat.shape
        k = indices.shape[1]
        device = hidden_flat.device
        dtype = hidden_flat.dtype

        output = torch.zeros((T, H), device=device, dtype=dtype)

        # For each top-k position, group tokens by expert and apply experts
        for j in range(k):
            idx_col = indices[:, j]  # [T]
            p_col = probs[:, j].to(dtype=dtype)  # [T]

            # For each expert, run the tokens belonging to it
            for e, expert in enumerate(self.experts):
                mask = idx_col == e
                if not mask.any():
                    continue
                tokens_e = hidden_flat[mask]  # [N_e, H]
                out_e = expert(tokens_e)  # [N_e, H]
                out_e = out_e * p_col[mask].unsqueeze(-1)  # weight by prob
                output[mask] += out_e

        return output  # [T, H]


class MoeMLP(nn.Module):
    """
    MoE MLP following Megatron's structure.
    Uses Megatron naming conventions for expert internals: linear_fc1, linear_fc2.

    MLP will take the input with h hidden state, project it to 4*h (or moe_ffn_hidden_size)
    within each expert, apply nonlinearity (SwiGLU optionally), and project back to h hidden.
    Tokens are routed to top-k experts and combined by router probabilities.

    Returns an output and a bias to be added to the output.
    To match HF integration and simplicity, we include bias inside expert linear layers,
    and return bias=None (skip_add_bias=True is the expected usage).
    """

    def __init__(self, config: MegatronConfig, input_size: int | None = None):
        super().__init__()
        self.config = config
        self.input_size = input_size if input_size is not None else config.hidden_size

        # Router
        self.router = TopKRouter(config)

        # Experts
        self.experts = MoeExperts(config, input_size=self.input_size)

        # Dropout after combining (to mirror dense MLP post-activation dropout)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))

    def forward(
        self, hidden_states: torch.Tensor, skip_add_bias: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        """Forward pass through the MoE MLP.

        Args:
            hidden_states: [seq_len, batch, hidden_size]
            skip_add_bias: If False, would return an additional bias like dense MLP.
                           Here we always embed bias in expert linear layers, so returns None.

        Returns:
            output: [seq_len, batch, hidden_size]
            bias: None (bias is already included above)
        """
        S, B, H = hidden_states.shape
        T = S * B
        hidden_flat = hidden_states.reshape(T, H)

        # Route tokens
        probs, indices = self.router(hidden_states)  # probs/indices: [T, k]
        # Compute per-expert outputs and combine by probs
        combined = self.experts(hidden_flat, probs, indices)

        # Optional dropout at the end (as in dense MLP after activation+fc2)
        combined = self.dropout(combined)

        output = combined.reshape(S, B, H)

        if not skip_add_bias:
            return output, None
        else:
            return output
