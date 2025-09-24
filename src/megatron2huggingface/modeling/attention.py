"""Megatron-style attention implementation for HuggingFace compatibility.

This is a 1:1 translation of Megatron's attention module without tensor
parallelism.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging

from ..configuration_megatron import MegatronConfig
from .layer_norm import LinearLayerNorm

logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(
    tensor, cos, sin, position_ids: torch.Tensor | None, rotary_interleaved=False
):
    """Apply rotary positional embedding to tensor."""
    # TODO: integrate position_ids
    if rotary_interleaved:
        # Interleaved format: [x0, x1, x2, x3, ...] -> [x0, x2, x1, x3, ...]
        x1, x2 = tensor[..., 0::2], tensor[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    else:
        # Standard format: split in half
        half_dim = tensor.shape[-1] // 2
        x1, x2 = tensor[..., :half_dim], tensor[..., half_dim:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CoreAttention(nn.Module):
    """Core attention computation following Megatron's structure."""

    def __init__(self, config: MegatronConfig):
        super().__init__()
        self.config = config

        self.hidden_size_per_attention_head = (
            config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = getattr(
            config, "num_query_groups", config.num_attention_heads
        )

        # Attention scaling
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if hasattr(config, "softmax_scale") and config.softmax_scale is not None:
            self.norm_factor = 1.0 / config.softmax_scale
        else:
            self.norm_factor = 1.0 / self.norm_factor

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of core attention using
        torch.nn.functional.scaled_dot_product_attention. This provides
        efficient backends including FlashAttention and FlexAttention.

        Args:
            query: [seq_len, batch_size, num_heads, head_dim]
            key: [seq_len, batch_size, num_query_groups, head_dim]
            value: [seq_len, batch_size, num_query_groups, head_dim]
            attention_mask: Optional attention mask
            attention_bias: Optional attention bias

        Returns:
            context: [seq_len, batch_size, num_heads, head_dim]
        """
        seq_len, batch_size, num_heads, head_dim = query.shape

        # Handle grouped query attention
        if self.num_query_groups < self.num_attention_heads:
            # Expand key and value to match query heads
            num_heads_per_group = self.num_attention_heads // self.num_query_groups
            key = key.repeat_interleave(num_heads_per_group, dim=2)
            value = value.repeat_interleave(num_heads_per_group, dim=2)

        # Transpose for attention computation: [seq_len, batch, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        query = query.transpose(0, 1).transpose(
            1, 2
        )  # [batch, heads, seq_len, head_dim]
        key = key.transpose(0, 1).transpose(
            1, 2
        )  # [batch, heads, kv_seq_len, head_dim]
        value = value.transpose(0, 1).transpose(
            1, 2
        )  # [batch, heads, kv_seq_len, head_dim]

        # Determine causal behavior without adding new config fields:
        # if no explicit attention_mask is provided, assume causal masking (GPT-style).
        is_causal = attention_mask is None

        # Prepare attention mask for SDPA
        # SDPA expects mask to be None, boolean mask, or additive mask
        attn_mask = None
        if attention_mask is not None or attention_bias is not None:
            # Combine attention_mask and attention_bias into a single additive mask
            if attention_mask is None:
                # attention_mask should be broadcastable to [batch, heads, seq_len, kv_seq_len]
                attn_mask = attention_bias
            elif attention_bias is None:
                # attention_bias should be broadcastable to [batch, heads, seq_len, kv_seq_len]
                attn_mask = attention_mask
            else:
                attn_mask = attention_bias + attention_mask

        # Use torch.nn.functional.scaled_dot_product_attention for efficient computation
        # This automatically selects the best backend (FlashAttention, FlexAttention, etc.)
        context = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=is_causal,  # Enforce causal when requested and no explicit mask provided
            # Use PyTorch default scaling (1/sqrt(d_k)) for parity with Megatron unless an explicit mask requires otherwise.
            scale=None,
        )

        # Transpose back: [batch, heads, seq_len, head_dim] -> [seq_len, batch, heads, head_dim]
        context = context.transpose(1, 2).transpose(0, 1)

        return context


class SelfAttention(nn.Module):
    """Self-attention layer following Megatron's structure.

    Uses Megatron naming conventions: linear_qkv, core_attention, linear_proj.
    """

    def __init__(
        self,
        config: MegatronConfig,
    ):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = getattr(
            config, "num_query_groups", config.num_attention_heads
        )
        self.hidden_size_per_attention_head = (
            config.kv_channels
            if config.kv_channels is not None
            else (config.hidden_size // config.num_attention_heads)
        )

        # Projection sizes
        self.query_projection_size = (
            self.hidden_size_per_attention_head * self.num_attention_heads
        )
        self.kv_projection_size = (
            (self.hidden_size_per_attention_head * self.num_query_groups)
            if config.group_query_attention
            else self.query_projection_size
        )

        # QKV projection - Megatron naming
        # In Megatron, add_qkv_bias controls QKV bias; add_bias_linear controls output/MLP biases.
        self.linear_qkv = LinearLayerNorm(
            self.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            bias=config.add_qkv_bias or config.add_bias_linear,
            ln_bias=config.add_qkv_bias or config.add_bias_linear,
            norm_type=config.normalization,
        )

        # Core attention computation
        self.core_attention = CoreAttention(config)

        # Output projection - Megatron naming
        self.linear_proj = nn.Linear(
            self.query_projection_size,
            self.hidden_size,
            bias=config.add_bias_linear,
        )

        # Optional Q/K layer norms (for stability)
        if getattr(config, "qk_layernorm", False):
            self.q_layernorm = nn.LayerNorm(
                self.hidden_size_per_attention_head,
                eps=getattr(config, "layernorm_epsilon", 1e-5),
            )
            self.k_layernorm = nn.LayerNorm(
                self.hidden_size_per_attention_head,
                eps=getattr(config, "layernorm_epsilon", 1e-5),
            )
        else:
            self.q_layernorm = None
            self.k_layernorm = None

    def get_query_key_value_tensors(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derives query, key and value tensors from hidden_states.

        Follows Megatron's exact tensor reshaping logic.
        """
        # QKV projection: [seq_len, batch, hidden] -> [seq_len, batch, qkv_size]
        mixed_qkv = self.linear_qkv(hidden_states)

        seq_len, batch_size = mixed_qkv.shape[:2]

        # Reshape to separate query groups: [seq_len, batch, qkv_size] -> [seq_len, batch, num_query_groups, group_size]
        new_tensor_shape = (
            seq_len,
            batch_size,
            self.num_query_groups,
            (self.num_attention_heads // self.num_query_groups + 2)
            * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Split into Q, K, V
        split_sizes = [
            (self.num_attention_heads // self.num_query_groups)
            * self.hidden_size_per_attention_head,  # Q
            self.hidden_size_per_attention_head,  # K
            self.hidden_size_per_attention_head,  # V
        ]

        query, key, value = torch.split(mixed_qkv, split_sizes, dim=3)

        # Reshape query: [seq_len, batch, num_query_groups, q_size] -> [seq_len, batch, num_heads, head_dim]
        query = query.reshape(
            seq_len,
            batch_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        )

        # Apply layer norms if configured
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass following Megatron's structure.

        Args:
            hidden_states: [seq_len, batch, hidden_size]
            attention_mask: Optional attention mask
            rotary_pos_emb: Optional rotary position embeddings
            attention_bias: Optional attention bias

        Returns:
            output: [seq_len, batch, hidden_size]
            bias: Optional output bias (None if add_bias_linear=False)
        """
        # Get Q, K, V tensors
        query, key, value = self.get_query_key_value_tensors(hidden_states)

        # Apply rotary position embeddings if provided
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                q_pos_emb, k_pos_emb = rotary_pos_emb
            else:
                q_pos_emb = k_pos_emb = rotary_pos_emb

            if q_pos_emb is not None:
                cos, sin = q_pos_emb
                query = apply_rotary_pos_emb(
                    query,
                    cos,
                    sin,
                    rotary_interleaved=getattr(
                        self.config, "rotary_interleaved", False
                    ),
                )
            if k_pos_emb is not None:
                cos, sin = k_pos_emb
                key = apply_rotary_pos_emb(
                    key,
                    cos,
                    sin,
                    rotary_interleaved=getattr(
                        self.config, "rotary_interleaved", False
                    ),
                )

        # Core attention computation
        context = self.core_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
        )

        # Reshape context for output projection: [seq_len, batch, num_heads, head_dim] -> [seq_len, batch, hidden_size]
        seq_len, batch_size = context.shape[:2]
        context = context.reshape(seq_len, batch_size, self.hidden_size)

        # Output projection
        output = self.linear_proj(context)

        # Return output and bias (bias is None if add_bias_linear=False)
        bias = None
        if hasattr(self.linear_proj, "bias") and self.linear_proj.bias is not None:
            bias = self.linear_proj.bias

        return output, bias, None
