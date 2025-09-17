"""Megatron-style MLP implementation for HuggingFace compatibility.

This is a 1:1 translation of Megatron's MLP module without tensor
parallelism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging

from ..configuration_megatron import MegatronConfig
from .layer_norm import LinearLayerNorm

logger = logging.get_logger(__name__)


class MLP(nn.Module):
    """
    MLP following Megatron's structure.
    Uses Megatron naming conventions: linear_fc1, linear_fc2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.
    """

    def __init__(self, config: MegatronConfig, input_size: int | None = None):
        super().__init__()
        self.config = config

        # Input size defaults to hidden_size
        self.input_size = input_size if input_size is not None else config.hidden_size

        # FFN hidden size
        ffn_hidden_size = config.ffn_hidden_size

        # First linear layer - Megatron naming
        self.linear_fc1 = LinearLayerNorm(
            self.input_size,
            ffn_hidden_size * 2 if config.gated_linear_unit else ffn_hidden_size,
            bias=config.add_bias_linear,
            ln_bias=config.add_bias_linear,
            norm_type=config.normalization,
        )

        # Activation function
        if hasattr(config, "activation_function"):
            if isinstance(config.activation_function, str):
                self.activation_func = ACT2FN[config.activation_function]
            else:
                self.activation_func = config.activation_function
        else:
            # Default to GeLU if not specified
            self.activation_func = F.gelu

        # Second linear layer - Megatron naming
        self.linear_fc2 = nn.Linear(
            ffn_hidden_size,
            config.hidden_size,
            bias=getattr(config, "add_bias_linear", True),
        )

        # Dropout
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))

    def forward(
        self, hidden_states: torch.Tensor, skip_add_bias: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the MLP block following Megatron's structure.

        Args:
            hidden_states: [seq_len, batch, hidden_size]

        Returns:
            output: [seq_len, batch, hidden_size]
            bias: Optional output bias (None if add_bias_linear=False)
        """
        # First linear transformation: [seq_len, batch, hidden] -> [seq_len, batch, ffn_hidden]
        intermediate_parallel = self.linear_fc1(hidden_states)

        # Handle gated linear unit (SwiGLU, GeGLU, etc.)
        if getattr(self.config, "gated_linear_unit", False):
            # Split the intermediate representation in half for gating
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.activation_func(x[0]) * x[1]

            intermediate_parallel = glu(intermediate_parallel)
        else:
            # Standard activation
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # Apply dropout
        intermediate_parallel = self.dropout(intermediate_parallel)

        # Second linear transformation: [seq_len, batch, ffn_hidden] -> [seq_len, batch, hidden]
        output = self.linear_fc2(intermediate_parallel)

        # Return output and bias (bias is None if add_bias_linear=False)
        bias = None
        if hasattr(self.linear_fc2, "bias") and self.linear_fc2.bias is not None:
            bias = self.linear_fc2.bias

        if not skip_add_bias:
            return output, bias
        else:
            return output


class SwiGLUMLP(MLP):
    """SwiGLU MLP variant following Megatron's structure.

    This is a specialized version that explicitly uses SwiGLU
    activation.
    """

    def __init__(self, config: MegatronConfig, input_size: int | None = None):
        # Force gated linear unit and SiLU activation for SwiGLU
        config_copy = config.__class__(**config.__dict__)
        config_copy.gated_linear_unit = True
        config_copy.activation_function = "silu"

        super().__init__(config_copy, input_size)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        SwiGLU forward pass: SwiGLU(x) = Swish(xW1) ⊙ (xW2)
        where ⊙ denotes element-wise multiplication.
        """
        # First linear transformation
        gate_proj = self.linear_fc1(hidden_states)

        # Split into gate and up projections
        gate, up = torch.chunk(gate_proj, 2, dim=-1)

        # Apply SwiGLU: Swish(gate) * up
        intermediate = F.silu(gate) * up

        # Apply dropout
        intermediate = self.dropout(intermediate)

        # Second linear transformation
        output = self.linear_fc2(intermediate)

        # Return output and bias
        bias = None
        if hasattr(self.linear_fc2, "bias") and self.linear_fc2.bias is not None:
            bias = self.linear_fc2.bias

        return output, bias


class GeGLUMLP(MLP):
    """GeGLU MLP variant following Megatron's structure.

    This is a specialized version that explicitly uses GeGLU activation.
    """

    def __init__(self, config: MegatronConfig, input_size: int | None = None):
        # Force gated linear unit and GeLU activation for GeGLU
        config_copy = config.__class__(**config.__dict__)
        config_copy.gated_linear_unit = True
        config_copy.activation_function = "gelu"

        super().__init__(config_copy, input_size)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        GeGLU forward pass: GeGLU(x) = GELU(xW1) ⊙ (xW2)
        where ⊙ denotes element-wise multiplication.
        """
        # First linear transformation
        gate_proj = self.linear_fc1(hidden_states)

        # Split into gate and up projections
        gate, up = torch.chunk(gate_proj, 2, dim=-1)

        # Apply GeGLU: GELU(gate) * up
        intermediate = F.gelu(gate) * up

        # Apply dropout
        intermediate = self.dropout(intermediate)

        # Second linear transformation
        output = self.linear_fc2(intermediate)

        # Return output and bias
        bias = None
        if hasattr(self.linear_fc2, "bias") and self.linear_fc2.bias is not None:
            bias = self.linear_fc2.bias

        return output, bias
