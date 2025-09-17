"""Layer normalization components for Megatron models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class RMSNorm(nn.Module):
    """RMSNorm implementation compatible with Megatron models."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize RMSNorm layer.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Normalized tensor of the same shape
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    """LayerNorm implementation compatible with Megatron models."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """Initialize LayerNorm layer.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm to the input.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Normalized tensor of the same shape
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype) + self.bias


class LinearLayerNorm(nn.Linear):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        eps: float = 1e-5,
        pre_norm: bool = True,
        bias: bool = True,
        ln_bias: bool = True,
        norm_type: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
    ):
        super().__init__(hidden_size, output_size, bias=bias)
        self.hidden_size = hidden_size
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size))
        if ln_bias and norm_type == "LayerNorm":
            self.layer_norm_bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.layer_norm_bias = None
        self.variance_epsilon = eps
        self.pre_norm = pre_norm
        self.norm_type = norm_type

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.pre_norm:
            hidden_states = nn.Linear.forward(self, hidden_states)
        if self.norm_type == "LayerNorm":
            hidden_states = F.layer_norm(
                hidden_states,
                normalized_shape=(hidden_states.shape[-1],),
                weight=self.layer_norm_weight,
                bias=self.layer_norm_bias,
                eps=self.variance_epsilon,
            )
        elif self.norm_type == "RMSNorm":
            hidden_states = F.rms_norm(
                hidden_states,
                normalized_shape=(hidden_states.shape[-1],),
                weight=self.layer_norm_weight,
                eps=self.variance_epsilon,
            )
        if self.pre_norm:
            hidden_states = nn.Linear.forward(self, hidden_states)

        return hidden_states
