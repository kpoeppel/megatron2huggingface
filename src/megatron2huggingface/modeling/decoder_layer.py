"""
Megatron-style decoder layer implementation for HuggingFace compatibility.
This is a 1:1 translation of Megatron's transformer layer without tensor parallelism.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.utils import logging

from ..configuration_megatron import MegatronConfig
from .attention import SelfAttention
from .mlp import MLP
from .layer_norm import RMSNorm, LayerNorm

logger = logging.get_logger(__name__)


class TransformerLayer(nn.Module):
    """
    A single transformer layer following Megatron's structure.
    Uses Megatron naming conventions: self_attention, mlp, input_layernorm, post_attention_layernorm.

    Transformer layer takes input with size [seq_len, batch, hidden_size] and returns an
    output of the same size.
    """

    def __init__(self, config: MegatronConfig):
        super().__init__()
        self.config = config

        # Layer norm at the beginning of the transformer layer.
        if config.normalization == "LayerNorm":
            self.input_layernorm = LayerNorm(
                config.hidden_size,
                eps=getattr(config, "layernorm_epsilon", 1e-5),
                bias=getattr(config, "add_bias_linear", True),
            )
        else:  # RMSNorm
            self.input_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "rms_norm_eps", 1e-6),
            )

        # Self attention.
        self.self_attention = SelfAttention(config)

        # Layer norm after the self attention.
        if config.normalization == "LayerNorm":
            self.post_attention_layernorm = LayerNorm(
                config.hidden_size,
                eps=getattr(config, "layernorm_epsilon", 1e-5),
                bias=getattr(config, "add_bias_linear", True),
            )
        else:  # RMSNorm
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "rms_norm_eps", 1e-6),
            )

        # MLP
        self.mlp = MLP(config)

        # Dropout
        self.hidden_dropout = nn.Dropout(getattr(config, "hidden_dropout", 0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer layer following Megatron's structure.

        Args:
            hidden_states: [seq_len, batch, hidden_size]
            attention_mask: Optional attention mask
            rotary_pos_emb: Optional rotary position embeddings
            attention_bias: Optional attention bias

        Returns:
            output: [seq_len, batch, hidden_size]
            bias: Optional output bias (None if add_bias_linear=False)
        """
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, attention_bias = self.self_attention(
            layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            attention_bias=attention_bias,
        )

        # Residual connection.
        if attention_bias is not None:
            attention_output = attention_output + attention_bias
        layernorm_input = hidden_states + attention_output

        # Layer norm after the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if mlp_bias is not None:
            mlp_output = mlp_output + mlp_bias

        output = layernorm_input + mlp_output

        # Return output and bias (bias is None if add_bias_linear=False)
        return output, None


class ParallelTransformerLayer(TransformerLayer):
    """
    Alias for TransformerLayer to maintain compatibility with Megatron naming.
    In actual Megatron, this would handle tensor parallelism, but we don't need that for inference.
    """

    pass


# Alias for compatibility with HuggingFace-style naming
MegatronDecoderLayer = TransformerLayer
