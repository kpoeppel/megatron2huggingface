"""
Attention converter for Megatron-LM to HuggingFace Transformers.

This module provides modular conversion for attention blocks, using Megatron naming
conventions (linear_qkv, linear_proj) and keeping fused QKV weights intact.
"""

from typing import Dict, Any, Optional
import logging

import torch
import torch.nn as nn

from megatron2huggingface.conversion.base import BaseConverter, ConversionRegistry
from megatron2huggingface.modeling.attention import SelfAttention
from megatron2huggingface.configuration_megatron import MegatronConfig

logger = logging.getLogger(__name__)


@ConversionRegistry.register("attention")
class AttentionConverter(BaseConverter):
    """Converter for attention blocks using Megatron naming conventions."""

    def convert_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Convert attention weights from Megatron format to our Megatron-style HuggingFace format.

        This converter maintains Megatron naming conventions:
        - linear_qkv: Fused QKV projection (kept as-is)
        - linear_proj: Output projection

        Args:
            megatron_state: Megatron model state dictionary
            **kwargs: Additional conversion parameters

        Returns:
            Dictionary of converted weights in Megatron-style HuggingFace format
        """
        hf_state = {}

        logger.debug(f"Converting attention layer ")

        # Convert QKV projection (keep fused)
        self._convert_qkv_projection(megatron_state, hf_state)

        # Convert output projection
        self._convert_output_projection(megatron_state, hf_state)

        return hf_state

    def _convert_qkv_projection(
        self,
        megatron_state: Dict[str, torch.Tensor],
        hf_state: Dict[str, torch.Tensor],
    ):
        """Convert QKV projection, keeping it fused with Megatron naming."""
        # Check for fused QKV projection (most common case)
        qkv_weight_key = f"linear_qkv.weight"
        if qkv_weight_key in megatron_state:
            # Keep the fused QKV weight as-is, just rename to linear_qkv
            hf_state[f"linear_qkv.weight"] = megatron_state[qkv_weight_key]
            logger.debug(f"Converted fused QKV weight: {megatron_state[qkv_weight_key].shape}")

        # Handle QKV bias
        qkv_bias_key = f"linear_qkv.bias"
        if qkv_bias_key in megatron_state:
            # Keep the fused QKV bias as-is
            hf_state[f"linear_qkv.bias"] = megatron_state[qkv_bias_key]
            logger.debug(f"Converted fused QKV bias: {megatron_state[qkv_bias_key].shape}")

    def _convert_output_projection(
        self,
        megatron_state: Dict[str, torch.Tensor],
        hf_state: Dict[str, torch.Tensor],
    ):
        """Convert output projection weights using Megatron naming."""
        # Output projection weight (dense -> linear_proj)
        dense_weight_key = f"linear_proj.weight"
        if dense_weight_key in megatron_state:
            hf_state[f"linear_proj.weight"] = megatron_state[dense_weight_key]
            logger.debug(f"Converted output projection weight: {megatron_state[dense_weight_key].shape}")

        # Output projection bias
        dense_bias_key = f"linear_proj.bias"
        if dense_bias_key in megatron_state:
            hf_state[f"linear_proj.bias"] = megatron_state[dense_bias_key]
            logger.debug(f"Converted output projection bias: {megatron_state[dense_bias_key].shape}")

    def create_hf_module(self, config: MegatronConfig, **kwargs) -> SelfAttention:
        """Create a HuggingFace-compatible attention module using our Megatron-style implementation."""
        return SelfAttention(config)

    def create_megatron_module(self, **kwargs):
        """Create a Megatron attention module for comparison."""
        logger.debug("Creating fallback Megatron attention implementation for testing")
        return self._create_fallback_megatron_attention()

    def _create_fallback_megatron_attention(self):
        """Create a fallback attention implementation that matches Megatron's interface."""
        hidden_size = self.megatron_config.get("hidden_size", 768)
        num_heads = self.megatron_config.get("num_attention_heads", 12)
        kv_channels = self.megatron_config.get("kv_channels", hidden_size // num_heads)
        num_query_groups = self.megatron_config.get("num_query_groups", num_heads)
        add_bias = self.megatron_config.get("add_bias_linear", True)
        add_qkv_bias = self.megatron_config.get("add_qkv_bias", False)
        attention_dropout = self.megatron_config.get("attention_dropout", 0.1)

        class FallbackMegatronAttention(nn.Module):
            """Fallback Megatron attention that matches the expected interface."""

            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = kv_channels
                self.num_key_value_heads = num_query_groups

                # Calculate projection sizes
                q_size = num_heads * kv_channels
                kv_size = num_query_groups * kv_channels
                total_qkv_size = q_size + 2 * kv_size

                # Use Megatron naming: linear_qkv and linear_proj
                self.linear_qkv = nn.Linear(hidden_size, total_qkv_size, bias=add_qkv_bias)
                self.linear_proj = nn.Linear(q_size, hidden_size, bias=add_bias)
                self.attention_dropout = nn.Dropout(attention_dropout)

            def forward(self, hidden_states, attention_mask=None, **kwargs):
                """Forward pass matching Megatron's expected interface."""
                batch_size, seq_len, _ = hidden_states.shape

                # QKV projection
                qkv = self.linear_qkv(hidden_states)

                # Split QKV
                q_size = self.num_heads * self.head_dim
                kv_size = self.num_key_value_heads * self.head_dim

                query = qkv[:, :, :q_size]
                key = qkv[:, :, q_size : q_size + kv_size]
                value = qkv[:, :, q_size + kv_size :]

                # Reshape for attention computation
                query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                # Handle grouped query attention
                if self.num_key_value_heads != self.num_heads:
                    key = key.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
                    value = value.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

                # Attention computation
                scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)

                if attention_mask is not None:
                    scores = scores + attention_mask

                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = self.attention_dropout(attn_weights)

                attn_output = torch.matmul(attn_weights, value)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

                # Output projection
                output = self.linear_proj(attn_output)

                # Return (output, bias) tuple to match Megatron interface
                bias = None
                if hasattr(self.linear_proj, "bias") and self.linear_proj.bias is not None:
                    bias = self.linear_proj.bias

                return output, bias

        return FallbackMegatronAttention()

    def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Extract weights for the Megatron attention module."""

        return megatron_state


def test_attention_conversion(
    hidden_size: int = 512,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 8,
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
    debug: bool = False,
) -> bool:
    """
    Test attention conversion with synthetic weights.

    Args:
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        batch_size: Batch size for test input
        seq_len: Sequence length for test input
        device: Device to run test on
        debug: Whether to raise exceptions or return False

    Returns:
        True if test passes, False otherwise
    """
    try:
        # Create synthetic Megatron config
        kv_channels = hidden_size // num_attention_heads
        megatron_config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "kv_channels": kv_channels,
            "num_query_groups": num_key_value_heads,
            "max_position_embeddings": 128,
            "add_bias_linear": True,
            "add_qkv_bias": False,
            "attention_dropout": 0.1,
            "num_layers": 2,
        }

        # Create converter
        converter = AttentionConverter(megatron_config)

        # Create HuggingFace config
        hf_config = MegatronConfig(**megatron_config)

        # Create synthetic Megatron state dict
        megatron_state = {}

        # QKV projection (fused)
        q_size = num_attention_heads * kv_channels
        kv_size = num_key_value_heads * kv_channels
        qkv_size = q_size + 2 * kv_size

        megatron_state["linear_qkv.weight"] = torch.randn(qkv_size, hidden_size, device=device)
        if megatron_config["add_qkv_bias"]:
            megatron_state["linear_qkv.bias"] = torch.randn(qkv_size, device=device)

        # Output projection
        megatron_state["linear_proj.weight"] = torch.randn(hidden_size, q_size, device=device)
        if megatron_config["add_bias_linear"]:
            megatron_state["linear_proj.bias"] = torch.randn(hidden_size, device=device)

        # Create test input
        test_input = torch.randn(batch_size, seq_len, hidden_size, device=device)

        # Test conversion
        results = converter.test_conversion(megatron_state=megatron_state, hf_config=hf_config, test_input=test_input)

        logger.info(
            f"Attention conversion test results (heads: {num_attention_heads}, kv_heads: {num_key_value_heads}):"
        )
        logger.info(f"  MSE: {results['mse']:.2e}")
        logger.info(f"  Max diff: {results['max_diff']:.2e}")
        logger.info(f"  Relative error: {results['relative_error']:.2e}")
        logger.info(f"  Test passed: {results['test_passed']}")

        return results["test_passed"]

    except Exception as e:
        if debug:
            raise
        logger.error(f"Attention conversion test failed: {e}")
        return False


def test_attention_conversion_from_checkpoint(
    megatron_checkpoint_path: str,
    megatron_repo_path: str,
    batch_size: int = 2,
    seq_length: int = 128,
) -> Dict[str, Any]:
    """
    Test attention conversion for a specific layer from actual checkpoint.

    Args:
        megatron_checkpoint_path: Path to Megatron checkpoint
        megatron_repo_path: Path to Megatron-LM repository
        batch_size: Batch size for test input
        seq_length: Sequence length for test input

    Returns:
        Test results dictionary
    """
    from .base import MegatronCheckpointLoader

    # Load checkpoint
    loader = MegatronCheckpointLoader(megatron_checkpoint_path, megatron_repo_path)
    checkpoint = loader.load_distributed_checkpoint()
    megatron_config = loader.get_model_config(checkpoint)

    # Create converter
    converter = AttentionConverter(megatron_config)

    # Create HuggingFace config
    hf_config = MegatronConfig(**megatron_config)

    # Create test input
    hidden_size = megatron_config.get("hidden_size", 768)
    test_input = torch.randn(batch_size, seq_length, hidden_size)

    # Test conversion
    results = converter.test_conversion(
        megatron_state=checkpoint["model"],
        hf_config=hf_config,
        test_input=test_input,
    )

    logger.info("Attention conversion test results for layer:")
    logger.info(f"  MSE: {results['mse']:.2e}")
    logger.info(f"  Max diff: {results['max_diff']:.2e}")
    logger.info(f"  Relative error: {results['relative_error']:.2e}")
    logger.info(f"  Test passed: {results['test_passed']}")

    return results
