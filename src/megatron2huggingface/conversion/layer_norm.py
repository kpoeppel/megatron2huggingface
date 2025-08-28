"""
Layer normalization converter for Megatron-LM to HuggingFace Transformers.
Handles RMSNorm and LayerNorm conversion.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from megatron2huggingface.conversion.base import BaseConverter

logger = logging.getLogger(__name__)


class LayerNormConverter(BaseConverter):
    """Converter for layer normalization layers."""

    def __init__(self, megatron_config: Dict[str, Any]):
        """Initialize the layer norm converter."""
        super().__init__(megatron_config)

    def convert_weights(
        self,
        megatron_weights: Dict[str, torch.Tensor],
        norm_type: str = "input_layernorm",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert Megatron layer norm weights to HuggingFace format.

        Args:
            megatron_weights: Dictionary containing Megatron weights
            config: Model configuration
            layer_idx: Layer index for transformer layers (None for final norm)
            norm_type: Type of normalization ("input_layernorm", "post_attention_layernorm", "final_layernorm")
            **kwargs: Additional conversion parameters

        Returns:
            Dictionary with converted HuggingFace weights
        """
        hf_weights = {}

        # Convert weight (for RMSNorm, this is the only parameter)
        weight_key = "weight"
        if weight_key in megatron_weights:
            weight = megatron_weights[weight_key]

            # Handle tensor parallelism - usually layer norms are not parallelized
            if isinstance(weight, list):
                logger.warning("Found tensor parallel layer norm weights - this is unusual")
                weight = weight[0]  # Take first replica as they should be identical

            hf_weights["weight"] = weight
            logger.info(f"Converted {norm_type} weight: {weight.shape}")

        # Convert bias (if present - RMSNorm typically doesn't have bias)
        bias_key = "bias"
        if bias_key in megatron_weights:
            bias = megatron_weights[bias_key]

            if isinstance(bias, list):
                logger.warning("Found tensor parallel layer norm bias - this is unusual")
                bias = bias[0]  # Take first replica as they should be identical

            hf_weights["bias"] = bias
            logger.info(f"Converted {norm_type} bias: {bias.shape}")

        return hf_weights

    def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Extract weights for the Megatron attention module."""

        return megatron_state

    def get_expected_keys(
        self, config: Any, layer_idx: int = None, norm_type: str = "input_layernorm"
    ) -> Tuple[list, list]:
        """
        Get expected input and output keys for this converter.

        Args:
            config: Model configuration
            layer_idx: Layer index for transformer layers (None for final norm)
            norm_type: Type of normalization

        Returns:
            Tuple of (input_keys, output_keys)
        """
        input_keys = ["weight"]
        output_keys = ["weight"]

        # Add bias if the normalization type supports it
        normalization_type = getattr(config, "normalization", "LayerNorm")
        if normalization_type == "LayerNorm":  # LayerNorm has bias, RMSNorm typically doesn't
            input_keys.append("bias")
            output_keys.append("bias")

        return input_keys, output_keys


class FallbackMegatronRMSNorm(nn.Module):
    """
    Fallback Megatron-style RMSNorm implementation for testing.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Normalized tensor [batch_size, seq_len, hidden_size]
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FallbackMegatronLayerNorm(nn.Module):
    """
    Fallback Megatron-style LayerNorm implementation for testing.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm to the input.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Normalized tensor [batch_size, seq_len, hidden_size]
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype) + self.bias


def test_layer_norm_conversion(
    hidden_size: int = 4096, batch_size: int = 2, seq_len: int = 10, norm_type: str = "RMSNorm", device: str = "cpu"
) -> bool:
    """
    Test layer norm conversion between Megatron and HuggingFace formats.

    Args:
        hidden_size: Hidden dimension size
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        norm_type: Type of normalization ("RMSNorm" or "LayerNorm")
        device: Device to run test on

    Returns:
        True if conversion test passes
    """
    logger.info(f"Testing {norm_type} conversion...")

    # Create fallback Megatron layer norm
    if norm_type == "RMSNorm":
        megatron_norm = FallbackMegatronRMSNorm(hidden_size).to(device)
        hf_norm = FallbackMegatronRMSNorm(hidden_size).to(device)  # Use same implementation for comparison
    else:
        megatron_norm = FallbackMegatronLayerNorm(hidden_size).to(device)
        hf_norm = nn.LayerNorm(hidden_size).to(device)

    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Get Megatron output
    with torch.no_grad():
        megatron_output = megatron_norm(hidden_states)

    # Prepare weights for conversion
    megatron_weights = {"weight": megatron_norm.weight}

    if hasattr(megatron_norm, "bias"):
        megatron_weights["bias"] = megatron_norm.bias

    # Convert weights
    megatron_config = {
        "hidden_size": hidden_size,
        "normalization": norm_type,
    }
    converter = LayerNormConverter(megatron_config)
    hf_weights = converter.convert_weights(megatron_weights, layer_idx=0, norm_type="input_layernorm")

    # Apply converted weights to HuggingFace norm
    hf_norm.weight.data = hf_weights["weight"]
    if "bias" in hf_weights:
        hf_norm.bias.data = hf_weights["bias"]

    # Get HuggingFace output
    with torch.no_grad():
        hf_output = hf_norm(hidden_states)

    # Compare outputs
    max_diff = torch.max(torch.abs(megatron_output - hf_output)).item()
    logger.info(f"Maximum difference between outputs: {max_diff}")

    # Test should pass with very small numerical differences
    success = max_diff < 1e-5

    if success:
        logger.info("✓ Layer norm conversion test passed!")
    else:
        logger.error(f"✗ Layer norm conversion test failed! Max diff: {max_diff}")

    return success


if __name__ == "__main__":
    # Test RMSNorm
    print("Testing RMSNorm conversion:")
    test_layer_norm_conversion(norm_type="RMSNorm")

    print("\nTesting LayerNorm conversion:")
    test_layer_norm_conversion(norm_type="LayerNorm")
