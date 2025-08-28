"""
MLP converter for Megatron-LM to HuggingFace Transformers.

This module provides modular conversion for MLP blocks, using Megatron naming
conventions (linear_fc1, linear_fc2) and handling gated linear units properly.
"""

from typing import Dict, Any, Optional
import logging

import torch
import torch.nn as nn

from megatron2huggingface.conversion.base import BaseConverter, ConversionRegistry
from megatron2huggingface.modeling.mlp import MLP
from megatron2huggingface.configuration_megatron import MegatronConfig

logger = logging.getLogger(__name__)


@ConversionRegistry.register("mlp")
class MLPConverter(BaseConverter):
    """Converter for MLP blocks using Megatron naming conventions."""

    def convert_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Convert MLP weights from Megatron format to our Megatron-style HuggingFace format.

        This converter maintains Megatron naming conventions:
        - linear_fc1: First linear layer (input -> intermediate)
        - linear_fc2: Second linear layer (intermediate -> output)

        Args:
            megatron_state: Megatron model state dictionary
            layer_idx: Layer index for the MLP block
            **kwargs: Additional conversion parameters

        Returns:
            Dictionary of converted weights in Megatron-style HuggingFace format
        """
        hf_state = {}

        logger.debug("Converting MLP layer")

        self._convert_fc1_layer(megatron_state, hf_state)

        self._convert_fc2_layer(megatron_state, hf_state)

        return hf_state

    def _convert_fc1_layer(
        self,
        megatron_state: Dict[str, torch.Tensor],
        hf_state: Dict[str, torch.Tensor],
    ):
        """Convert the first linear layer (input -> intermediate) using Megatron naming."""
        # First linear layer weight (dense_h_to_4h -> linear_fc1)
        fc1_weight_key = "linear_fc1.weight"
        if fc1_weight_key in megatron_state:
            hf_state["linear_fc1.weight"] = megatron_state[fc1_weight_key]
            logger.debug(f"Converted FC1 weight: {megatron_state[fc1_weight_key].shape}")
        else:
            logger.warning(f"Could not find FC1 weight: {fc1_weight_key}")

        # First linear layer bias
        fc1_bias_key = "linear_fc1.bias"
        if fc1_bias_key in megatron_state:
            hf_state["linear_fc1.bias"] = megatron_state[fc1_bias_key]
            logger.debug(f"Converted FC1 bias: {megatron_state[fc1_bias_key].shape}")

    def _convert_fc2_layer(
        self,
        megatron_state: Dict[str, torch.Tensor],
        hf_state: Dict[str, torch.Tensor],
    ):
        """Convert the second linear layer (intermediate -> output) using Megatron naming."""
        # Second linear layer weight (linear_fc2 -> linear_fc2)
        fc2_weight_key = "linear_fc2.weight"
        if fc2_weight_key in megatron_state:
            hf_state["linear_fc2.weight"] = megatron_state[fc2_weight_key]
            logger.debug(f"Converted FC2 weight: {megatron_state[fc2_weight_key].shape}")
        else:
            logger.warning(f"Could not find FC2 weight: {fc2_weight_key}")

        # Second linear layer bias
        fc2_bias_key = "linear_fc2.bias"
        if fc2_bias_key in megatron_state:
            hf_state["linear_fc2.bias"] = megatron_state[fc2_bias_key]
            logger.debug(f"Converted FC2 bias: {megatron_state[fc2_bias_key].shape}")

    def create_hf_module(self, config: MegatronConfig, layer_idx: int = 0, **kwargs) -> MLP:
        """Create a HuggingFace-compatible MLP module using our Megatron-style implementation."""
        return MLP(config)

    def create_megatron_module(self, layer_idx: int = 0, **kwargs):
        """Create a Megatron MLP module for comparison."""
        logger.debug("Creating fallback Megatron MLP implementation for testing")
        return self._create_fallback_megatron_mlp()

    def _create_fallback_megatron_mlp(self):
        """Create a fallback MLP implementation that matches Megatron's interface."""
        hidden_size = self.megatron_config.get("hidden_size", 768)
        ffn_hidden_size = self.megatron_config.get("ffn_hidden_size", 4 * hidden_size)
        add_bias = self.megatron_config.get("add_bias_linear", True)
        gated_linear_unit = self.megatron_config.get("gated_linear_unit", False)
        activation_function = self.megatron_config.get("activation_function", "gelu")

        # If gated linear unit, double the intermediate size
        if gated_linear_unit:
            ffn_hidden_size *= 2

        class FallbackMegatronMLP(nn.Module):
            """Fallback Megatron MLP that matches the expected interface."""

            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_size
                self.ffn_hidden_size = ffn_hidden_size
                self.gated_linear_unit = gated_linear_unit

                # Use Megatron naming: linear_fc1 and linear_fc2
                self.linear_fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=add_bias)
                self.linear_fc2 = nn.Linear(
                    ffn_hidden_size // 2 if gated_linear_unit else ffn_hidden_size, hidden_size, bias=add_bias
                )

                # Activation function
                if activation_function == "gelu":
                    self.activation_func = torch.nn.functional.gelu
                elif activation_function == "silu" or activation_function == "swish":
                    self.activation_func = torch.nn.functional.silu
                elif activation_function == "relu":
                    self.activation_func = torch.nn.functional.relu
                else:
                    self.activation_func = torch.nn.functional.gelu

            def forward(self, hidden_states, **kwargs):
                """Forward pass matching Megatron's expected interface."""
                # First linear transformation
                intermediate = self.linear_fc1(hidden_states)

                # Handle gated linear unit
                if self.gated_linear_unit:
                    # Split into gate and up projections
                    gate, up = torch.chunk(intermediate, 2, dim=-1)
                    # Apply activation to gate and multiply with up
                    intermediate = self.activation_func(gate) * up
                else:
                    # Standard activation
                    intermediate = self.activation_func(intermediate)

                # Second linear transformation
                output = self.linear_fc2(intermediate)

                # Return (output, bias) tuple to match Megatron interface
                bias = None
                if hasattr(self.linear_fc2, "bias") and self.linear_fc2.bias is not None:
                    bias = self.linear_fc2.bias

                return output, bias

        return FallbackMegatronMLP()

    def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Extract weights for the Megatron attention module."""

        return megatron_state


def test_mlp_conversion(
    hidden_size: int = 512,
    ffn_hidden_size: Optional[int] = None,
    gated_linear_unit: bool = False,
    activation_function: str = "gelu",
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
    debug: bool = False,
) -> bool:
    """
    Test MLP conversion with synthetic weights.

    Args:
        hidden_size: Hidden dimension size
        ffn_hidden_size: FFN hidden dimension size (defaults to 4 * hidden_size)
        gated_linear_unit: Whether to use gated linear unit
        activation_function: Activation function to use
        batch_size: Batch size for test input
        seq_len: Sequence length for test input
        device: Device to run test on
        debug: Whether to raise exceptions or return False

    Returns:
        True if test passes, False otherwise
    """
    try:
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        # Create synthetic Megatron config
        megatron_config = {
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "gated_linear_unit": gated_linear_unit,
            "activation_function": activation_function,
            "add_bias_linear": True,
            "hidden_dropout": 0.1,
            "num_layers": 2,
        }

        # Create converter
        converter = MLPConverter(megatron_config)

        # Create HuggingFace config
        hf_config = MegatronConfig(**megatron_config)

        # Create synthetic Megatron state dict

        megatron_state = {}

        # Calculate actual FFN size (doubled for gated linear unit)
        actual_ffn_size = ffn_hidden_size * 2 if gated_linear_unit else ffn_hidden_size

        # First linear layer (h -> 4h or h -> 8h for gated)
        megatron_state["linear_fc1.weight"] = torch.randn(actual_ffn_size, hidden_size, device=device)
        if megatron_config["add_bias_linear"]:
            megatron_state["linear_fc1.bias"] = torch.randn(actual_ffn_size, device=device)

        # Second linear layer (4h -> h or 4h -> h for gated, since gated splits the intermediate)
        megatron_state["linear_fc2.weight"] = torch.randn(hidden_size, ffn_hidden_size, device=device)
        if megatron_config["add_bias_linear"]:
            megatron_state["linear_fc2.bias"] = torch.randn(hidden_size, device=device)

        # Create test input
        test_input = torch.randn(batch_size, seq_len, hidden_size, device=device)

        # Test conversion
        results = converter.test_conversion(
            megatron_state=megatron_state,
            hf_config=hf_config,
            test_input=test_input,
        )

        logger.info(f"MLP conversion test results (gated: {gated_linear_unit}, activation: {activation_function}):")
        logger.info(f"  MSE: {results['mse']:.2e}")
        logger.info(f"  Max diff: {results['max_diff']:.2e}")
        logger.info(f"  Relative error: {results['relative_error']:.2e}")
        logger.info(f"  Test passed: {results['test_passed']}")

        return results["test_passed"]

    except Exception as e:
        if debug:
            raise
        logger.error(f"MLP conversion test failed: {e}")
        return False


def test_mlp_conversion_from_checkpoint(
    megatron_checkpoint_path: str,
    megatron_repo_path: str,
    batch_size: int = 2,
    seq_length: int = 128,
) -> Dict[str, Any]:
    """
    Test MLP conversion for a specific layer from actual checkpoint.

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
    converter = MLPConverter(megatron_config)

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

    logger.info("MLP conversion test results for layer:")
    logger.info(f"  MSE: {results['mse']:.2e}")
    logger.info(f"  Max diff: {results['max_diff']:.2e}")
    logger.info(f"  Relative error: {results['relative_error']:.2e}")
    logger.info(f"  Test passed: {results['test_passed']}")

    return results
