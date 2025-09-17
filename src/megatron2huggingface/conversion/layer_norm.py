"""Layer normalization converter for Megatron-LM to HuggingFace Transformers.

Handles RMSNorm and LayerNorm conversion.
"""

import logging
import torch
from typing import Any

from megatron2huggingface.conversion.base import BaseConverter

logger = logging.getLogger(__name__)


class LayerNormConverter(BaseConverter):
    """Converter for layer normalization layers."""

    def __init__(self, megatron_config: dict[str, Any]):
        """Initialize the layer norm converter."""
        super().__init__(megatron_config)

    def convert_weights(
        self,
        megatron_weights: dict[str, torch.Tensor],
        norm_type: str = "input_layernorm",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Convert Megatron layer norm weights to HuggingFace format.

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
                logger.warning(
                    "Found tensor parallel layer norm weights - this is unusual"
                )
                weight = weight[0]  # Take first replica as they should be identical

            hf_weights["weight"] = weight
            logger.info(f"Converted {norm_type} weight: {weight.shape}")

        # Convert bias (if present - RMSNorm typically doesn't have bias)
        bias_key = "bias"
        if bias_key in megatron_weights:
            bias = megatron_weights[bias_key]

            if isinstance(bias, list):
                logger.warning(
                    "Found tensor parallel layer norm bias - this is unusual"
                )
                bias = bias[0]  # Take first replica as they should be identical

            hf_weights["bias"] = bias
            logger.info(f"Converted {norm_type} bias: {bias.shape}")

        return hf_weights

    def create_megatron_module(self, **kwargs):
        """Instantiate a real Megatron-Core layer norm (no fallbacks/mocks)."""
        from megatron.core.models.backends import LocalSpecProvider
        from megatron.core.transformer.spec_utils import build_module
        from megatron2huggingface.conversion.config import megatron2transformer_config

        cfg_dict = dict(self.megatron_config)
        cfg = megatron2transformer_config(cfg_dict)
        hidden_size = cfg.hidden_size

        normalization = getattr(cfg, "normalization", "LayerNorm")
        rms = normalization == "RMSNorm"

        backend = LocalSpecProvider()
        norm_spec = backend.layer_norm(rms_norm=rms, for_qk=False)

        # Build the norm module with Megatron config and hidden size
        norm_module = build_module(norm_spec, hidden_size, config=cfg)
        return norm_module

    def create_hf_module(self, config: Any, **kwargs):
        """Instantiate the HF-side norm module matching Megatron semantics."""
        from megatron2huggingface.modeling.layer_norm import (
            RMSNorm as HF_RMSNorm,
            LayerNorm as HF_LayerNorm,
        )

        normalization = getattr(config, "normalization", "LayerNorm")
        eps = (
            getattr(config, "rms_norm_eps", None)
            or getattr(config, "layernorm_epsilon", None)
            or getattr(config, "norm_epsilon", 1e-5)
        )

        if normalization == "RMSNorm":
            return HF_RMSNorm(config.hidden_size, eps=eps)
        else:
            return HF_LayerNorm(config.hidden_size, eps=eps)

    # def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    #     """Extract weights for the Megatron attention module."""

    #     return megatron_state

    # def get_expected_keys(
    #     self, config: Any, layer_idx: int = None, norm_type: str = "input_layernorm"
    # ) -> Tuple[list, list]:
    #     """
    #     Get expected input and output keys for this converter.

    #     Args:
    #         config: Model configuration
    #         layer_idx: Layer index for transformer layers (None for final norm)
    #         norm_type: Type of normalization

    #     Returns:
    #         Tuple of (input_keys, output_keys)
    #     """
    #     input_keys = ["weight"]
    #     output_keys = ["weight"]

    #     # Add bias if the normalization type supports it
    #     normalization_type = getattr(config, "normalization", "LayerNorm")
    #     if normalization_type == "LayerNorm":  # LayerNorm has bias, RMSNorm typically doesn't
    #         input_keys.append("bias")
    #         output_keys.append("bias")

    #     return input_keys, output_keys


class LinearLayerNormConverter(BaseConverter):
    """Converter for layer normalization layers."""

    def __init__(self, megatron_config: dict[str, Any]):
        """Initialize the layer norm converter."""
        super().__init__(megatron_config)

    def convert_weights(
        self,
        megatron_weights: dict[str, torch.Tensor],
        norm_type: str = "input_layernorm",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Convert Megatron layer norm weights to HuggingFace format.

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
                logger.warning(
                    "Found tensor parallel layer norm weights - this is unusual"
                )
                weight = weight[0]  # Take first replica as they should be identical

            hf_weights["weight"] = weight
            logger.info(f"Converted {norm_type} weight: {weight.shape}")

        # Convert bias (if present - RMSNorm typically doesn't have bias)
        bias_key = "bias"
        if bias_key in megatron_weights:
            bias = megatron_weights[bias_key]

            if isinstance(bias, list):
                logger.warning(
                    "Found tensor parallel layer norm bias - this is unusual"
                )
                bias = bias[0]  # Take first replica as they should be identical

            hf_weights["bias"] = bias
            logger.info(f"Converted {norm_type} bias: {bias.shape}")

        weight_key = "layer_norm_weight"
        if weight_key in megatron_weights:
            weight = megatron_weights[weight_key]

            # Handle tensor parallelism - usually layer norms are not parallelized
            if isinstance(weight, list):
                logger.warning(
                    "Found tensor parallel layer norm weights - this is unusual"
                )
                weight = weight[0]  # Take first replica as they should be identical

            hf_weights["layer_norm_weight"] = weight
            logger.info(f"Converted {norm_type} layer_norm_weight: {weight.shape}")

        # Convert bias (if present - RMSNorm typically doesn't have bias)
        bias_key = "layer_norm_bias"
        if bias_key in megatron_weights:
            bias = megatron_weights[bias_key]

            if isinstance(bias, list):
                logger.warning(
                    "Found tensor parallel layer norm bias - this is unusual"
                )
                bias = bias[0]  # Take first replica as they should be identical

            hf_weights["layer_norm_bias"] = bias
            logger.info(f"Converted {norm_type} layer_norm_bias: {bias.shape}")

        return hf_weights

    def create_megatron_module(self, **kwargs):
        """Instantiate a real Megatron-Core layer norm (no fallbacks/mocks)."""
        from megatron.core.models.backends import TESpecProvider
        from megatron.core.transformer.spec_utils import build_module
        from megatron2huggingface.conversion.config import megatron2transformer_config

        cfg_dict = dict(self.megatron_config)
        cfg = megatron2transformer_config(cfg_dict)
        hidden_size = kwargs["hidden_size"]
        out_size = kwargs["out_size"]

        normalization = getattr(cfg, "normalization", "LayerNorm")
        rms = normalization == "RMSNorm"

        backend = TESpecProvider()
        linear_norm_spec = backend.column_parallel_layer_norm_linear(
            rms_norm=rms, for_qk=False
        )

        # Build the norm module with Megatron config and hidden size
        module = build_module(linear_norm_spec, hidden_size, out_size, config=cfg)
        return module

    def create_hf_module(self, config: Any, **kwargs):
        """Instantiate the HF-side norm module matching Megatron semantics."""
        from megatron2huggingface.modeling.layer_norm import LinearLayerNorm

        normalization = getattr(config, "normalization", "LayerNorm")
        eps = (
            getattr(config, "rms_norm_eps", None)
            or getattr(config, "layernorm_epsilon", None)
            or getattr(config, "norm_epsilon", 1e-5)
        )
        hidden_size = kwargs["hidden_size"]
        out_size = kwargs["out_size"]

        return LinearLayerNorm(hidden_size, out_size, norm_type=normalization, eps=eps)

    # def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    #     """Extract weights for the Megatron attention module."""

    #     return megatron_state

    # def get_expected_keys(
    #     self, config: Any, layer_idx: int = None, norm_type: str = "input_layernorm"
    # ) -> Tuple[list, list]:
    #     """
    #     Get expected input and output keys for this converter.

    #     Args:
    #         config: Model configuration
    #         layer_idx: Layer index for transformer layers (None for final norm)
    #         norm_type: Type of normalization

    #     Returns:
    #         Tuple of (input_keys, output_keys)
    #     """
    #     input_keys = ["weight"]
    #     output_keys = ["weight"]

    #     # Add bias if the normalization type supports it
    #     normalization_type = getattr(config, "normalization", "LayerNorm")
    #     if normalization_type == "LayerNorm":  # LayerNorm has bias, RMSNorm typically doesn't
    #         input_keys.append("bias")
    #         output_keys.append("bias")

    #     return input_keys, output_keys
