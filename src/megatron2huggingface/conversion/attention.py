"""Attention converter for Megatron-LM to HuggingFace Transformers.

This module provides modular conversion for attention blocks, using
Megatron naming conventions (linear_qkv, linear_proj) and keeping fused
QKV weights intact.
"""

from typing import Any
import logging

import torch
import torch.nn as nn


from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)

from megatron2huggingface.conversion.base import (
    BaseConverter,
    ConversionRegistry,
    extract_submodule_state_dict,
    add_prefix_to_state_dict,
)
from megatron2huggingface.conversion.config import megatron2transformer_config
from megatron2huggingface.modeling.attention import SelfAttention
from megatron2huggingface.configuration_megatron import MegatronConfig
from megatron2huggingface.conversion.layer_norm import LinearLayerNormConverter


logger = logging.getLogger(__name__)


@ConversionRegistry.register("attention")
class AttentionConverter(BaseConverter):
    """Converter for attention blocks using Megatron naming conventions."""

    def __init__(self, megatron_config: dict[str, Any]):
        super().__init__(megatron_config)
        self.linear_layer_norm_converter = LinearLayerNormConverter(megatron_config)

    def convert_weights(
        self, megatron_state: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert attention weights from Megatron format to our Megatron-style
        HuggingFace format.

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

        logger.debug("Converting attention layer ")

        # Convert QKV projection (keep fused)
        self._convert_qkv_projection(megatron_state, hf_state)

        # Convert output projection
        self._convert_output_projection(megatron_state, hf_state)

        return hf_state

    def _convert_qkv_projection(
        self,
        megatron_state: dict[str, torch.Tensor],
        hf_state: dict[str, torch.Tensor],
    ):
        """Convert QKV projection, keeping it fused with Megatron naming."""
        # Check for fused QKV projection (most common case)
        qkv_key = "linear_qkv"
        qkv_submodule_weights = extract_submodule_state_dict(megatron_state, qkv_key)
        qkv_converted = self.linear_layer_norm_converter.convert_weights(
            qkv_submodule_weights
        )
        hf_state.update(add_prefix_to_state_dict(qkv_converted, "linear_qkv"))

    def _convert_output_projection(
        self,
        megatron_state: dict[str, torch.Tensor],
        hf_state: dict[str, torch.Tensor],
    ):
        """Convert output projection weights using Megatron naming."""
        # Output projection weight (dense -> linear_proj)
        dense_weight_key = "linear_proj.weight"
        if dense_weight_key in megatron_state:
            hf_state["linear_proj.weight"] = megatron_state[dense_weight_key]
            logger.debug(
                f"Converted output projection weight: {megatron_state[dense_weight_key].shape}"
            )

        # Output projection bias
        dense_bias_key = "linear_proj.bias"
        if dense_bias_key in megatron_state:
            hf_state["linear_proj.bias"] = megatron_state[dense_bias_key]
            logger.debug(
                f"Converted output projection bias: {megatron_state[dense_bias_key].shape}"
            )

    def create_hf_module(self, config: MegatronConfig, **kwargs) -> SelfAttention:
        """Create a HuggingFace-compatible attention module using our Megatron-
        style implementation."""
        # Do not add new config fields; causal behavior is inferred in HF attention when no mask is provided.
        return SelfAttention(config)

    def create_megatron_module(self, **kwargs) -> nn.Module:
        # Import Megatron-LM modules
        from megatron.core.transformer.attention import (
            SelfAttention as MegatronSelfAttention,
            AttnMaskType,
        )

        """Create a Megatron attention module for comparison."""
        logger.debug("Creating Megatron SelfAttention module for layer 0")
        config = self.megatron_config

        # Define submodules for SelfAttention
        submodules = get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules

        # Create ModelCommProcessGroups (assuming default setup for now)
        # This needs to be properly initialized based on Megatron-LM's distributed setup
        # For testing purposes, we might need a dummy or simplified version
        # model_comm_pgs = ModelCommProcessGroups()  # This will use mpu.get_defaults()

        # Instantiate Megatron-LM's SelfAttention
        megatron_attention = MegatronSelfAttention(
            config=megatron2transformer_config(config),
            submodules=submodules,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,  # Use causal mask for GPT-style modeling
            model_comm_pgs=None,  # model_comm_pgs,
        )
        return megatron_attention

    # def get_expected_keys(self, config, layer_idx: int | None = None):
    #     """
    #     Expected input and output keys for attention conversion.
    #     Includes optional fused input layernorm parameters on linear_qkv if present.
    #     """
    #     # Megatron input keys (what we read)
    #     input_keys = [
    #         "linear_qkv.weight",
    #         "linear_proj.weight",
    #     ]
    #     if getattr(config, "add_qkv_bias", False):
    #         input_keys.append("linear_qkv.bias")
    #     if getattr(config, "add_bias_linear", False):
    #         input_keys.append("linear_proj.bias")

    #     # Optional fused LN on QKV
    #     # In Megatron, RMSNorm has only weight; LayerNorm has weight and bias.
    #     # We list both keys as optional so validator callers can treat them as optional.
    #     input_keys.extend(
    #         [
    #             "linear_qkv.layer_norm_weight",
    #             "linear_qkv.layer_norm_bias",
    #         ]
    #     )

    #     # HF output keys (what we produce for the HF module)
    #     output_keys = [
    #         "linear_qkv.weight",
    #         "linear_proj.weight",
    #     ]
    #     if getattr(config, "add_qkv_bias", False):
    #         output_keys.append("linear_qkv.bias")
    #     if getattr(config, "add_bias_linear", False):
    #         output_keys.append("linear_proj.bias")

    #     # Fused LN mapped names in our converted dict
    #     output_keys.extend(
    #         [
    #             "qkv_layer_norm.weight",
    #             "qkv_layer_norm.bias",
    #         ]
    #     )

    #     return input_keys, output_keys

    # def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    #     """Extract weights for the Megatron attention module."""

    #     return megatron_state
