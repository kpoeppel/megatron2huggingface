"""MLP converter for Megatron-LM to HuggingFace Transformers.

This module provides modular conversion for MLP blocks, using Megatron
naming conventions (linear_fc1, linear_fc2) and handling gated linear
units properly.
"""

from typing import Any
import logging

import torch

from megatron2huggingface.conversion.base import (
    BaseConverter,
    ConversionRegistry,
    extract_submodule_state_dict,
    add_prefix_to_state_dict,
)
from megatron2huggingface.modeling.mlp import MLP
from megatron2huggingface.configuration_megatron import MegatronConfig
from megatron2huggingface.conversion.layer_norm import LinearLayerNormConverter

logger = logging.getLogger(__name__)


@ConversionRegistry.register("mlp")
class MLPConverter(BaseConverter):
    """Converter for MLP blocks using Megatron naming conventions."""

    def __init__(self, megatron_config: dict[str, Any]):
        super().__init__(megatron_config)
        self.linear_layer_norm_converter = LinearLayerNormConverter(megatron_config)

    def convert_weights(
        self, megatron_state: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert MLP weights from Megatron format to our Megatron-style
        HuggingFace format.

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
        megatron_state: dict[str, torch.Tensor],
        hf_state: dict[str, torch.Tensor],
    ):
        fc1_prefix = "linear_fc1"
        fc1_submodule_weights = extract_submodule_state_dict(megatron_state, fc1_prefix)
        fc1_converted = self.linear_layer_norm_converter.convert_weights(
            fc1_submodule_weights
        )
        hf_state.update(add_prefix_to_state_dict(fc1_converted, "linear_fc1"))

    def _convert_fc2_layer(
        self,
        megatron_state: dict[str, torch.Tensor],
        hf_state: dict[str, torch.Tensor],
    ):
        """Convert the second linear layer (intermediate -> output) using
        Megatron naming."""
        # Second linear layer weight (linear_fc2 -> linear_fc2)
        fc2_weight_key = "linear_fc2.weight"
        if fc2_weight_key in megatron_state:
            hf_state["linear_fc2.weight"] = megatron_state[fc2_weight_key]
            logger.debug(
                f"Converted FC2 weight: {megatron_state[fc2_weight_key].shape}"
            )
        else:
            logger.warning(f"Could not find FC2 weight: {fc2_weight_key}")

        # Second linear layer bias
        fc2_bias_key = "linear_fc2.bias"
        if fc2_bias_key in megatron_state:
            hf_state["linear_fc2.bias"] = megatron_state[fc2_bias_key]
            logger.debug(f"Converted FC2 bias: {megatron_state[fc2_bias_key].shape}")

    def create_hf_module(self, layer_idx: int = 0, **kwargs) -> MLP:
        """Create a HuggingFace-compatible MLP module using our Megatron-style
        implementation."""
        config = MegatronConfig(**self.megatron_config)
        return MLP(config)

    def create_megatron_module(self, layer_idx: int = 0, **kwargs):
        """Create a real Megatron-Core MLP module for comparison (no
        fallbacks/mocks)."""
        from megatron.core.transformer.mlp import MLP as MegatronMLP
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron2huggingface.conversion.config import megatron2transformer_config

        # Build TransformerConfig from our dict
        transformer_config = megatron2transformer_config(self.megatron_config)

        submodules = (
            get_gpt_decoder_block_spec(
                transformer_config,
                use_transformer_engine=True,
                normalization=self.megatron_config["normalization"],
            )
            .layer_specs[0]
            .submodules.mlp.submodules
        )

        # Instantiate Megatron MLP with correct signature
        # - config: TransformerConfig
        # - submodules: MLPSubmodules (linear_fc1/linear_fc2)
        # - is_expert: False (standard dense MLP)
        # - input_size: optional; let it default to config.hidden_size
        # - ffn_hidden_size: optional; let MLP derive from config.ffn_hidden_size
        return MegatronMLP(
            config=transformer_config, submodules=submodules, is_expert=False
        )

    # def get_expected_keys(self, config, layer_idx: int | None = None):
    #     """
    #     Expected input and output keys for MLP conversion.
    #     Includes optional fused pre-MLP layernorm parameters on linear_fc1 if present.
    #     """
    #     # Megatron input keys (what we read)
    #     input_keys = [
    #         "linear_fc1.weight",
    #         "linear_fc2.weight",
    #     ]
    #     if getattr(config, "add_bias_linear", False):
    #         input_keys.extend(
    #             [
    #                 "linear_fc1.bias",
    #                 "linear_fc2.bias",
    #             ]
    #         )

    #     # Optional fused LN on FC1
    #     # In Megatron, RMSNorm has only weight; LayerNorm has weight and bias.
    #     input_keys.extend(
    #         [
    #             "linear_fc1.layer_norm_weight",
    #             "linear_fc1.layer_norm_bias",
    #         ]
    #     )

    #     # HF output keys (what we produce for the HF module)
    #     output_keys = [
    #         "linear_fc1.weight",
    #         "linear_fc2.weight",
    #     ]
    #     if getattr(config, "add_bias_linear", False):
    #         output_keys.extend(
    #             [
    #                 "linear_fc1.bias",
    #                 "linear_fc2.bias",
    #             ]
    #         )

    #     # Fused LN mapped names in our converted dict
    #     output_keys.extend(
    #         [
    #             "fc1_layer_norm.weight",
    #             "fc1_layer_norm.bias",
    #         ]
    #     )

    #     return input_keys, output_keys

    # def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    #     """Extract weights for the Megatron attention module."""

    #     return megatron_state
