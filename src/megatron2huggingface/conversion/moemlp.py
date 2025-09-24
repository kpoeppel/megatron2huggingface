"""MoE MLP converter for Megatron-LM to HuggingFace Transformers.

This converter maps Megatron-Core MoE MLP weights to our HuggingFace-compatible
MoeMLP module. Unlike dense MLP conversion, there is NO fused layernorm handled
here (layernorm is outside the MoE MLP module).

Expected Megatron-style inputs (per layer scope):
- Router:
    router.weight -> router.weight
- Experts (sequential experts layout):
    experts.local_experts.{i}.linear_fc1.weight -> experts.experts.{i}.linear_fc1.weight
    experts.local_experts.{i}.linear_fc1.bias   -> experts.experts.{i}.linear_fc1.bias
    experts.local_experts.{i}.linear_fc2.weight -> experts.experts.{i}.linear_fc2.weight
    experts.local_experts.{i}.linear_fc2.bias   -> experts.experts.{i}.linear_fc2.bias

Grouped-MLP / TE grouped layouts with consolidated expert weights are not
handled here.

Registration name: "moemlp"
"""

from typing import Any
import logging
import re

import torch

from megatron2huggingface.conversion.base import (
    BaseConverter,
    ConversionRegistry,
)
from megatron2huggingface.modeling.moemlp import MoeMLP
from megatron2huggingface.configuration_megatron import MegatronConfig

logger = logging.getLogger(__name__)


@ConversionRegistry.register("moemlp")
class MoeMLPConverter(BaseConverter):
    """Converter for MoE MLP blocks (router + sequential experts)."""

    def __init__(self, megatron_config: dict[str, Any]):
        super().__init__(megatron_config)

    def convert_weights(
        self, megatron_state: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert MoE MLP weights from Megatron format to our HF MoE MLP.

        Notes:
        - LayerNorm (if any) is outside this module; do not handle LN params here.
        - Only the router and expert linear layers are mapped.
        """
        hf_state: dict[str, torch.Tensor] = {}

        # 1) Router
        if "router.weight" in megatron_state:
            hf_state["router.weight"] = megatron_state["router.weight"]
            logger.debug(f"Converted router.weight: {hf_state['router.weight'].shape}")
        else:
            logger.warning(
                "router.weight not found in input state for MoE MLP conversion."
            )

        # 2) Experts (sequential)
        # Accept both explicit 'experts.local_experts' naming.
        # Map to our module's path: experts.experts.{i}.linear_fc*.*
        expert_key_regex = re.compile(r"^experts\.(linear_fc1|linear_fc2)\.weight(\d+)")

        found_any_expert = False
        for k, v in megatron_state.items():
            m = expert_key_regex.match(k)
            if m is None:
                logger.debug(f"Non-matched parameter: {k}")
                continue
            fc, idx = m.groups()
            dst_key = f"experts.experts.{idx}.{fc}.weight"
            hf_state[dst_key] = v
            found_any_expert = True
            logger.debug(f"Converted {k} -> {dst_key}: {tuple(v.shape)}")

        if not found_any_expert:
            expert_key_regex = re.compile(
                r"^experts.local_experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight"
            )

            found_any_expert = False
            for k, v in megatron_state.items():
                m = expert_key_regex.match(k)
                if m is None:
                    logger.debug(f"Non-matched parameter: {k}")
                    continue
                idx, fc = m.groups()
                dst_key = f"experts.experts.{idx}.{fc}.weight"
                hf_state[dst_key] = v
                found_any_expert = True
                logger.debug(f"Converted {k} -> {dst_key}: {tuple(v.shape)}")

        if not found_any_expert:
            logger.warning(
                "No sequential experts found under experts.experts.* in input state."
            )

        return hf_state

    def create_hf_module(self, layer_idx: int = 0, **kwargs) -> MoeMLP:
        """Create a HuggingFace-compatible MoeMLP module."""
        config = MegatronConfig(**self.megatron_config)
        return MoeMLP(config)

    def create_megatron_module(self, layer_idx: int = 0, **kwargs):
        """Instantiate the Megatron-Core MoE layer for output-matching checks.

        Assumes Transformer Engine is available, and builds the MoE layer spec from
        Megatron-Core using TE-backed modules. We extract the MoE MLP ModuleSpec
        from the GPT layer spec and construct the MoELayer with its submodules.
        """
        from megatron2huggingface.conversion.config import megatron2transformer_config
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_block_spec,
        )

        # Build Megatron-Core TransformerConfig from our stored config dict
        transformer_config = megatron2transformer_config(self.megatron_config)

        # Get a TE-backed GPT layer spec with MoE enabled
        layer_spec = get_gpt_decoder_block_spec(
            transformer_config,
            use_transformer_engine=True,
            normalization=self.megatron_config["normalization"],
        ).layer_specs[0]

        # Extract the MoE MLP ModuleSpec and instantiate the actual module
        moe_mlp_spec = layer_spec.submodules.mlp  # ModuleSpec for MoE MLP (MoELayer)
        moe_module_cls = moe_mlp_spec.module
        moe_submodules = (
            moe_mlp_spec.submodules
        )  # MoESubmodules for experts/shared_experts
        return moe_module_cls(config=transformer_config, submodules=moe_submodules)
