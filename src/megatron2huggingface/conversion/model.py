"""Full model converter for Megatron-LM to HuggingFace Transformers.

Orchestrates all component converters to convert complete models. Uses
proper Megatron naming conventions and works with real megatron_core
models.
"""

import logging
import torch
import torch.nn as nn
from typing import Any

from ..configuration_megatron import MegatronConfig
from ..modeling_megatron import MegatronForCausalLM
from .base import BaseConverter, extract_submodule_state_dict, add_prefix_to_state_dict
from .embedding import EmbeddingConverter
from .attention import AttentionConverter
from .mlp import MLPConverter
from .layer_norm import LayerNormConverter

logger = logging.getLogger(__name__)


class ModelConverter(BaseConverter):
    """Full model converter that orchestrates all component converters.

    Uses proper Megatron naming conventions and modular architecture.
    """

    def __init__(self, megatron_config: dict[str, Any]):
        super().__init__(megatron_config)
        self.embedding_converter = EmbeddingConverter(megatron_config)
        self.attention_converter = AttentionConverter(megatron_config)
        self.mlp_converter = MLPConverter(megatron_config)
        self.layer_norm_converter = LayerNormConverter(megatron_config)

    def convert_weights(
        self, megatron_state_dict: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert complete Megatron model weights to our Megatron-style
        format.

        This converts from megatron_core naming to our consistent Megatron naming:
        - Uses linear_qkv, linear_proj for attention
        - Uses linear_fc1, linear_fc2 for MLP
        - Maintains proper module hierarchy
        - Uses proper prefix handling for submodule converters

        Args:
            megatron_state_dict: Dictionary containing Megatron weights from megatron_core
            **kwargs: Additional conversion parameters

        Returns:
            Dictionary with converted weights using our Megatron naming conventions
        """
        converted_weights = {}

        logger.info("Starting full model conversion...")
        logger.info(f"Input state dict has {len(megatron_state_dict)} keys")

        # Convert embeddings
        logger.info("Converting embeddings...")
        # Extract embedding weights (typically under "embedding" prefix)
        embedding_prefix = "embedding"
        embedding_submodule_weights = extract_submodule_state_dict(
            megatron_state_dict, embedding_prefix
        )

        embedding_converted = self.embedding_converter.convert_weights(
            embedding_submodule_weights, **kwargs
        )
        converted_weights.update(add_prefix_to_state_dict(embedding_converted, ""))

        # Convert transformer layers
        num_layers = self.megatron_config["num_layers"]
        logger.info(f"Converting {num_layers} transformer layers...")

        for layer_idx in range(num_layers):
            logger.info(f"Converting layer {layer_idx}...")
            layer_prefix = f"decoder.layers.{layer_idx}"

            # Convert attention
            attention_prefix = f"{layer_prefix}.self_attention"
            attention_submodule_weights = extract_submodule_state_dict(
                megatron_state_dict, attention_prefix
            )
            if attention_submodule_weights:
                attention_converted = self.attention_converter.convert_weights(
                    attention_submodule_weights, layer_idx=layer_idx, **kwargs
                )
                converted_weights.update(
                    add_prefix_to_state_dict(
                        attention_converted, f"layers.{layer_idx}.self_attn"
                    )
                )

            # Convert MLP
            mlp_prefix = f"{layer_prefix}.mlp"
            mlp_submodule_weights = extract_submodule_state_dict(
                megatron_state_dict, mlp_prefix
            )
            if mlp_submodule_weights:
                mlp_converted = self.mlp_converter.convert_weights(
                    mlp_submodule_weights, layer_idx=layer_idx, **kwargs
                )
                converted_weights.update(
                    add_prefix_to_state_dict(mlp_converted, f"layers.{layer_idx}.mlp")
                )

        # Convert final layer norm
        logger.info("Converting final layer norm...")
        final_norm_prefix = "decoder.final_layernorm"
        final_norm_submodule_weights = extract_submodule_state_dict(
            megatron_state_dict, final_norm_prefix
        )
        if final_norm_submodule_weights:
            final_norm_converted = self.layer_norm_converter.convert_weights(
                final_norm_submodule_weights,
                layer_idx=None,
                norm_type="final_layernorm",
                **kwargs,
            )
            converted_weights.update(
                add_prefix_to_state_dict(final_norm_converted, "norm")
            )

        converted_weights = add_prefix_to_state_dict(converted_weights, "model")

        output_layer_weight = None
        output_layer_weight = megatron_state_dict["output_layer.weight"]
        logger.info("Found output layer at key: output_layer.weight")

        if output_layer_weight is not None:
            logger.info("Converting language modeling head...")

            # Handle tensor parallelism if needed
            if isinstance(output_layer_weight, list):
                logger.info("Concatenating tensor parallel LM head weights")
                output_layer_weight = torch.cat(output_layer_weight, dim=0)

            converted_weights["lm_head.weight"] = output_layer_weight
            logger.info(f"Converted LM head: {output_layer_weight.shape}")
        else:
            logger.warning("No output layer found in checkpoint")

        logger.info(
            f"Model conversion complete! Converted {len(converted_weights)} weight tensors."
        )
        return converted_weights

    def create_hf_module(self, **kwargs) -> nn.Module:
        return MegatronForCausalLM(MegatronConfig(**self.megatron_config))

    # def get_expected_keys(self, config: dict[str, Any]) -> Tuple[list, list]:
    #     """
    #     Get expected input and output keys for the full model.

    #     Args:
    #         config: Model configuration

    #     Returns:
    #         Tuple of (input_keys, output_keys)
    #     """
    #     input_keys = []
    #     output_keys = []

    #     # Embedding keys
    #     emb_input, emb_output = self.embedding_converter.get_expected_keys(config)
    #     input_keys.extend(emb_input)
    #     output_keys.extend(emb_output)

    #     # Layer keys
    #     for layer_idx in range(config["num_layers"]):
    #         # Attention
    #         attn_input, attn_output = self.attention_converter.get_expected_keys(config, layer_idx=layer_idx)
    #         input_keys.extend(attn_input)
    #         output_keys.extend(attn_output)

    #         # MLP
    #         mlp_input, mlp_output = self.mlp_converter.get_expected_keys(config, layer_idx=layer_idx)
    #         input_keys.extend(mlp_input)
    #         output_keys.extend(mlp_output)

    #     # Final layer norm
    #     final_norm_input, final_norm_output = self.layer_norm_converter.get_expected_keys(
    #         config, layer_idx=None, norm_type="final_layernorm"
    #     )
    #     input_keys.extend(final_norm_input)
    #     output_keys.extend(final_norm_output)

    #     # Language modeling head
    #     input_keys.extend(["output_layer.weight", "lm_head.weight", "output_layer.weight"])
    #     output_keys.append("lm_head.weight")

    #     return input_keys, output_keys

    def create_megatron_module(self, **kwargs) -> nn.Module:
        """Create a Megatron model using megatron_core for testing purposes.
        This creates a real Megatron model that we can then convert.

        Args:
            config: Model configuration
            device: Device to create model on

        Returns:
            Megatron model instance
        """
        # Try to import megatron_core
        from megatron.core import parallel_state
        from megatron.core.models.gpt import GPTModel
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )

        config = self.megatron_config

        # Initialize parallel state (required for megatron_core)
        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

        # Create transformer config
        transformer_config = TransformerConfig(
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            ffn_hidden_size=config.get("ffn_hidden_size", config["hidden_size"] * 4),
            num_attention_heads=config["num_attention_heads"],
            num_query_groups=config.get(
                "num_query_groups", config["num_attention_heads"]
            ),
            # max_position_embeddings=config.get("max_position_embeddings", 2048),
            use_cpu_initialization=True,
            activation_func=config.get("activation_func"),
            gated_linear_unit=True,
            bias_activation_fusion=False,
            add_bias_linear=config.get("add_bias_linear", True),
            normalization="RMSNorm",
            layernorm_epsilon=config.get("rms_norm_eps", 1e-6),
            attention_dropout=config.get("attention_dropout", 0.0),
            hidden_dropout=config.get("hidden_dropout", 0.0),
        )

        # Create GPT model
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                num_experts=config["num_experts"],
                moe_grouped_gemm=config["moe_grouped_gemm"],
            ),
            vocab_size=config["vocab_size"],
            max_sequence_length=config.get("max_position_embeddings", 2048),
            pre_process=True,
            post_process=True,
            fp16_lm_cross_entropy=False,
            parallel_output=False,
            share_embeddings_and_output_weights=config.get(
                "tie_word_embeddings", False
            ),
        )

        logger.info(
            f"Created Megatron model with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        return model
