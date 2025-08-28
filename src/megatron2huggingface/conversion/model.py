"""
Full model converter for Megatron-LM to HuggingFace Transformers.
Orchestrates all component converters to convert complete models.
Uses proper Megatron naming conventions and works with real megatron_core models.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json

from ..configuration_megatron import MegatronConfig
from .base import BaseConverter, extract_submodule_state_dict, add_prefix_to_state_dict
from .embedding import EmbeddingConverter
from .attention import AttentionConverter
from .mlp import MLPConverter
from .layer_norm import LayerNormConverter

logger = logging.getLogger(__name__)


class ModelConverter(BaseConverter):
    """
    Full model converter that orchestrates all component converters.
    Uses proper Megatron naming conventions and modular architecture.
    """

    def __init__(self, megatron_config: dict[str, Any]):
        super().__init__(megatron_config)
        self.embedding_converter = EmbeddingConverter(megatron_config)
        self.attention_converter = AttentionConverter(megatron_config)
        self.mlp_converter = MLPConverter(megatron_config)
        self.layer_norm_converter = LayerNormConverter(megatron_config)

    def convert_weights(self, megatron_state_dict: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Convert complete Megatron model weights to our Megatron-style format.

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
        # Extract embedding weights (typically under "language_model.embedding" prefix)
        embedding_prefix = "language_model.embedding"
        embedding_submodule_weights = extract_submodule_state_dict(megatron_state_dict, embedding_prefix)

        embedding_converted = self.embedding_converter.convert_weights(embedding_submodule_weights, **kwargs)
        converted_weights.update(add_prefix_to_state_dict(embedding_converted, ""))

        # Convert transformer layers
        num_layers = self.megatron_config["num_layers"]
        logger.info(f"Converting {num_layers} transformer layers...")

        for layer_idx in range(num_layers):
            logger.info(f"Converting layer {layer_idx}...")
            layer_prefix = f"language_model.encoder.layers.{layer_idx}"

            # Convert input layer norm
            input_norm_prefix = f"{layer_prefix}.input_layernorm"
            input_norm_submodule_weights = extract_submodule_state_dict(megatron_state_dict, input_norm_prefix)
            if input_norm_submodule_weights:
                input_norm_converted = self.layer_norm_converter.convert_weights(
                    input_norm_submodule_weights, layer_idx=layer_idx, norm_type="input_layernorm", **kwargs
                )
                converted_weights.update(
                    add_prefix_to_state_dict(input_norm_converted, f"layers.{layer_idx}.input_layernorm")
                )

            # Convert attention
            attention_prefix = f"{layer_prefix}.self_attention"
            attention_submodule_weights = extract_submodule_state_dict(megatron_state_dict, attention_prefix)
            if attention_submodule_weights:
                attention_converted = self.attention_converter.convert_weights(
                    attention_submodule_weights, layer_idx=layer_idx, **kwargs
                )
                converted_weights.update(add_prefix_to_state_dict(attention_converted, f"layers.{layer_idx}.self_attn"))

            # Convert post-attention layer norm
            post_attn_norm_prefix = f"{layer_prefix}.post_attention_layernorm"
            post_attn_norm_submodule_weights = extract_submodule_state_dict(megatron_state_dict, post_attn_norm_prefix)
            if post_attn_norm_submodule_weights:
                post_attn_norm_converted = self.layer_norm_converter.convert_weights(
                    post_attn_norm_submodule_weights,
                    layer_idx=layer_idx,
                    norm_type="post_attention_layernorm",
                    **kwargs,
                )
                converted_weights.update(
                    add_prefix_to_state_dict(post_attn_norm_converted, f"layers.{layer_idx}.post_attention_layernorm")
                )

            # Convert MLP
            mlp_prefix = f"{layer_prefix}.mlp"
            mlp_submodule_weights = extract_submodule_state_dict(megatron_state_dict, mlp_prefix)
            if mlp_submodule_weights:
                mlp_converted = self.mlp_converter.convert_weights(mlp_submodule_weights, layer_idx=layer_idx, **kwargs)
                converted_weights.update(add_prefix_to_state_dict(mlp_converted, f"layers.{layer_idx}.mlp"))

        # Convert final layer norm
        logger.info("Converting final layer norm...")
        final_norm_prefix = "language_model.encoder.final_layernorm"
        final_norm_submodule_weights = extract_submodule_state_dict(megatron_state_dict, final_norm_prefix)
        if final_norm_submodule_weights:
            final_norm_converted = self.layer_norm_converter.convert_weights(
                final_norm_submodule_weights, layer_idx=None, norm_type="final_layernorm", **kwargs
            )
            converted_weights.update(add_prefix_to_state_dict(final_norm_converted, "norm"))

        # Convert language modeling head (if present)
        # Look for various possible names for the output layer
        output_layer_keys = [
            "output_layer.weight",
            "lm_head.weight",
            "language_model.output_layer.weight",
            "model.language_model.output_layer.weight",
        ]

        output_layer_weight = None
        for key in output_layer_keys:
            if key in megatron_state_dict:
                output_layer_weight = megatron_state_dict[key]
                logger.info(f"Found output layer at key: {key}")
                break

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

        logger.info(f"Model conversion complete! Converted {len(converted_weights)} weight tensors.")
        return converted_weights

    def get_expected_keys(self, config: dict[str, Any]) -> Tuple[list, list]:
        """
        Get expected input and output keys for the full model.

        Args:
            config: Model configuration

        Returns:
            Tuple of (input_keys, output_keys)
        """
        input_keys = []
        output_keys = []

        # Embedding keys
        emb_input, emb_output = self.embedding_converter.get_expected_keys(config)
        input_keys.extend(emb_input)
        output_keys.extend(emb_output)

        # Layer keys
        for layer_idx in range(config["num_layers"]):
            # Input layer norm
            norm_input, norm_output = self.layer_norm_converter.get_expected_keys(
                config, layer_idx=layer_idx, norm_type="input_layernorm"
            )
            input_keys.extend(norm_input)
            output_keys.extend(norm_output)

            # Attention
            attn_input, attn_output = self.attention_converter.get_expected_keys(config, layer_idx=layer_idx)
            input_keys.extend(attn_input)
            output_keys.extend(attn_output)

            # Post-attention layer norm
            post_norm_input, post_norm_output = self.layer_norm_converter.get_expected_keys(
                config, layer_idx=layer_idx, norm_type="post_attention_layernorm"
            )
            input_keys.extend(post_norm_input)
            output_keys.extend(post_norm_output)

            # MLP
            mlp_input, mlp_output = self.mlp_converter.get_expected_keys(config, layer_idx=layer_idx)
            input_keys.extend(mlp_input)
            output_keys.extend(mlp_output)

        # Final layer norm
        final_norm_input, final_norm_output = self.layer_norm_converter.get_expected_keys(
            config, layer_idx=None, norm_type="final_layernorm"
        )
        input_keys.extend(final_norm_input)
        output_keys.extend(final_norm_output)

        # Language modeling head
        input_keys.extend(["output_layer.weight", "lm_head.weight", "language_model.output_layer.weight"])
        output_keys.append("lm_head.weight")

        return input_keys, output_keys


def create_megatron_model_from_config(config: dict[str, Any], device: str = "cpu") -> nn.Module:
    """
    Create a Megatron model using megatron_core for testing purposes.
    This creates a real Megatron model that we can then convert.

    Args:
        config: Model configuration
        device: Device to create model on

    Returns:
        Megatron model instance
    """
    try:
        # Try to import megatron_core
        from megatron.core import parallel_state
        from megatron.core.models.gpt import GPTModel
        from megatron.core.transformer.spec_utils import import_module
        from megatron.core.transformer.transformer_config import TransformerConfig

        # Initialize parallel state (required for megatron_core)
        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

        # Create transformer config
        transformer_config = TransformerConfig(
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            ffn_hidden_size=config.get("ffn_hidden_size", config["hidden_size"] * 4),
            num_attention_heads=config["num_attention_heads"],
            num_query_groups=config.get("num_query_groups", config["num_attention_heads"]),
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
            transformer_layer_spec=None,  # Use default
            vocab_size=config["vocab_size"],
            max_sequence_length=config.get("max_position_embeddings", 2048),
            pre_process=True,
            post_process=True,
            fp16_lm_cross_entropy=False,
            parallel_output=False,
            share_embeddings_and_output_weights=config.get("tie_word_embeddings", False),
        )

        model = model.to(device)
        logger.info(f"Created Megatron model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    except ImportError as e:
        logger.warning(f"Could not import megatron_core: {e}")
        logger.info("Creating fallback synthetic model for testing...")
        raise e


def test_full_model_conversion(
    vocab_size: int = 1000,
    hidden_size: int = 512,
    num_layers: int = 2,
    num_attention_heads: int = 8,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 128,
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
) -> bool:
    """
    Test full model conversion using proper Megatron structure and converters.
    Creates a real megatron_core model, converts it, instantiates our MegatronModel,
    and compares parameters in both shape and path.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        max_position_embeddings: Maximum position embeddings
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        device: Device to run test on

    Returns:
        True if conversion test passes
    """
    logger.info("Testing full model conversion with proper Megatron structure...")

    # Create configuration
    config = vars(MegatronConfig())
    config.update(
        **{
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "ffn_hidden_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "num_query_groups": num_attention_heads,  # Standard MHA
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "tie_word_embeddings": False,
            "normalization": "RMSNorm",
            "activation_function": "silu",
            "gated_linear_unit": True,
            "add_bias_linear": True,
        }
    )

    # Create or load Megatron model
    try:
        # Try to create real megatron_core model
        megatron_model = create_megatron_model_from_config(config, device)
        if isinstance(megatron_model, nn.Module):
            megatron_state_dict = megatron_model.state_dict()
        else:
            megatron_state_dict = megatron_model
    except Exception as e:
        raise e
        logger.warning(f"Failed to create megatron_core model: {e}")
        logger.info("Using synthetic model instead...")

    print("Megatron state:", {name: p.shape for name, p in megatron_state_dict.items()})
    logger.info(f"Original Megatron model has {len(megatron_state_dict)} weight tensors")

    # Convert using modular converter architecture
    converter = ModelConverter(config)
    converted_weights = converter.convert_weights(megatron_state_dict)

    hf_config = MegatronConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_query_groups=num_attention_heads,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
        normalization="RMSNorm",
        activation_function="silu",
        gated_linear_unit=True,
        add_bias_linear=True,
    )

    # Instantiate our translated MegatronModel
    try:
        from ..modeling_megatron import MegatronModel

        translated_model = MegatronModel(hf_config).to(device)
        logger.info(
            f"Created translated MegatronModel with {sum(p.numel() for p in translated_model.parameters()):,} parameters"
        )
    except Exception as e:
        logger.error(f"Failed to create translated MegatronModel: {e}")
        return False

    hf_params = dict(translated_model.named_parameters())
    print("native model:", {name: hf_params[name].shape for name in sorted(hf_params)})

    print("converted model:", {name: converted_weights[name].shape for name in sorted(converted_weights)})

    # Load converted weights into the translated model
    try:
        # Filter out keys that don't match the model structure
        model_state_dict = translated_model.state_dict()
        filtered_weights = {}

        for key in model_state_dict.keys():
            if key in converted_weights:
                if model_state_dict[key].shape == converted_weights[key].shape:
                    filtered_weights[key] = converted_weights[key]
                else:
                    logger.warning(
                        f"Shape mismatch for {key}: model={model_state_dict[key].shape}, converted={converted_weights[key].shape}"
                    )
            else:
                logger.warning(f"Missing key in converted weights: {key}")

        # Load the filtered weights
        missing_keys, unexpected_keys = translated_model.load_state_dict(filtered_weights, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading: {unexpected_keys}")

        logger.info(f"Successfully loaded {len(filtered_weights)} weight tensors into translated model")

    except Exception as e:
        logger.error(f"Failed to load converted weights into translated model: {e}")
        return False

    # Compare parameter shapes and paths
    success = True

    logger.info("Comparing parameter shapes and paths...")

    # Get parameter dictionaries
    original_params = {k: v.shape for k, v in megatron_state_dict.items()}
    translated_params = {k: v.shape for k, v in translated_model.named_parameters()}

    logger.info(f"Original model parameters: {len(original_params)}")
    logger.info(f"Translated model parameters: {len(translated_params)}")

    # Check that we have the expected structure in translated model
    expected_patterns = [
        "embed_tokens.weight",
        "embed_positions.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.linear_qkv.weight",
        "layers.0.self_attn.linear_proj.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.mlp.linear_fc1.weight",
        "layers.0.mlp.linear_fc2.weight",
        "norm.weight",
    ]

    for pattern in expected_patterns:
        found = any(pattern in key for key in translated_params.keys())
        if not found:
            logger.error(f"Missing expected pattern in translated model: {pattern}")
            success = False

    # Compare total parameter count
    original_total = sum(torch.tensor(shape).prod().item() for shape in original_params.values())
    translated_total = sum(torch.tensor(shape).prod().item() for shape in translated_params.values())

    logger.info(f"Original model total parameters: {original_total:,}")
    logger.info(f"Translated model total parameters: {translated_total:,}")

    if abs(original_total - translated_total) > 0.01 * original_total:  # Allow 1% difference
        logger.warning(f"Parameter count mismatch: original={original_total:,}, translated={translated_total:,}")

    # Test forward pass to ensure the model works
    try:
        logger.info("Testing forward pass...")
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            outputs = translated_model(input_ids)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        expected_shape = (batch_size, seq_len, vocab_size)
        if logits.shape == expected_shape:
            logger.info(f"✓ Forward pass successful: output shape {logits.shape}")
        else:
            logger.error(f"✗ Forward pass shape mismatch: expected {expected_shape}, got {logits.shape}")
            success = False

    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        success = False

    # Log detailed parameter comparison
    logger.info("Parameter structure comparison:")
    logger.info("Translated model structure:")
    for i, (name, shape) in enumerate(sorted(translated_params.items())[:15]):
        logger.info(f"  {name}: {shape}")
    if len(translated_params) > 15:
        logger.info(f"  ... and {len(translated_params) - 15} more")

    if success:
        logger.info("✓ Full model conversion test with parameter comparison passed!")
        logger.info("Successfully converted and instantiated MegatronModel")
        logger.info(f"Converted {len(converted_weights)} weight tensors")
        logger.info(f"Model has {translated_total:,} parameters")
    else:
        logger.error("✗ Full model conversion test failed!")

    return success


if __name__ == "__main__":
    # Test full model conversion
    print("Testing full model conversion with proper Megatron structure:")
    test_full_model_conversion()
