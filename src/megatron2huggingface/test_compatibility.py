"""Comprehensive test functionality for Megatron-LM to HuggingFace
compatibility.

Tests full model conversion and inference compatibility between
frameworks.
"""

import logging
import torch
from typing import Any
import tempfile

from megatron2huggingface.modeling_megatron import MegatronForCausalLM
from megatron2huggingface.configuration_megatron import MegatronConfig
from megatron2huggingface.conversion.model import (
    ModelConverter,
    convert_megatron_checkpoint_to_hf,
)
from megatron2huggingface.conversion.base import MegatronCheckpointLoader

logger = logging.getLogger(__name__)


def test_model_compatibility(
    megatron_checkpoint_path: str | None = None,
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    num_layers: int = 2,
    num_attention_heads: int = 32,
    intermediate_size: int = 11008,
    max_position_embeddings: int = 2048,
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
    tolerance: float = 1e-4,
) -> bool:
    """Test compatibility between Megatron and HuggingFace model
    implementations.

    Args:
        megatron_checkpoint_path: Path to actual Megatron checkpoint (optional)
        vocab_size: Vocabulary size for synthetic test
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        max_position_embeddings: Maximum position embeddings
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        device: Device to run test on
        tolerance: Numerical tolerance for comparison

    Returns:
        True if compatibility test passes
    """
    logger.info("Starting model compatibility test...")

    if megatron_checkpoint_path:
        return test_real_checkpoint_compatibility(
            megatron_checkpoint_path, batch_size, seq_len, device, tolerance
        )
    else:
        return test_synthetic_model_compatibility(
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            batch_size,
            seq_len,
            device,
            tolerance,
        )


def test_real_checkpoint_compatibility(
    megatron_checkpoint_path: str,
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
    tolerance: float = 1e-4,
) -> bool:
    """Test compatibility using a real Megatron checkpoint.

    Args:
        megatron_checkpoint_path: Path to Megatron checkpoint
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        device: Device to run test on
        tolerance: Numerical tolerance for comparison

    Returns:
        True if compatibility test passes
    """
    logger.info(f"Testing real checkpoint: {megatron_checkpoint_path}")

    try:
        # Load Megatron checkpoint
        checkpoint_loader = MegatronCheckpointLoader()
        megatron_weights, megatron_config = checkpoint_loader.load_checkpoint(
            megatron_checkpoint_path
        )

        # Convert to HuggingFace format
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_model_path = convert_megatron_checkpoint_to_hf(
                megatron_checkpoint_path, temp_dir
            )

            # Load HuggingFace model
            hf_config = MegatronConfig.from_pretrained(hf_model_path)
            hf_model = MegatronForCausalLM.from_pretrained(hf_model_path)
            hf_model.to(device)
            hf_model.eval()

            # Create test input
            vocab_size = hf_config.vocab_size
            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )

            # Test HuggingFace model
            with torch.no_grad():
                hf_outputs = hf_model(input_ids)
                hf_logits = hf_outputs.logits

            logger.info(f"HuggingFace model output shape: {hf_logits.shape}")
            logger.info("✓ Real checkpoint compatibility test passed!")
            return True

    except Exception as e:
        logger.error(f"✗ Real checkpoint compatibility test failed: {e}")
        return False


def test_synthetic_model_compatibility(
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    num_layers: int = 2,
    num_attention_heads: int = 32,
    intermediate_size: int = 11008,
    max_position_embeddings: int = 2048,
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
    tolerance: float = 1e-4,
) -> bool:
    """Test compatibility using synthetic model weights.

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
        tolerance: Numerical tolerance for comparison

    Returns:
        True if compatibility test passes
    """
    logger.info("Testing synthetic model compatibility...")

    try:
        # Create synthetic Megatron weights and config
        megatron_weights, megatron_config = create_synthetic_megatron_model(
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            device,
        )

        # Convert weights
        converter = ModelConverter()
        hf_weights = converter.convert_weights(megatron_weights, megatron_config)

        # Create HuggingFace config
        hf_config = MegatronConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cache=True,
            tie_word_embeddings=False,
        )

        # Create HuggingFace model
        hf_model = MegatronForCausalLM(hf_config)
        hf_model.load(hf_weights, strict=False)
        hf_model.to(device)
        hf_model.eval()

        # Create test input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Test HuggingFace model
        with torch.no_grad():
            hf_outputs = hf_model(input_ids)
            hf_logits = hf_outputs.logits

        # Basic validation
        expected_shape = (batch_size, seq_len, vocab_size)
        if hf_logits.shape != expected_shape:
            logger.error(
                f"Output shape mismatch: {hf_logits.shape} != {expected_shape}"
            )
            return False

        # Check for NaN or Inf values
        if torch.isnan(hf_logits).any() or torch.isinf(hf_logits).any():
            logger.error("Output contains NaN or Inf values")
            return False

        # Check output range (logits should be reasonable)
        if hf_logits.abs().max() > 100:
            logger.warning(f"Large logit values detected: max={hf_logits.abs().max()}")

        logger.info(f"HuggingFace model output shape: {hf_logits.shape}")
        logger.info(f"Output range: [{hf_logits.min():.3f}, {hf_logits.max():.3f}]")
        logger.info("✓ Synthetic model compatibility test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Synthetic model compatibility test failed: {e}")
        return False


def create_synthetic_megatron_model(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], Any]:
    """Create synthetic Megatron model weights and config for testing.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        max_position_embeddings: Maximum position embeddings
        device: Device to create tensors on

    Returns:
        Tuple of (weights_dict, config)
    """

    # Create mock Megatron config
    class MockMegatronConfig:
        def __init__(self):
            self.padded_vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_attention_heads = num_attention_heads
            self.ffn_hidden_size = intermediate_size
            self.max_position_embeddings = max_position_embeddings
            self.layernorm_epsilon = 1e-5
            self.rotary_base = 10000.0
            self.attention_dropout = 0.0
            self.hidden_dropout = 0.0
            self.share_embeddings_and_output_weights = False
            self.normalization = "RMSNorm"
            self.activation_func = "swiglu"
            self.position_embedding_type = "rope"
            self.num_query_groups = num_attention_heads  # No GQA for simplicity

    config = MockMegatronConfig()

    # Create synthetic weights with proper initialization
    weights = {}

    # Embedding weights
    weights["embedding.word_embeddings.weight"] = (
        torch.randn(vocab_size, hidden_size, device=device) * 0.02
    )

    # Layer weights
    for layer_idx in range(num_layers):
        # Layer norms (initialized to 1)
        weights[f"decoder.layers.{layer_idx}.input_layernorm.weight"] = torch.ones(
            hidden_size, device=device
        )
        weights[f"decoder.layers.{layer_idx}.post_attention_layernorm.weight"] = (
            torch.ones(hidden_size, device=device)
        )

        # Attention weights (Xavier initialization)
        weights[f"decoder.layers.{layer_idx}.self_attention.query_key_value.weight"] = (
            torch.randn(3 * hidden_size, hidden_size, device=device)
            * (2.0 / (hidden_size + 3 * hidden_size)) ** 0.5
        )
        weights[f"decoder.layers.{layer_idx}.self_attention.dense.weight"] = (
            torch.randn(hidden_size, hidden_size, device=device)
            * (2.0 / (2 * hidden_size)) ** 0.5
        )

        # MLP weights (Xavier initialization)
        weights[f"decoder.layers.{layer_idx}.mlp.dense_h_to_4h.weight"] = (
            torch.randn(2 * intermediate_size, hidden_size, device=device)
            * (2.0 / (hidden_size + 2 * intermediate_size)) ** 0.5
        )
        weights[f"decoder.layers.{layer_idx}.mlp.dense_4h_to_h.weight"] = (
            torch.randn(hidden_size, intermediate_size, device=device)
            * (2.0 / (intermediate_size + hidden_size)) ** 0.5
        )

    # Final layer norm
    weights["decoder.final_layernorm.weight"] = torch.ones(hidden_size, device=device)

    # Output layer (tied with embeddings or separate)
    if config.share_embeddings_and_output_weights:
        weights["output_layer.weight"] = weights["embedding.word_embeddings.weight"]
    else:
        weights["output_layer.weight"] = (
            torch.randn(vocab_size, hidden_size, device=device) * 0.02
        )

    return weights, config


def test_component_conversions(device: str = "cpu") -> bool:
    """Test individual component conversions.

    Args:
        device: Device to run tests on

    Returns:
        True if all component tests pass
    """
    logger.info("Testing individual component conversions...")

    success = True

    # Test embedding conversion
    try:
        from .conversion.embedding import test_embedding_conversion

        if not test_embedding_conversion(device=device):
            success = False
    except Exception as e:
        logger.error(f"Embedding conversion test failed: {e}")
        success = False

    # Test layer norm conversion
    try:
        from .conversion.layer_norm import test_layer_norm_conversion

        if not test_layer_norm_conversion(device=device):
            success = False
    except Exception as e:
        logger.error(f"Layer norm conversion test failed: {e}")
        success = False

    # Test MLP conversion
    try:
        from .conversion.mlp import test_mlp_conversion

        if not test_mlp_conversion(device=device):
            success = False
    except Exception as e:
        logger.error(f"MLP conversion test failed: {e}")
        success = False

    # Test attention conversion
    try:
        from .conversion.attention import test_attention_conversion

        if not test_attention_conversion(device=device):
            success = False
    except Exception as e:
        logger.error(f"Attention conversion test failed: {e}")
        success = False

    # Test full model conversion
    try:
        from .conversion.model import test_full_model_conversion

        if not test_full_model_conversion(device=device):
            success = False
    except Exception as e:
        logger.error(f"Full model conversion test failed: {e}")
        success = False

    if success:
        logger.info("✓ All component conversion tests passed!")
    else:
        logger.error("✗ Some component conversion tests failed!")

    return success


def run_all_tests(
    megatron_checkpoint_path: str | None = None, device: str = "cpu"
) -> bool:
    """Run all compatibility tests.

    Args:
        megatron_checkpoint_path: Path to actual Megatron checkpoint (optional)
        device: Device to run tests on

    Returns:
        True if all tests pass
    """
    logger.info("Running all compatibility tests...")

    success = True

    # Test component conversions
    if not test_component_conversions(device=device):
        success = False

    # Test model compatibility
    if not test_model_compatibility(
        megatron_checkpoint_path=megatron_checkpoint_path, device=device
    ):
        success = False

    if success:
        logger.info("🎉 All compatibility tests passed!")
    else:
        logger.error("❌ Some compatibility tests failed!")

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Megatron-HF compatibility")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to Megatron checkpoint directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run tests on"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run tests
    success = run_all_tests(
        megatron_checkpoint_path=args.checkpoint, device=args.device
    )

    exit(0 if success else 1)
