#!/usr/bin/env python3
"""
Simple test runner for Megatron translation utility.
Can be run on a laptop without requiring actual Megatron checkpoints.
"""

import os
import sys
import logging
from pathlib import Path

import torch.distributed as dist

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "65432"
dist.init_process_group(backend="gloo", rank=0, world_size=1)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
    )


def test_individual_components(device: str = "cpu", debug: bool = False):
    """Test individual modular components."""
    print("=" * 60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)

    success_count = 0
    total_count = 0

    # Test embedding conversion
    print("\n1. Testing Embedding Conversion...")
    try:
        from megatron2huggingface.conversion.embedding import test_embedding_conversion

        # Test with learned position embeddings
        print("   - Testing with learned position embeddings...")
        if test_embedding_conversion(
            vocab_size=1000, hidden_size=512, max_position_embeddings=128, batch_size=2, seq_len=10, device=device
        ):
            print("   ✓ Learned position embeddings test passed")
            success_count += 1
        else:
            print("   ✗ Learned position embeddings test failed")
        total_count += 1

        # Test without position embeddings (RoPE style)
        print("   - Testing without position embeddings (RoPE style)...")
        if test_embedding_conversion(
            vocab_size=1000, hidden_size=512, max_position_embeddings=None, batch_size=2, seq_len=10, device=device
        ):
            print("   ✓ RoPE style test passed")
            success_count += 1
        else:
            print("   ✗ RoPE style test failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ Embedding test failed with error: {e}")
        total_count += 2

    # Test layer norm conversion
    print("\n2. Testing Layer Norm Conversion...")
    try:
        from megatron2huggingface.conversion.layer_norm import test_layer_norm_conversion

        # Test RMSNorm
        print("   - Testing RMSNorm...")
        if test_layer_norm_conversion(hidden_size=512, batch_size=2, seq_len=10, norm_type="RMSNorm", device=device):
            print("   ✓ RMSNorm test passed")
            success_count += 1
        else:
            print("   ✗ RMSNorm test failed")
        total_count += 1

        # Test LayerNorm
        print("   - Testing LayerNorm...")
        if test_layer_norm_conversion(hidden_size=512, batch_size=2, seq_len=10, norm_type="LayerNorm", device=device):
            print("   ✓ LayerNorm test passed")
            success_count += 1
        else:
            print("   ✗ LayerNorm test failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ Layer norm test failed with error: {e}")
        total_count += 2

    # Test MLP conversion
    print("\n3. Testing MLP Conversion...")
    try:
        from megatron2huggingface.conversion.mlp import test_mlp_conversion

        # Test SwiGLU MLP
        print("   - Testing SwiGLU MLP...")
        if test_mlp_conversion(
            hidden_size=512,
            ffn_hidden_size=1024,
            batch_size=2,
            seq_len=10,
            activation_function="swish",
            device=device,
            debug=debug,
        ):
            print("   ✓ SwiGLU MLP test passed")
            success_count += 1
        else:
            print("   ✗ SwiGLU MLP test failed")
        total_count += 1

        # Test GELU MLP
        print("   - Testing GELU MLP...")
        if test_mlp_conversion(
            hidden_size=512,
            ffn_hidden_size=1024,
            batch_size=2,
            seq_len=10,
            activation_function="gelu",
            device=device,
            debug=debug,
        ):
            print("   ✓ GELU MLP test passed")
            success_count += 1
        else:
            print("   ✗ GELU MLP test failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ MLP test failed with error: {e}")
        total_count += 2

    # Test attention conversion
    print("\n4. Testing Attention Conversion...")
    try:
        from megatron2huggingface.conversion.attention import test_attention_conversion

        # Test standard attention
        print("   - Testing standard multi-head attention...")
        if test_attention_conversion(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=8,
            batch_size=2,
            seq_len=10,
            device=device,
            debug=debug,
        ):
            print("   ✓ Standard attention test passed")
            success_count += 1
        else:
            print("   ✗ Standard attention test failed")
        total_count += 1

        # Test Group Query Attention
        print("   - Testing Group Query Attention (GQA)...")
        if test_attention_conversion(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            batch_size=2,
            seq_len=10,
            device=device,
            debug=debug,
        ):
            print("   ✓ GQA test passed")
            success_count += 1
        else:
            print("   ✗ GQA test failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ Attention test failed with error: {e}")
        total_count += 2

    print(f"\nComponent Tests Summary: {success_count}/{total_count} passed")
    return success_count, total_count


def test_full_model_conversion(device: str = "cpu", debug: bool = False):
    """Test full model conversion with synthetic weights."""
    print("\n" + "=" * 60)
    print("TESTING FULL MODEL CONVERSION")
    print("=" * 60)

    try:
        from megatron2huggingface.conversion.model import test_full_model_conversion

        print("\nTesting full model conversion with synthetic weights...")
        print("(This creates a small model and tests the complete conversion pipeline)")

        success = test_full_model_conversion(
            vocab_size=1000,  # Small vocab for laptop testing
            hidden_size=512,  # Small hidden size
            num_layers=2,  # Just 2 layers
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=128,
            batch_size=2,
            seq_len=10,
            device=device,
        )

        if success:
            print("✓ Full model conversion test passed!")
            return True
        else:
            print("✗ Full model conversion test failed!")
            return False

    except Exception as e:
        if debug:
            raise e

        print(f"✗ Full model conversion test failed with error: {e}")
        return False


def test_modular_components(device: str = "cpu", debug: bool = False):
    """Test the new modular HuggingFace components."""
    print("\n" + "=" * 60)
    print("TESTING MODULAR HUGGINGFACE COMPONENTS")
    print("=" * 60)

    success_count = 0
    total_count = 0

    # Test individual modular components
    print("\n1. Testing Modular Layer Norm...")
    try:
        from megatron2huggingface.modeling.layer_norm import RMSNorm, LayerNorm
        import torch

        hidden_size = 512
        batch_size, seq_len = 2, 10

        # Test RMSNorm
        rms_norm = RMSNorm(hidden_size).to(device)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        output = rms_norm(x)

        if output.shape == x.shape:
            print("   ✓ RMSNorm component works")
            success_count += 1
        else:
            print("   ✗ RMSNorm component failed")
        total_count += 1

        # Test LayerNorm
        layer_norm = LayerNorm(hidden_size).to(device)
        output = layer_norm(x)

        if output.shape == x.shape:
            print("   ✓ LayerNorm component works")
            success_count += 1
        else:
            print("   ✗ LayerNorm component failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ Layer norm components failed: {e}")
        total_count += 2

    # Test MLP components
    print("\n2. Testing Modular MLP...")
    try:
        from megatron2huggingface.modeling.mlp import MLP, SwiGLUMLP
        from megatron2huggingface.configuration_megatron import MegatronConfig
        import torch

        hidden_size = 512
        intermediate_size = 1024
        batch_size, seq_len = 2, 10

        # Create a basic config for testing
        config = MegatronConfig(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            gated_linear_unit=True,
            activation_function="silu",
            add_bias_linear=True,
            hidden_dropout=0.0,
        )

        # Test SwiGLU MLP - input should be [seq_len, batch, hidden_size] for Megatron format
        mlp = SwiGLUMLP(config).to(device)
        x = torch.randn(seq_len, batch_size, hidden_size, device=device)
        output, bias = mlp(x)

        if output.shape == x.shape:
            print("   ✓ SwiGLU MLP component works")
            success_count += 1
        else:
            print("   ✗ SwiGLU MLP component failed")
        total_count += 1

        # Test standard MLP
        config_std = MegatronConfig(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            gated_linear_unit=False,
            activation_function="gelu",
            add_bias_linear=True,
            hidden_dropout=0.0,
        )
        mlp_std = MLP(config_std).to(device)
        output_std, bias_std = mlp_std(x)

        if output_std.shape == x.shape:
            print("   ✓ Standard MLP component works")
            success_count += 1
        else:
            print("   ✗ Standard MLP component failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ MLP components failed: {e}")
        total_count += 2

    # Test Attention components
    print("\n3. Testing Modular Attention...")
    try:
        from megatron2huggingface.modeling.attention import SelfAttention
        from megatron2huggingface.configuration_megatron import MegatronConfig
        import torch

        hidden_size = 512
        num_heads = 8
        batch_size, seq_len = 2, 10

        # Create a basic config for testing
        config = MegatronConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_query_groups=num_heads,  # Standard multi-head attention
            attention_dropout=0.0,
            add_bias_linear=True,
            add_qkv_bias=False,
            qk_layernorm=False,
            layernorm_epsilon=1e-5,
        )

        attention = SelfAttention(config).to(device)

        # Input should be [seq_len, batch, hidden_size] for Megatron format
        x = torch.randn(seq_len, batch_size, hidden_size, device=device)
        output, bias = attention(x)

        if output.shape == x.shape:
            print("   ✓ Attention component works")
            success_count += 1
        else:
            print("   ✗ Attention component failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ Attention components failed: {e}")
        total_count += 1

    # Test Decoder Layer
    print("\n4. Testing Modular Decoder Layer...")
    try:
        from megatron2huggingface.modeling.decoder_layer import TransformerLayer
        from megatron2huggingface.configuration_megatron import MegatronConfig
        import torch

        hidden_size = 512
        intermediate_size = 1024
        num_heads = 8
        batch_size, seq_len = 2, 10

        # Create a basic config for testing
        config = MegatronConfig(
            hidden_size=hidden_size,
            ffn_hidden_size=intermediate_size,
            num_attention_heads=num_heads,
            num_query_groups=num_heads,
            normalization="RMSNorm",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            add_bias_linear=True,
            add_qkv_bias=False,
            qk_layernorm=False,
            layernorm_epsilon=1e-5,
            gated_linear_unit=True,
            activation_function="silu",
        )

        decoder_layer = TransformerLayer(config).to(device)

        # Input should be [seq_len, batch, hidden_size] for Megatron format
        x = torch.randn(seq_len, batch_size, hidden_size, device=device)
        output, bias = decoder_layer(x)

        if output.shape == x.shape:
            print("   ✓ Decoder layer component works")
            success_count += 1
        else:
            print("   ✗ Decoder layer component failed")
        total_count += 1

    except Exception as e:
        if debug:
            raise e

        print(f"   ✗ Decoder layer components failed: {e}")
        total_count += 1

    print(f"\nModular Components Summary: {success_count}/{total_count} passed")
    return success_count, total_count


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Megatron translation utility")
    parser.add_argument("--device", default="cpu", help="Device to run tests on (cpu/cuda)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--components-only", action="store_true", help="Only test individual components")
    parser.add_argument("--modular-only", action="store_true", help="Only test modular components")
    parser.add_argument("--full-only", action="store_true", help="Only test full model conversion")
    parser.add_argument("--debug", action="store_true", help="Raise Exceptions to be caught by debugger")

    args = parser.parse_args()

    setup_logging(args.verbose)

    print("🚀 Megatron Translation Utility Test Runner")
    print(f"Device: {args.device}")
    print(f"Verbose: {args.verbose}")
    print(f"Debug: {args.debug}")

    total_success = 0
    total_tests = 0

    if not args.modular_only and not args.full_only:
        # Test individual components
        comp_success, comp_total = test_individual_components(args.device, args.debug)
        total_success += comp_success
        total_tests += comp_total

    if not args.components_only and not args.full_only:
        # Test modular components
        mod_success, mod_total = test_modular_components(args.device, args.debug)
        total_success += mod_success
        total_tests += mod_total

    if not args.components_only and not args.modular_only:
        # Test full model conversion
        if test_full_model_conversion(args.device, args.debug):
            total_success += 1
        total_tests += 1

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total tests passed: {total_success}/{total_tests}")

    if total_success == total_tests:
        print("🎉 All tests passed! The utility is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
