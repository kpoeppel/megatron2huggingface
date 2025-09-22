"""Tests for attention conversion in Megatron translation utility."""

import pytest

import torch

from megatron2huggingface.conversion.attention import AttentionConverter
from megatron2huggingface.configuration_megatron import MegatronConfig


def test_modular_attention(
    megatron_config_filled_dict, device: str = "cpu", debug: bool = False
):
    """Test the modular Attention components."""

    from megatron2huggingface.modeling.attention import SelfAttention
    from megatron2huggingface.configuration_megatron import MegatronConfig
    import torch

    hidden_size = 512
    seq_len = 10
    batch_size = 2

    # update config
    config = dict(**megatron_config_filled_dict)
    config["hidden_size"] = hidden_size

    attention = SelfAttention(MegatronConfig(**config)).to(device)

    # Input should be [seq_len, batch, hidden_size] for Megatron format
    x = torch.randn(seq_len, batch_size, hidden_size, device=device)
    output, bias, key_value_cache = attention(x)

    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.parametrize(
    "hidden_size,num_attention_heads,num_key_value_heads,batch_size,seq_len",
    [
        (64, 2, 2, 2, 3),  # small
        (64, 2, 2, 2, 256),  # long
        (64, 2, 2, 128, 5),  # batch
        (512, 2, 2, 2, 3),  # broad
        (512, 8, 4, 1, 1024),  # large
    ],
)
def test_attention_conversion(
    megatron_config_filled_dict,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    batch_size: int,
    seq_len: int,
    device: str = "cuda",
    debug: bool = False,
) -> bool:
    """Test attention conversion with synthetic weights.

    Args:
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        batch_size: Batch size for test input
        seq_len: Sequence length for test input
        device: Device to run test on
        debug: Whether to raise exceptions or return False

    Returns:
        True if test passes, False otherwise
    """
    # Create synthetic Megatron config
    kv_channels = hidden_size // num_attention_heads
    megatron_config = dict(**megatron_config_filled_dict)
    megatron_config.update(
        **{
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "kv_channels": kv_channels,
            "num_query_groups": num_key_value_heads,
            "add_bias_linear": False,
            "normalization": "RMSNorm",
        }
    )

    # Create converter
    converter = AttentionConverter(megatron_config)

    # Create HuggingFace config
    hf_config = MegatronConfig(**megatron_config)

    megatron_module = converter.create_megatron_module()
    megatron_state = megatron_module.state_dict()

    # Create test input
    test_input = torch.randn(seq_len, batch_size, hidden_size, device=device)
    # Fo TE attention, no attention_mask should be passed (masking handled internally).
    # attention_mask = None

    # Strict filtered load-state check to catch missing/unexpected keys early
    converted = converter.convert_weights(megatron_state)
    hf_module = converter.create_hf_module(hf_config).to(device)
    missing, unexpected = hf_module.load(converted, strict=False)
    assert not missing and not unexpected

    # Test conversion
    results = converter.test_conversion(
        megatron_state=megatron_state,
        hf_config=hf_config,
        test_input=test_input,
        additional_inputs={"attention_mask": None},
        atol=1e-3,
        rtol=1e-2,
    )

    print(results)
    assert results["test_passed"]
