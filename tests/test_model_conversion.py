"""Tests for full model conversion in Megatron translation utility."""

import torch

from megatron2huggingface.configuration_megatron import MegatronConfig
from megatron2huggingface.conversion.model import ModelConverter


def test_modular_decoder_layer(device: str = "cpu", debug: bool = False):
    """Smoke test for HF-side TransformerLayer component (kept minimal, aligns
    with strict style)."""
    from megatron2huggingface.modeling.decoder_layer import TransformerLayer

    hidden_size = 512
    intermediate_size = 1024
    num_heads = 8
    batch_size, seq_len = 2, 10

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
        swiglu=True,
        activation_function="silu",
    )

    layer = TransformerLayer(config).to(device)
    x = torch.randn(seq_len, batch_size, hidden_size, device=device)  # S,B,E layout
    y, _, _ = layer(x)
    assert y.shape == x.shape


def test_full_model_conversion(
    megatron_config_filled_dict,
    vocab_size: int = 1000,
    hidden_size: int = 256,
    num_layers: int = 1,
    num_attention_heads: int = 2,
    intermediate_size: int = 256,
    max_position_embeddings: int = 128,
    batch_size: int = 3,
    seq_len: int = 10,
    device: str = "cuda",
) -> bool:
    """
    Strict full-model conversion test:
    - Create a real Megatron-Core GPTModel
    - Convert its weights with ModelConverter
    - Instantiate our MegatronModel and load converted weights
    - Validate structure and shape parity and run a forward pass
    """
    # Build config from fixture
    cfg = dict(**megatron_config_filled_dict)
    cfg.update(
        dict(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            num_kv_channels=hidden_size // num_attention_heads,
            kv_channels=hidden_size // num_attention_heads,
            num_query_groups=num_attention_heads,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            tie_word_embeddings=False,
            normalization="RMSNorm",
            activation_function="silu",
            swiglu=True,
            add_bias_linear=True,
        )
    )

    # Create Megatron model and extract its state dict
    megatron_model = ModelConverter(cfg).create_megatron_module().to(device)
    megatron_state = (
        megatron_model.state_dict()
        if hasattr(megatron_model, "state_dict")
        else megatron_model
    )

    # Convert complete model weights
    converter = ModelConverter(cfg)
    converted = converter.convert_weights(megatron_state)

    # Instantiate our Megatron-style HF model
    from megatron2huggingface.modeling_megatron import MegatronForCausalLM

    hf_config = MegatronConfig(**cfg)
    translated_model = MegatronForCausalLM(hf_config).to(device)

    # Load converted weights (filter to matching keys)
    converted = converter.convert_weights(megatron_state)
    hf_module = converter.create_hf_module().to(device)
    missing, unexpected = hf_module.load_state_dict(converted, strict=False)
    assert not missing and not unexpected

    # Compare parameter counts
    original_total = sum(p.numel() for p in megatron_model.parameters())
    translated_total = sum(p.numel() for p in translated_model.parameters())
    assert original_total == translated_total

    # Forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        outputs = translated_model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    results = converter.test_conversion(
        megatron_state=megatron_state,
        test_input=input_ids,
        atol=1e-3,
        rtol=1e-2,
        additional_inputs={
            "attention_mask": None,
            "position_ids": torch.arange(input_ids.shape[1], device=input_ids.device)[
                None, :
            ].broadcast_to(input_ids.shape),
        },
        hf_output_conversion=lambda x: x.logits,
    )

    print(results)
    assert results["test_passed"]

    assert logits.shape == (batch_size, seq_len, vocab_size)
