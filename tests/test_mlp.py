"""Tests for MLP conversion in Megatron translation utility."""

import torch

from megatron2huggingface.conversion.mlp import MLPConverter
from megatron2huggingface.configuration_megatron import MegatronConfig


def test_mlp_conversion(
    megatron_config_filled_dict,
    hidden_size: int = 512,
    ffn_hidden_size: int | None = None,
    gated_linear_unit: bool = False,
    activation_function: str = "gelu",
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cuda",
) -> bool:
    """Strict-style MLP conversion test using real Megatron-Core MLP and HF
    wrapper via the converter.

    Aligns with test_attention.py (no fallbacks/mocks, relies on
    conftest for init).
    """
    if ffn_hidden_size is None:
        ffn_hidden_size = 4 * hidden_size

    # Build config from fixture and override MLP-related params
    cfg = dict(**megatron_config_filled_dict)
    cfg.update(
        dict(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            gated_linear_unit=gated_linear_unit,
            activation_function=activation_function,
            add_bias_linear=True,
            hidden_dropout=0.0,
        )
    )

    converter = MLPConverter(cfg)
    megatron_state = {
        k: v.to(device=device)
        for k, v in converter.create_megatron_module().state_dict().items()
    }

    for k in megatron_state:
        megatron_state[k] = 0.0 * megatron_state[k]

    hf_config = MegatronConfig(**cfg)

    # Inputs: for MLP we follow Megatron format [seq_len, batch, hidden]
    test_input = torch.arange(
        hidden_size, dtype=torch.float32, device=device
    ).broadcast_to(
        seq_len, batch_size, hidden_size
    )  # torch.randn(seq_len, batch_size, hidden_size, device=device)

    # Strict filtered load-state check to catch missing/unexpected keys early
    converted = converter.convert_weights(megatron_state)
    hf_module = converter.create_hf_module(hf_config).to(device)
    missing, unexpected = hf_module.load_state_dict(converted, strict=False)
    assert not missing and not unexpected

    results = converter.test_conversion(
        megatron_state=megatron_state,
        hf_config=hf_config,
        test_input=test_input,
        additional_input={"skip_bias_add": True},
    )

    print(results)
    assert results["test_passed"]
