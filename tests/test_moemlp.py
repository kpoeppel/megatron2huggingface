"""Tests for MoE MLP conversion in Megatron translation utility."""

import torch

from megatron2huggingface.conversion.moemlp import MoeMLPConverter


def test_moemlp_conversion(
    megatron_config_filled_dict,
    hidden_size: int = 256,
    ffn_hidden_size: int | None = None,
    num_experts: int = 4,
    moe_router_topk: int = 2,
    swiglu: bool = False,
    activation_function: str = "gelu",
    batch_size: int = 2,
    seq_len: int = 8,
    device: str = "cuda",
) -> bool:
    """Strict-style MoE MLP conversion test using real Megatron-Core MoE layer and HF
    wrapper via the converter. Mirrors test_mlp.py structure.

    Assumes TE is available for the Megatron-Core MoE spec.
    """
    if ffn_hidden_size is None:
        ffn_hidden_size = 4 * hidden_size

    # Build config from fixture and override MLP/MoE-related params
    cfg = dict(**megatron_config_filled_dict)
    cfg.update(
        dict(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_experts=num_experts,
            moe_router_topk=moe_router_topk,
            swiglu=swiglu,
            activation_function=activation_function,
            add_bias_linear=False,  # True is not supported by Megatron
            hidden_dropout=0.0,
        )
    )

    converter = MoeMLPConverter(cfg)

    # Create Megatron-Core module and pull its weights
    megatron_module = converter.create_megatron_module()
    megatron_state = {
        k: v.to(device=device) for k, v in megatron_module.state_dict().items()
    }

    # Zero all Megatron weights so both sides are deterministic/easy to compare
    for k in megatron_state:
        megatron_state[k] = 0.0 * megatron_state[k]

    # Inputs follow Megatron format [seq_len, batch, hidden]
    test_input = torch.arange(
        hidden_size, dtype=torch.float32, device=device
    ).broadcast_to(seq_len, batch_size, hidden_size)

    # Early strict filtered load-state check on HF side
    converted = converter.convert_weights(megatron_state)
    hf_module = converter.create_hf_module().to(device)
    missing, unexpected = hf_module.load_state_dict(converted, strict=False)
    assert not missing and not unexpected

    # Use the common conversion test utility.
    # Note: Follow test_mlp.py and pass 'additional_input' (singular) so it is ignored
    # by the base tester (avoids forwarding skip_bias_add to Megatron-Core).
    results = converter.test_conversion(
        megatron_state=megatron_state,
        test_input=test_input,
        additional_input={"skip_bias_add": True},
    )

    print(results)
    assert results["test_passed"]
