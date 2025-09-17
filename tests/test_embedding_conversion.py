"""Tests for embedding conversion in Megatron translation utility."""

import torch

from megatron2huggingface.conversion.embedding import EmbeddingConverter
from megatron2huggingface.configuration_megatron import MegatronConfig


def test_embedding_conversion(
    megatron_config_filled_dict,
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    max_position_embeddings: int | None = None,
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
) -> bool:
    """Test embedding conversion using real Megatron-LM embedding and HF
    wrapper via the converter.

    Aligns with the strict style of test_attention.py (no
    fallbacks/mocks).
    """

    # Build base config from fixture and override embedding params
    cfg = dict(**megatron_config_filled_dict)
    cfg.update(
        dict(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            # position embedding type: learned if positions provided, otherwise none/rope (handled in converter)
            add_position_embedding=max_position_embeddings is not None,
            position_embedding_type="learned_absolute"
            if max_position_embeddings is not None
            else "none",
        )
    )

    # Random Megatron weights in expected keys
    megatron_state = {}
    megatron_state["word_embeddings.weight"] = torch.randn(
        vocab_size, hidden_size, device=device
    )
    if max_position_embeddings is not None:
        megatron_state["position_embeddings.weight"] = torch.randn(
            max_position_embeddings, hidden_size, device=device
        )

    # Converter and HF config
    converter = EmbeddingConverter(cfg)
    hf_config = MegatronConfig(**cfg)

    # Inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    additional = {}
    # if max_position_embeddings is not None:
    position_ids = (
        torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    additional["position_ids"] = position_ids

    # Use the common converter test harness to compare Megatron vs HF implementations
    results = converter.test_conversion(
        megatron_state=megatron_state,
        hf_config=hf_config,
        test_input=input_ids,
        additional_inputs=additional,
        permute_megatron_output=(1, 0, 2),
    )

    print(results)
    assert results["test_passed"]
