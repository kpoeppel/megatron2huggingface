"""
Embedding converter for Megatron-LM to HuggingFace Transformers.
Handles word embeddings and position embeddings conversion.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from megatron2huggingface.conversion.base import BaseConverter

logger = logging.getLogger(__name__)


class EmbeddingConverter(BaseConverter):
    """Converter for embedding layers."""

    def __init__(self, megatron_config: Dict[str, Any]):
        """Initialize the embedding converter."""
        super().__init__(megatron_config)

    def convert_weights(self, megatron_weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Convert Megatron embedding weights to HuggingFace format.

        Args:
            megatron_weights: Dictionary containing Megatron weights
            config: Model configuration
            **kwargs: Additional conversion parameters

        Returns:
            Dictionary with converted HuggingFace weights
        """
        hf_weights = {}

        # Convert word embeddings
        if "word_embeddings.weight" in megatron_weights:
            word_emb_weight = megatron_weights["word_embeddings.weight"]

            # Handle tensor parallelism - concatenate if needed
            if isinstance(word_emb_weight, list):
                logger.info("Concatenating tensor parallel word embeddings")
                word_emb_weight = torch.cat(word_emb_weight, dim=0)

            hf_weights["embed_tokens.weight"] = word_emb_weight
            logger.info(f"Converted word embeddings: {word_emb_weight.shape}")

        # Convert position embeddings (if present - some models use RoPE instead)
        if "position_embeddings.weight" in megatron_weights:
            pos_emb_weight = megatron_weights["position_embeddings.weight"]

            if isinstance(pos_emb_weight, list):
                logger.info("Concatenating tensor parallel position embeddings")
                pos_emb_weight = torch.cat(pos_emb_weight, dim=0)

            hf_weights["embed_positions.weight"] = pos_emb_weight
            logger.info(f"Converted position embeddings: {pos_emb_weight.shape}")

        return hf_weights

    def get_expected_keys(self, config: Any) -> Tuple[list, list]:
        """
        Get expected input and output keys for this converter.

        Args:
            config: Model configuration

        Returns:
            Tuple of (input_keys, output_keys)
        """
        input_keys = ["word_embeddings.weight"]
        output_keys = ["embed_tokens.weight"]

        # Position embeddings are optional (RoPE models don't have them)
        if hasattr(config, "position_embedding_type") and config.position_embedding_type == "learned":
            input_keys.append("position_embeddings.weight")
            output_keys.append("embed_positions.weight")

        return input_keys, output_keys

    def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Extract weights for the Megatron attention module."""

        return megatron_state


class FallbackMegatronEmbedding(nn.Module):
    """
    Fallback Megatron-style embedding implementation for testing.
    Simplified version without parallelism.
    """

    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        if max_position_embeddings is not None:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        else:
            self.position_embeddings = None

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (optional)

        Returns:
            Embedded representations [batch_size, seq_len, hidden_size]
        """
        embeddings = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            if position_ids is None:
                seq_len = input_ids.size(1)
                position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        return embeddings


def test_embedding_conversion(
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    max_position_embeddings: Optional[int] = None,
    batch_size: int = 2,
    seq_len: int = 10,
    device: str = "cpu",
) -> bool:
    """
    Test embedding conversion between Megatron and HuggingFace formats.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        max_position_embeddings: Maximum position embeddings (None for RoPE models)
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        device: Device to run test on

    Returns:
        True if conversion test passes
    """
    logger.info("Testing embedding conversion...")

    # Create fallback Megatron embedding
    megatron_embedding = FallbackMegatronEmbedding(
        vocab_size=vocab_size, hidden_size=hidden_size, max_position_embeddings=max_position_embeddings
    ).to(device)

    # Create test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    position_ids = None
    if max_position_embeddings is not None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Get Megatron output
    with torch.no_grad():
        megatron_output = megatron_embedding(input_ids, position_ids)

    # Prepare weights for conversion
    megatron_weights = {"word_embeddings.weight": megatron_embedding.word_embeddings.weight}

    if megatron_embedding.position_embeddings is not None:
        megatron_weights["position_embeddings.weight"] = megatron_embedding.position_embeddings.weight

    # Create mock config
    class MockConfig:
        def __init__(self):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.max_position_embeddings = max_position_embeddings
            if max_position_embeddings is not None:
                self.position_embedding_type = "learned"

    # Convert weights
    megatron_config = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "max_position_embeddings": max_position_embeddings,
    }
    converter = EmbeddingConverter(megatron_config)
    hf_weights = converter.convert_weights(megatron_weights)

    # Create HuggingFace-style embedding
    hf_embedding = nn.Embedding(vocab_size, hidden_size).to(device)
    hf_embedding.weight.data = hf_weights["embed_tokens.weight"]

    hf_pos_embedding = None
    if "embed_positions.weight" in hf_weights:
        hf_pos_embedding = nn.Embedding(max_position_embeddings, hidden_size).to(device)
        hf_pos_embedding.weight.data = hf_weights["embed_positions.weight"]

    # Get HuggingFace output
    with torch.no_grad():
        hf_output = hf_embedding(input_ids)
        if hf_pos_embedding is not None:
            hf_output = hf_output + hf_pos_embedding(position_ids)

    # Compare outputs
    max_diff = torch.max(torch.abs(megatron_output - hf_output)).item()
    logger.info(f"Maximum difference between outputs: {max_diff}")

    # Test should pass with very small numerical differences
    success = max_diff < 1e-5

    if success:
        logger.info("✓ Embedding conversion test passed!")
    else:
        logger.error(f"✗ Embedding conversion test failed! Max diff: {max_diff}")

    return success


if __name__ == "__main__":
    # Test with learned position embeddings
    print("Testing with learned position embeddings:")
    test_embedding_conversion(max_position_embeddings=2048)

    print("\nTesting without position embeddings (RoPE-style):")
    test_embedding_conversion(max_position_embeddings=None)
