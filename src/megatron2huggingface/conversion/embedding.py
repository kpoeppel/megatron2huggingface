"""Embedding converter for Megatron-LM to HuggingFace Transformers.

Handles word embeddings and position embeddings conversion.
"""

import logging
import torch
import torch.nn as nn
from typing import Any

from megatron2huggingface.conversion.config import megatron2transformer_config
from megatron2huggingface.conversion.base import BaseConverter
from megatron2huggingface.configuration_megatron import MegatronConfig

logger = logging.getLogger(__name__)


class EmbeddingConverter(BaseConverter):
    """Converter for embedding layers."""

    def __init__(self, megatron_config: dict[str, Any]):
        """Initialize the embedding converter."""
        super().__init__(megatron_config)

    def convert_weights(
        self, megatron_weights: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert Megatron embedding weights to HuggingFace format.

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

    def create_megatron_module(self, **kwargs):
        """Instantiate the Megatron-LM LanguageModelEmbedding (no mocks)."""
        from megatron.core.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        cfg = dict(self.megatron_config)

        # Map config fields to Megatron-LM API
        vocab_size = cfg.get("vocab_size")
        max_seq_length = cfg.get(
            "max_sequence_length", cfg.get("max_position_embeddings", 0) or 0
        )

        # Decide position embedding type
        pos_type = cfg.get("position_embedding_type", None)
        if pos_type is None:
            add_pos = cfg.get("add_position_embedding", True)
            pos_type = (
                "learned_absolute" if (add_pos and max_seq_length > 0) else "none"
            )

        return LanguageModelEmbedding(
            config=megatron2transformer_config(cfg),
            vocab_size=vocab_size,
            max_sequence_length=int(max_seq_length),
            position_embedding_type=pos_type,
        )

    def create_hf_module(self, **kwargs):
        """Instantiate a simple HF-style embedding module that mirrors
        Megatron."""
        config = MegatronConfig(**self.megatron_config)

        class HFEmbedding(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
                self.add_position_embedding = getattr(
                    cfg, "add_position_embedding", True
                )
                self.max_position_embeddings = getattr(
                    cfg, "max_position_embeddings", None
                )
                self.position_embedding_type = getattr(
                    cfg, "position_embedding_type", "learned_absolute"
                )
                if (
                    self.add_position_embedding
                    and self.max_position_embeddings is not None
                    and self.position_embedding_type == "learned_absolute"
                ):
                    self.embed_positions = nn.Embedding(
                        self.max_position_embeddings, cfg.hidden_size
                    )
                else:
                    self.embed_positions = None

            def forward(
                self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None
            ):
                x = self.embed_tokens(input_ids)
                if self.embed_positions is not None and position_ids is not None:
                    x = x + self.embed_positions(position_ids)
                return x

        return HFEmbedding(config)

    # def get_expected_keys(self, config: Any) -> Tuple[list, list]:
    #     """
    #     Get expected input and output keys for this converter.

    #     Args:
    #         config: Model configuration

    #     Returns:
    #         Tuple of (input_keys, output_keys)
    #     """
    #     input_keys = ["word_embeddings.weight"]
    #     output_keys = ["embed_tokens.weight"]

    #     # Position embeddings are optional (RoPE models don't have them)
    #     if hasattr(config, "position_embedding_type") and config.position_embedding_type == "learned":
    #         input_keys.append("position_embeddings.weight")
    #         output_keys.append("embed_positions.weight")

    #     return input_keys, output_keys

    # def _extract_megatron_weights(self, megatron_state: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    #     """Extract weights for the Megatron attention module."""

    #     return megatron_state
