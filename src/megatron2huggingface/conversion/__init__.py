"""
Modular checkpoint conversion utilities for Megatron-LM to HuggingFace Transformers.

This package provides modular conversion utilities that allow for component-wise
conversion and testing, inspired by the plstm conversion approach.
"""

from .base import ConversionRegistry, MegatronCheckpointLoader
from .attention import AttentionConverter
from .mlp import MLPConverter
from .embedding import EmbeddingConverter
from .layer_norm import LayerNormConverter
from .model import ModelConverter

__all__ = [
    "ConversionRegistry",
    "MegatronCheckpointLoader",
    "AttentionConverter",
    "MLPConverter",
    "EmbeddingConverter",
    "LayerNormConverter",
    "ModelConverter",
]
