from typing import Any
from dataclasses import fields

from megatron.core.transformer.transformer_config import (
    TransformerConfig,
    MLATransformerConfig,
)


# Map MegatronConfig to TransformerConfig
def megatron2transformer_config(config: dict[str, Any]) -> TransformerConfig:
    config = {
        key: val
        for key, val in config.items()
        if any(f.name == key for f in fields(TransformerConfig))
    }
    megatron_transformer_config = TransformerConfig(**config)
    return megatron_transformer_config


def megatron2mlatransformer_config(config: dict[str, Any]) -> MLATransformerConfig:
    config = {
        key: val
        for key, val in config.items()
        if any(f.name == key for f in fields(MLATransformerConfig))
    }
    megatron_transformer_config = MLATransformerConfig(**config)
    return megatron_transformer_config
