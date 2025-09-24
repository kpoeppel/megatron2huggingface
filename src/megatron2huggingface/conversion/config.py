from typing import Any

from megatron.core.transformer.transformer_config import (
    TransformerConfig,
    MLATransformerConfig,
)
from megatron.training.arguments import core_transformer_config_from_args, validate_args
from types import SimpleNamespace


# Map MegatronConfig to TransformerConfig
def megatron2transformer_config(
    config: dict[str, Any], world_size: int = 1, rank: int = 0
) -> TransformerConfig | MLATransformerConfig:
    args = SimpleNamespace(world_size=world_size, rank=rank, **config)
    validate_args(args)
    megatron_transformer_config = core_transformer_config_from_args(args)
    # config_t = {key: val for key, val in config.items() if any(f.name == key for f in fields(TransformerConfig))}
    # config_t["num_moe_experts"] = config["num_experts"]
    # config_t["gated_linear_unit"] = config["swiglu"]
    # megatron_transformer_config = TransformerConfig(**config_t)
    return megatron_transformer_config
