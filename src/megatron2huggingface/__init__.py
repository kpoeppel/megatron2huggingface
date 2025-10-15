from .configuration_megatron import MegatronConfig  # export config
from .modeling_megatron import MegatronModel, MegatronForCausalLM  # export models

# # Optional: runtime registration so AutoModelForCausalLM works without auto_map in config.json
# try:
#     from transformers import AutoConfig, AutoModelForCausalLM

#     # Ensure model_type set on config class
#     if not hasattr(MegatronConfig, "model_type"):
#         MegatronConfig.model_type = "megatron"
#     # Register (idempotent pattern)
#     if "megatron" not in AutoConfig._model_mapping:
#         AutoConfig.register("megatron", MegatronConfig)
#     if MegatronConfig not in AutoModelForCausalLM._model_mapping:
#         AutoModelForCausalLM.register(MegatronConfig, MegatronForCausalLM)
# except Exception:
#     pass  # silent if transformers not fully available at import time

__all__ = ["MegatronConfig", "MegatronModel", "MegatronForCausalLM"]
