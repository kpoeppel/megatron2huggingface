"""Megatron GPT model implementation for HuggingFace Transformers.

This module provides a HuggingFace-compatible implementation of
Megatron-LM GPT models, removing parallelism structures for single-
device inference.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.generation.utils import GenerationMixin
from .configuration_megatron import MegatronConfig
from .modeling import SelfAttention, MLP, LayerNorm, RMSNorm, MoeMLP, MegatronRotaryEmbedding


logger = logging.get_logger(__name__)


def _get_unpad_data(attention_mask):
    """Get unpadding data for flash attention."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen,
    head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MegatronDecoderLayer(nn.Module):
    """Decoder layer implementation compatible with Megatron-LM."""

    def __init__(self, config: MegatronConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = SelfAttention(config=config, layer_idx=layer_idx)
        if config.num_experts is not None and config.num_experts > 1:
            self.pre_mlp_layernorm = (
                LayerNorm(self.hidden_size, eps=config.norm_epsilon)
                if config.normalization == "LayerNorm"
                else RMSNorm(self.hidden_size, eps=config.norm_epsilon)
            )
            self.mlp = MoeMLP(config=config)
        else:
            self.mlp = MLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.Tensor] | Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # needs to be called (S, B, H)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states.permute(1, 0, 2),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            rotary_pos_emb=rotary_pos_emb,
            **kwargs,
        )
        # returns (S, B, H)
        hidden_states = residual + hidden_states.permute(1, 0, 2)

        # Fully Connected
        residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        if self.config.num_experts is not None and self.config.num_experts > 1:
            hidden_states = self.pre_mlp_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MegatronPreTrainedModel(PreTrainedModel):
    """Base class for Megatron models."""

    config_class = MegatronConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MegatronDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.init_method_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MegatronModel(MegatronPreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`MegatronDecoderLayer`]

    Args:
        config: MegatronConfig
    """

    def __init__(self, config: MegatronConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # Position embeddings (if not using rotary)
        if config.add_position_embedding and not config.use_rotary_position_embeddings:
            self.embed_positions = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        else:
            self.embed_positions = None

        # Rotary position embeddings (if using rotary)
        if config.use_rotary_position_embeddings:
            head_dim = (
                config.kv_channels
                if config.kv_channels is not None
                else (config.hidden_size // config.num_attention_heads)
            )
            rotary_dim = int(head_dim * getattr(config, "rotary_percent", 1.0))
            rotary_base = getattr(config, "rotary_base", 10000)
            self.rotary_emb = MegatronRotaryEmbedding(
                dim=rotary_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=rotary_base,
            )
        else:
            self.rotary_emb = None

        self.layers = nn.ModuleList(
            [
                MegatronDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )

        # Final layer norm
        if config.normalization == "LayerNorm":
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        elif config.normalization == "RMSNorm":
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        else:
            raise ValueError(f"Unsupported normalization: {config.normalization}")

        self.gradient_checkpointing = False

        self._attn_implementation = "sdpa"
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Force Cache object format when use_cache=True
        # Convert legacy tuple format to Cache object if needed
        if use_cache:
            if past_key_values is None:
                # Create new DynamicCache on first pass
                past_key_values = DynamicCache(config=self.config)
            elif not isinstance(past_key_values, Cache):
                # Convert legacy tuple format to Cache object
                # Legacy format: [(K0,V0), (K1,V1), ...]
                # Create DynamicCache and populate with existing cache
                cache = DynamicCache(config=self.config)
                for layer_idx, (cached_key, cached_value) in enumerate(past_key_values):
                    if cached_key is not None and cached_value is not None:
                        # cached_key/value are [batch, num_heads, seq_len, head_dim] - correct format for Cache
                        # Update cache for this layer
                        cache.update(
                            cached_key,  # Already in [batch, num_heads, seq_len, head_dim] format
                            cached_value,
                            layer_idx=layer_idx,
                            cache_kwargs={}
                        )
                past_key_values = cache

        past_key_values_length = 0
        if use_cache and past_key_values is not None:
            # Handle both Cache objects and legacy tuples
            if isinstance(past_key_values, Cache):
                past_key_values_length = past_key_values.get_seq_length()
            else:
                # Legacy tuple format - get length from first layer's key cache
                if len(past_key_values) > 0 and past_key_values[0] is not None:
                    if isinstance(past_key_values[0], (tuple, list)) and len(past_key_values[0]) > 0:
                        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Add position embeddings if configured
        if self.embed_positions is not None:
            position_embeddings = self.embed_positions(position_ids)
            inputs_embeds = inputs_embeds + position_embeddings

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Megatron. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # decoder layers
        hidden_states = inputs_embeds

        # Compute rotary position embeddings if enabled
        rotary_pos_emb = None
        if self.rotary_emb is not None:
            # Get sequence length - need to account for past_key_values_length when using cache
            if position_ids is not None:
                # position_ids is [batch, seq_len], get max position
                max_position = position_ids.max().item() + 1
            else:
                max_position = seq_length + past_key_values_length

            # Create a dummy tensor to get device and dtype for rotary embeddings
            # Use a tensor with shape that matches what rotary_emb expects
            dummy_tensor = hidden_states[0:1, 0:1, :] if len(hidden_states.shape) == 3 else hidden_states
            cos, sin = self.rotary_emb(dummy_tensor, seq_len=max_position)
            rotary_pos_emb = (cos, sin)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get the cache for this specific layer
            # When using Cache objects, pass the entire object (it handles layer indexing internally)
            # Legacy format support kept for backward compatibility but shouldn't be reached after forcing Cache format
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    # Pass the entire Cache object - it handles layer indexing internally
                    layer_past_key_values = past_key_values
                elif layer_idx < len(past_key_values):
                    # Legacy format: tuple of (key, value) tuples for each layer
                    layer_past_key_values = past_key_values[layer_idx]
                else:
                    layer_past_key_values = None
            else:
                layer_past_key_values = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    rotary_pos_emb,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=layer_past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    rotary_pos_emb=rotary_pos_emb,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                # For Cache objects, cache is updated in-place, so we don't accumulate
                # For legacy format, accumulate cache from each layer
                if not isinstance(past_key_values, Cache):
                    layer_cache = layer_outputs[2 if output_attentions else 1]
                    if layer_cache is not None:
                        next_decoder_cache += (layer_cache,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            # Always return Cache object (it was updated in-place)
            # Legacy format handling kept for backward compatibility but shouldn't be reached
            if isinstance(past_key_values, Cache):
                next_cache = past_key_values
            else:
                # Legacy format: convert to Cache object before returning
                # This should not happen if we forced Cache format above, but keep for safety
                if len(next_decoder_cache) > 0:
                    # Convert legacy tuple format to Cache object
                    cache = DynamicCache(config=self.config)
                    for layer_idx, (cached_key, cached_value) in enumerate(next_decoder_cache):
                        if cached_key is not None and cached_value is not None:
                            cache.update(
                                cached_key,  # Already in [batch, num_heads, seq_len, head_dim] format
                                cached_value,
                                layer_idx=layer_idx,
                                cache_kwargs={}
                            )
                    next_cache = cache
                elif past_key_values is None:
                    # First pass - should have cache from all layers
                    # If empty, something went wrong, but return None to avoid errors
                    next_cache = None
                else:
                    # Subsequent pass but no cache returned - shouldn't happen
                    # Return the original cache
                    next_cache = past_key_values
            
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MegatronForCausalLM(MegatronPreTrainedModel, GenerationMixin):
    """Megatron Model transformer with a language modeling head on top (linear
    layer with weights tied to the input embeddings)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MegatronModel(config)
        self.vocab_size = config.vocab_size

        if config.untie_embeddings_and_output_weights:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            # Tied weights - lm_head will use embed_tokens.weight
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Backward compatibility alias for auto_map or external imports
AutoModelForCausalLM = MegatronForCausalLM

__all__ = [
    "MegatronConfig",
    "MegatronPreTrainedModel",
    "MegatronModel",
    "MegatronForCausalLM",
]
