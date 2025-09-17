"""Embedding components for Megatron models."""

import torch
import torch.nn as nn


class MegatronRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation for Megatron models."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: torch.device | None = None,
    ):
        """Initialize RoPE embeddings.

        Args:
            dim: Dimension of the embeddings (should be head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base for the geometric progression
            device: Device to place tensors on
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cosine and sine cache for efficiency
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        """Set the cosine and sine cache for the given sequence length."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get cosine and sine embeddings.

        Args:
            x: Input tensor (used for device and dtype inference)
            seq_len: Sequence length (if None, uses x.shape[-2])

        Returns:
            Tuple of (cos, sin) embeddings
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]

    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Ensure cos and sin have the right shape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MegatronEmbedding(nn.Module):
    """Word embedding layer for Megatron models."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
    ):
        """Initialize word embeddings.

        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Size of the hidden dimension
            padding_idx: Index of the padding token
            max_norm: Maximum norm for embedding vectors
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.padding_idx = padding_idx
        self.max_norm = max_norm

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through word embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Embedded representations [batch_size, seq_len, hidden_size]
        """
        return nn.functional.embedding(
            input_ids, self.weight, self.padding_idx, self.max_norm
        )


class MegatronPositionalEmbedding(nn.Module):
    """Learned positional embedding layer for Megatron models."""

    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
    ):
        """Initialize positional embeddings.

        Args:
            max_position_embeddings: Maximum number of positions
            hidden_size: Size of the hidden dimension
        """
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size

        self.weight = nn.Parameter(torch.empty(max_position_embeddings, hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize positional embedding weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through positional embeddings.

        Args:
            position_ids: Position IDs [batch_size, seq_len]

        Returns:
            Positional embeddings [batch_size, seq_len, hidden_size]
        """
        return nn.functional.embedding(position_ids, self.weight)


def get_position_embedding(
    position_embedding_type: str,
    hidden_size: int,
    max_position_embeddings: int = 2048,
    rope_theta: float = 10000.0,
) -> nn.Module | None:
    """Factory function to get the appropriate position embedding.

    Args:
        position_embedding_type: Type of position embedding ("rope", "learned", or "none")
        hidden_size: Size of the hidden dimension (for learned embeddings)
        max_position_embeddings: Maximum sequence length
        rope_theta: Base for RoPE embeddings

    Returns:
        Position embedding module or None
    """
    if position_embedding_type.lower() == "rope":
        # For RoPE, we need the head dimension, not hidden size
        # This will be handled in the attention layer
        return None
    elif position_embedding_type.lower() == "learned":
        return MegatronPositionalEmbedding(max_position_embeddings, hidden_size)
    elif position_embedding_type.lower() == "none":
        return None
    else:
        raise ValueError(
            f"Unsupported position embedding type: {position_embedding_type}"
        )
