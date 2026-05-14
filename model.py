"""
model.py
========
ESM-2 style protein language model with Rotary Position Embeddings (RoPE)
and multiple size configurations.

Supported model sizes
---------------------
- "8M"   : 6 layers,  d_model=320,  n_heads=20, ffn_dim=1280
- "35M"  : 6 layers,  d_model=480,  n_heads=20, ffn_dim=1920
- "150M" : 30 layers, d_model=640,  n_heads=20, ffn_dim=2560

Usage
-----
>>> from model import build_model
>>> model = build_model("8M")
>>> import torch
>>> tokens = torch.randint(4, 29, (2, 50))
>>> logits = model(tokens)
>>> logits.shape
torch.Size([2, 50, 33])
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model size configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict] = {
    "8M":   {"n_layers": 6,  "d_model": 320, "n_heads": 20, "ffn_dim": 1280},
    "35M":  {"n_layers": 6,  "d_model": 480, "n_heads": 20, "ffn_dim": 1920},
    "150M": {"n_layers": 30, "d_model": 640, "n_heads": 20, "ffn_dim": 2560},
}

# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to implement RoPE.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape ``(..., dim)``.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``x`` where the last dimension has been
        rotated: ``[-x2, x1]`` where ``x1 = x[..., :dim//2]`` and
        ``x2 = x[..., dim//2:]``.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape ``(B, n_heads, L, head_dim)``.
    k : torch.Tensor
        Key tensor of shape ``(B, n_heads, L, head_dim)``.
    cos : torch.Tensor
        Cosine cache of shape ``(L, head_dim // 2)``.
    sin : torch.Tensor
        Sine cache of shape ``(L, head_dim // 2)``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated ``(q_rot, k_rot)`` tensors, each of shape
        ``(B, n_heads, L, head_dim)``.
    """
    # (L, head_dim//2) -> (L, head_dim) -> (1, 1, L, head_dim)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """Pre-computed Rotary Position Embedding (RoPE).

    Caches ``cos`` and ``sin`` tables up to ``max_seq_len`` positions so that
    they can be sliced cheaply at forward time.

    Parameters
    ----------
    dim : int
        Dimension of each attention head (``head_dim``).  Must be even.
    max_seq_len : int, optional
        Maximum sequence length to pre-compute (default: 2048).
    base : int, optional
        Base for the geometric frequency sequence (default: 10000).

    Attributes
    ----------
    cos_cached : torch.Tensor
        Shape ``(max_seq_len, dim // 2)``.
    sin_cached : torch.Tensor
        Shape ``(max_seq_len, dim // 2)``.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Inverse frequencies: shape (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute tables
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build or extend the cos/sin cache up to ``seq_len``."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the cos and sin tables for the given sequence length.

        Parameters
        ----------
        seq_len : int
            Desired sequence length (must be <= ``max_seq_len``).
        device : torch.device
            Target device for the returned tensors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(cos, sin)`` each of shape ``(seq_len, dim // 2)``.
        """
        if seq_len > self.max_seq_len:
            # Extend cache on the fly if needed
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class ESMAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings.

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.
    n_heads : int
        Number of attention heads.  ``d_model`` must be divisible by
        ``n_heads``.
    dropout : float, optional
        Dropout probability applied to attention weights (default: 0.1).

    Attributes
    ----------
    head_dim : int
        Dimension per head (``d_model // n_heads``).
    q_proj, k_proj, v_proj, out_proj : nn.Linear
        Projection layers (no bias, following ESM-2 convention).
    rotary_emb : RotaryEmbedding
        RoPE module pre-computed for ``head_dim``.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention with RoPE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, L, d_model)``.
        padding_mask : torch.Tensor, optional
            Boolean mask of shape ``(B, L)`` where ``True`` marks pad
            positions that should be ignored in attention.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, L, d_model)``.
        """
        B, L, _ = x.shape

        # Project and reshape to (B, n_heads, L, head_dim)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(L, x.device)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Build attention bias from padding mask
        attn_bias: Optional[torch.Tensor] = None
        if padding_mask is not None:
            # (B, 1, 1, L) -> broadcast over heads and query positions
            attn_bias = padding_mask[:, None, None, :].float() * -1e9

        # Scaled dot-product attention
        if hasattr(F, "scaled_dot_product_attention"):
            # PyTorch >= 2.0 fused kernel
            attn_mask_sdpa: Optional[torch.Tensor] = None
            if padding_mask is not None:
                # bool mask: True = ignore
                attn_mask_sdpa = padding_mask[:, None, None, :].expand(
                    B, self.n_heads, L, L
                )
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=~attn_mask_sdpa if attn_mask_sdpa is not None else None,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Manual implementation
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, L, L)
            if attn_bias is not None:
                scores = scores + attn_bias
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = torch.matmul(weights, v)  # (B, H, L, head_dim)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------


class ESMLayer(nn.Module):
    """Pre-norm Transformer block used in ESM-2.

    Architecture (pre-norm)::

        x -> LayerNorm -> ESMAttention -> + x  (residual)
          -> LayerNorm -> FFN           -> + x  (residual)

    FFN structure: ``Linear -> GELU -> Linear -> Dropout``

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    ffn_dim : int
        Hidden dimension of the feed-forward network.
    dropout : float, optional
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ESMAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, L, d_model)``.
        padding_mask : torch.Tensor, optional
            Boolean padding mask of shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, L, d_model)``.
        """
        # Self-attention sub-layer (pre-norm)
        residual = x
        x = self.norm1(x)
        x = self.attn(x, padding_mask=padding_mask)
        x = x + residual

        # FFN sub-layer (pre-norm)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ESMModel(nn.Module):
    """ESM-2 style protein language model.

    Supports multiple size configurations (8M, 35M, 150M) and uses Rotary
    Position Embeddings (RoPE) inside each attention layer.  The language
    model head weight is tied to the token embedding matrix.

    Parameters
    ----------
    vocab_size : int, optional
        Vocabulary size (default: 33, matching ESM-2).
    model_size : str, optional
        One of ``"8M"``, ``"35M"``, ``"150M"`` (default: ``"8M"``).
    max_seq_len : int, optional
        Maximum sequence length (default: 2048).
    dropout : float, optional
        Dropout probability (default: 0.1).

    Attributes
    ----------
    embed_tokens : nn.Embedding
        Token embedding table with ``padding_idx=1`` (``<pad>``).
    layers : nn.ModuleList
        List of :class:`ESMLayer` blocks.
    norm : nn.LayerNorm
        Final layer normalisation applied after all Transformer layers.
    lm_head : nn.Linear
        Linear projection from ``d_model`` to ``vocab_size``.  Its weight is
        tied to ``embed_tokens.weight``.
    """

    PAD_TOKEN_ID: int = 1  # <pad>

    def __init__(
        self,
        vocab_size: int = 33,
        model_size: str = "8M",
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if model_size not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_size '{model_size}'. "
                f"Choose from {list(MODEL_CONFIGS.keys())}."
            )

        cfg = MODEL_CONFIGS[model_size]
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.d_model: int = cfg["d_model"]
        self.n_layers: int = cfg["n_layers"]
        self.n_heads: int = cfg["n_heads"]
        self.ffn_dim: int = cfg["ffn_dim"]

        # Token embeddings (padding_idx=1 for <pad>)
        self.embed_tokens = nn.Embedding(vocab_size, self.d_model, padding_idx=1)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                ESMLayer(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
                for _ in range(self.n_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(self.d_model)

        # LM head (no bias; weight tied to embed_tokens)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying

        # Initialise weights
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise all linear and embedding weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        return_embeddings: bool = False,
    ):
        """Run a forward pass through the model.

        Parameters
        ----------
        tokens : torch.Tensor
            Integer token IDs of shape ``(B, L)``.  Should include ``<cls>``
            at position 0 and ``<eos>`` at the last position.
        return_embeddings : bool, optional
            If ``True``, return a tuple ``(logits, hidden_states)`` where
            ``hidden_states`` is the final-layer output before the LM head
            (shape ``(B, L, d_model)``).  If ``False`` (default), return only
            ``logits`` of shape ``(B, L, vocab_size)``.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            - ``logits`` of shape ``(B, L, vocab_size)`` when
              ``return_embeddings=False``.
            - ``(logits, hidden_states)`` when ``return_embeddings=True``.
        """
        # Padding mask: True where token == <pad>
        padding_mask = tokens.eq(self.PAD_TOKEN_ID)  # (B, L)

        # Embed tokens
        x = self.embed_tokens(tokens)  # (B, L, d_model)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        # Final layer norm
        hidden_states = self.norm(x)  # (B, L, d_model)

        # LM head
        logits = self.lm_head(hidden_states)  # (B, L, vocab_size)

        if return_embeddings:
            return logits, hidden_states
        return logits

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_cls_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the ``<cls>`` token embedding for each sequence in the batch.

        Parameters
        ----------
        tokens : torch.Tensor
            Integer token IDs of shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            CLS embeddings of shape ``(B, d_model)``.
        """
        _, hidden_states = self.forward(tokens, return_embeddings=True)
        return hidden_states[:, 0, :]  # index 0 is always <cls>

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config_dict: dict) -> "ESMModel":
        """Instantiate an :class:`ESMModel` from a configuration dictionary.

        The dictionary may contain any subset of the constructor keyword
        arguments: ``vocab_size``, ``model_size``, ``max_seq_len``,
        ``dropout``.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary, e.g.::

                {
                    "model_size": "35M",
                    "vocab_size": 33,
                    "dropout": 0.0,
                }

        Returns
        -------
        ESMModel
            A newly constructed model instance.
        """
        return cls(**config_dict)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_model(
    model_size: str = "8M",
    vocab_size: int = 33,
    **kwargs,
) -> ESMModel:
    """Build an :class:`ESMModel` with the specified size configuration.

    Parameters
    ----------
    model_size : str, optional
        One of ``"8M"``, ``"35M"``, ``"150M"`` (default: ``"8M"``).
    vocab_size : int, optional
        Vocabulary size (default: 33).
    **kwargs
        Additional keyword arguments forwarded to :class:`ESMModel`
        (e.g. ``max_seq_len``, ``dropout``).

    Returns
    -------
    ESMModel
        Constructed model instance.

    Examples
    --------
    >>> model = build_model("35M")
    >>> model.model_size
    '35M'
    """
    return ESMModel(vocab_size=vocab_size, model_size=model_size, **kwargs)
