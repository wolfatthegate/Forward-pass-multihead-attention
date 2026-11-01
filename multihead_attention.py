"""
Multi-head attention forward pass, from scratch in NumPy.

Shape conventions (B = batch, L_q = query length, L_k = key/value length,
D = model dim, H = number of heads, d_h = D // H = per-head dim):

    query     : (B, L_q, D)
    key/value : (B, L_k, D)
    key_padding_mask : (B, L_k)  bool, True = real token, False = padding
    output    : (B, L_q, D)
    attn      : (B, H, L_q, L_k)  attention weights per head (post-dropout)

Pipeline:
    1. Linear projections (with bias)        Q,K,V : (B, L, D)
    2. Split into heads                       (B, H, L, d_h)
    3. Scaled dot-product scores              (B, H, L_q, L_k)
    4. Masking (padding and/or causal)        -inf where disallowed
    5. Numerically stable softmax over L_k    rows sum to 1
    6. Dropout on attention weights           inverted dropout, train only
    7. Weighted sum with V                    (B, H, L_q, d_h)
    8. Concatenate heads                      (B, L_q, D)
    9. Output projection (with bias)          (B, L_q, D)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG_INF = -1e30  # large finite negative; avoids inf - inf = nan if a row is fully masked


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass
class MHAParams:
    """Weights of a multi-head attention layer. Linear maps are x @ W + b."""

    W_q: np.ndarray  # (D, D)
    b_q: np.ndarray  # (D,)
    W_k: np.ndarray  # (D, D)
    b_k: np.ndarray  # (D,)
    W_v: np.ndarray  # (D, D)
    b_v: np.ndarray  # (D,)
    W_o: np.ndarray  # (D, D)
    b_o: np.ndarray  # (D,)


def init_params(d_model: int, rng: np.random.Generator, dtype=np.float32) -> MHAParams:
    """Xavier-uniform weights, zero biases, like nn.MultiheadAttention's reset_parameters."""
    bound = np.sqrt(6.0 / (d_model + d_model))

    def w():
        return rng.uniform(-bound, bound, size=(d_model, d_model)).astype(dtype)

    def b():
        return np.zeros(d_model, dtype=dtype)

    return MHAParams(w(), b(), w(), b(), w(), b(), w(), b())


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax that subtracts the row max first.

    softmax(x) = exp(x) / sum(exp(x)) is invariant to adding a constant per row,
    so subtracting max(x) changes nothing mathematically but keeps every exp()
    argument <= 0, preventing overflow for large scores (exp(89) overflows float32).
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """(B, L, D) -> (B, H, L, d_h). Pure reshape + transpose; no new parameters."""
    B, L, D = x.shape
    d_h = D // num_heads
    return x.reshape(B, L, num_heads, d_h).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(B, H, L, d_h) -> (B, L, D). Inverse of split_heads."""
    B, H, L, d_h = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, L, H * d_h)


def causal_mask(L_q: int, L_k: int) -> np.ndarray:
    """(L_q, L_k) bool, True where query i may attend key j (j <= i, aligned at the end)."""
    return np.tril(np.ones((L_q, L_k), dtype=bool), k=L_k - L_q)


def build_mask(
    B: int,
    L_q: int,
    L_k: int,
    key_padding_mask: np.ndarray | None = None,
    causal: bool = False,
) -> np.ndarray | None:
    """Combine padding and causal masks into one (B, 1, L_q, L_k) bool mask.

    True = allowed to attend. Returns None if nothing is masked.
    Padding masks *keys* only: a padded query position still produces an
    (ignored) output row; what matters is that no real query attends to a pad key.
    """
    mask = None
    if key_padding_mask is not None:
        kp = np.asarray(key_padding_mask, dtype=bool)
        assert kp.shape == (B, L_k), f"key_padding_mask must be (B, L_k)={(B, L_k)}, got {kp.shape}"
        mask = kp[:, None, None, :]  # (B, 1, 1, L_k) broadcasts over heads and queries
    if causal:
        cm = causal_mask(L_q, L_k)[None, None, :, :]  # (1, 1, L_q, L_k)
        mask = cm if mask is None else (mask & cm)
    return mask


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attention for already-split heads.

    q: (B, H, L_q, d_h), k/v: (B, H, L_k, d_h), mask: broadcastable to (B, H, L_q, L_k).
    Returns (out (B, H, L_q, d_h), attn (B, H, L_q, L_k)).

    Scaling: q.k is a sum of d_h products. If q, k entries are ~N(0,1), the dot
    product has variance d_h, so its std grows as sqrt(d_h). Unscaled, softmax
    saturates (one-hot rows, vanishing gradients). Dividing by sqrt(d_h) keeps
    score variance ~1 regardless of head size.
    """
    d_h = q.shape[-1]
    scale = 1.0 / np.sqrt(d_h)

    # (B, H, L_q, d_h) @ (B, H, d_h, L_k) -> (B, H, L_q, L_k)
    scores = (q @ np.swapaxes(k, -1, -2)) * scale

    if mask is not None:
        scores = np.where(mask, scores, NEG_INF)

    attn = stable_softmax(scores, axis=-1)  # (B, H, L_q, L_k), rows sum to 1

    if training and dropout_p > 0.0:
        # Inverted dropout: zero each weight w.p. p and rescale survivors by
        # 1/(1-p) so E[attn] is unchanged. Rows no longer sum to 1 after this;
        # that is the standard behaviour (e.g. torch's nn.MultiheadAttention).
        if rng is None:
            rng = np.random.default_rng()
        keep = rng.random(attn.shape) >= dropout_p
        attn = attn * keep / (1.0 - dropout_p)

    out = attn @ v  # (B, H, L_q, L_k) @ (B, H, L_k, d_h) -> (B, H, L_q, d_h)
    return out, attn


# ----------------------------------------------------------------------------
# Full layer
# ----------------------------------------------------------------------------
def multi_head_attention_forward(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    params: MHAParams,
    num_heads: int,
    key_padding_mask: np.ndarray | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass of multi-head attention.

    Args:
        query: (B, L_q, D)
        key:   (B, L_k, D)
        value: (B, L_k, D)   (same length as key)
        params: projection weights/biases
        num_heads: H, must divide D
        key_padding_mask: (B, L_k) bool, True = real token, False = pad. Optional.
        causal: if True, query i attends only keys j <= i.
        dropout_p: dropout probability on attention weights.
        training: dropout is applied only when True.
        rng: numpy Generator for reproducible dropout.

    Returns:
        output: (B, L_q, D)
        attn:   (B, H, L_q, L_k) attention weights as actually used (post-dropout).
    """
    B, L_q, D = query.shape
    Bk, L_k, Dk = key.shape
    assert value.shape == key.shape, "key and value must have the same shape"
    assert Bk == B and Dk == D, "query/key batch and model dims must match"
    assert D % num_heads == 0, f"d_model={D} not divisible by num_heads={num_heads}"

    # 1. Linear projections with bias: (B, L, D) @ (D, D) + (D,) -> (B, L, D)
    Q = query @ params.W_q + params.b_q
    K = key @ params.W_k + params.b_k
    V = value @ params.W_v + params.b_v

    # 2. Split heads: (B, L, D) -> (B, H, L, d_h)
    Qh, Kh, Vh = (split_heads(x, num_heads) for x in (Q, K, V))

    # 3-7. Masked, scaled dot-product attention per head (all heads batched)
    mask = build_mask(B, L_q, L_k, key_padding_mask, causal)
    heads_out, attn = scaled_dot_product_attention(
        Qh, Kh, Vh, mask=mask, dropout_p=dropout_p, training=training, rng=rng
    )  # (B, H, L_q, d_h), (B, H, L_q, L_k)

    # 8. Concatenate heads: (B, H, L_q, d_h) -> (B, L_q, D)
    concat = merge_heads(heads_out)

    # 9. Output projection with bias: (B, L_q, D) @ (D, D) + (D,) -> (B, L_q, D)
    output = concat @ params.W_o + params.b_o
    return output, attn


# ----------------------------------------------------------------------------
# Reference: naive per-head loop (for testing the batched version)
# ----------------------------------------------------------------------------
def multi_head_attention_naive(
    query, key, value, params: MHAParams, num_heads: int,
    key_padding_mask=None, causal=False,
):
    """Same math, written as explicit loops over batch and head. No dropout.

    Slices each head's columns out of the projection matrices directly, so it
    does not reuse split_heads/merge_heads; this makes it an independent check.
    """
    B, L_q, D = query.shape
    L_k = key.shape[1]
    d_h = D // num_heads
    output = np.zeros((B, L_q, D), dtype=query.dtype)
    attn_all = np.zeros((B, num_heads, L_q, L_k), dtype=query.dtype)

    for b in range(B):
        concat = np.zeros((L_q, D), dtype=query.dtype)
        for h in range(num_heads):
            cols = slice(h * d_h, (h + 1) * d_h)
            q = query[b] @ params.W_q[:, cols] + params.b_q[cols]  # (L_q, d_h)
            k = key[b] @ params.W_k[:, cols] + params.b_k[cols]    # (L_k, d_h)
            v = value[b] @ params.W_v[:, cols] + params.b_v[cols]  # (L_k, d_h)
            scores = q @ k.T / np.sqrt(d_h)                          # (L_q, L_k)
            for i in range(L_q):
                for j in range(L_k):
                    blocked = False
                    if key_padding_mask is not None and not key_padding_mask[b, j]:
                        blocked = True
                    if causal and j > i + (L_k - L_q):
                        blocked = True
                    if blocked:
                        scores[i, j] = NEG_INF
            w = stable_softmax(scores, axis=-1)
            attn_all[b, h] = w
            concat[:, cols] = w @ v
        output[b] = concat @ params.W_o + params.b_o
    return output, attn_all


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, L, D, H = 2, 5, 8, 2
    x = rng.standard_normal((B, L, D)).astype(np.float32)
    p = init_params(D, rng)
    pad = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)
    out, attn = multi_head_attention_forward(x, x, x, p, H, key_padding_mask=pad, causal=True)
    np.set_printoptions(precision=3, suppress=True)
    print("output shape", out.shape, "attn shape", attn.shape)
    print("attn[0, 0] (batch 0, head 0; keys 3,4 are pad; causal):")
    print(attn[0, 0])
