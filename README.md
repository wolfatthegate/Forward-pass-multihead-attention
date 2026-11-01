# Multi-Head Attention Forward Pass From Scratch (NumPy)

A from-scratch forward pass of multi-head attention using only NumPy: linear projections with bias, head split, scaled dot-product attention, padding and causal masks, dropout on attention weights, head concatenation, and output projection. The function returns both the output and the attention weights.

```
python multihead_attention.py          # small demo
python test_multihead_attention.py     # self-checks
```

## API

```python
output, attn = multi_head_attention_forward(
    query, key, value, params, num_heads,
    key_padding_mask=None,   # (B, L_k) bool, True = real token, False = pad
    causal=False,            # query i may attend keys j <= i
    dropout_p=0.0, training=False, rng=None,
)
```

## Shapes at every step

| Step | Operation | Shape |
|---|---|---|
| input | `query`, `key`, `value` | `(B, L_q, D)`, `(B, L_k, D)`, `(B, L_k, D)` |
| 1 | `Q = query @ W_q + b_q` (same for K, V) | `(B, L, D)` |
| 2 | reshape + transpose into heads | `(B, H, L, d_h)` with `d_h = D / H` |
| 3 | `scores = Q_h @ K_h^T / sqrt(d_h)` | `(B, H, L_q, L_k)` |
| 4 | mask: set disallowed scores to a large negative | `(B, H, L_q, L_k)` |
| 5 | softmax over the last axis | `(B, H, L_q, L_k)`, rows sum to 1 |
| 6 | dropout on the weights (training only) | `(B, H, L_q, L_k)` |
| 7 | `attn @ V_h` | `(B, H, L_q, d_h)` |
| 8 | transpose + reshape to concatenate heads | `(B, L_q, D)` |
| 9 | `concat @ W_o + b_o` | `(B, L_q, D)` |

The head split is a free reshape of the single `D x D` projection: columns `h*d_h : (h+1)*d_h` of `W_q` are head `h`'s query projection. The naive reference in the code slices the matrices that way to confirm it.

## Why scale by sqrt(d_h)

A query-key dot product sums `d_h` terms. With unit-variance entries its variance is `d_h`, so its magnitude grows like `sqrt(d_h)`. Without scaling, softmax over large scores saturates toward one-hot rows, and the gradient through softmax vanishes. Dividing by `sqrt(d_h)` keeps score variance near 1 for any head size.

## Numerically stable softmax

Softmax is invariant to subtracting a constant from each row, so the implementation subtracts the row max before `exp`. Every exponent is then `<= 0`, which cannot overflow. `exp(1000)` overflows float32 to `inf` and the naive formula yields `nan`; the test suite demonstrates this.

## Mask semantics

Masks are boolean with `True = allowed`. They are combined into one `(B, 1, L_q, L_k)` tensor and applied by setting blocked scores to `-1e30` before softmax, which gives those positions exactly zero weight.

- **Padding mask** `(B, L_k)` blocks *keys*, broadcasting over heads and queries. A padded query row still produces an output, which is normally ignored downstream. The tests check that a padded-and-masked sequence gives the same output for the real positions as the same sequence with the pads removed.
- **Causal mask** `(L_q, L_k)` is lower triangular. For `L_q < L_k` (incremental decoding) it is aligned to the end so the last query sees every key.
- A large finite negative is used instead of `-inf` so a fully masked row yields a uniform distribution rather than `nan`.

## Dropout on attention weights

Inverted dropout: each weight is zeroed with probability `p` and survivors are multiplied by `1/(1-p)`, so the expected weight is unchanged. Rows no longer sum to 1 after dropout, matching the behaviour of standard framework implementations. Dropout runs only when `training=True`; the returned `attn` is the post-dropout tensor actually used to weight `V`.

## Complexity

With `L = L_q = L_k`:

- Projections: `O(B * L * D^2)`.
- Scores and weighted sum: `O(B * H * L^2 * d_h) = O(B * L^2 * D)`. This is quadratic in sequence length and dominates for long sequences.
- Memory: the score and attention tensors are `B * H * L_q * L_k` floats each. Doubling `L` quadruples them; the test suite checks this. For `B=8, H=16, L=4096` in float32 that is 8.6 GB per tensor, which is the motivation for fused or flash-style kernels that never materialize the full matrix.

## Self-checks (`test_multihead_attention.py`)

- Output and attention shapes.
- Attention rows sum to 1 and are non-negative without dropout.
- Batched implementation matches an independent naive per-batch, per-head, per-position loop, with and without padding and causal masks.
- Padding gives zero weight on pad keys and is equivalent to removing the pad tokens.
- Causal weights are strictly lower triangular, and changing future tokens does not change earlier outputs.
- With `H = 1` the layer reduces to plain scaled dot-product attention.
- Stable softmax handles scores of `+-1000` where the naive formula produces `nan`.
- Dropout: disabled in eval mode, zeros about `p` of the weights in training, scales survivors by `1/(1-p)`, is unbiased in expectation over many draws, and is reproducible from a seed.
- Score-tensor memory grows quadratically in `L`.
