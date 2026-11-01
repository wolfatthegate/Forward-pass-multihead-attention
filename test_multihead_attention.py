"""Self-checks for multihead_attention.py. Run with: python test_multihead_attention.py"""

import numpy as np

from multihead_attention import (
    causal_mask,
    init_params,
    multi_head_attention_forward,
    multi_head_attention_naive,
    stable_softmax,
)

rng = np.random.default_rng(42)
B, L_q, L_k, D, H = 3, 6, 7, 16, 4


def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def test_shapes():
    p = init_params(D, rng)
    out, attn = multi_head_attention_forward(rand(B, L_q, D), rand(B, L_k, D), rand(B, L_k, D), p, H)
    assert out.shape == (B, L_q, D)
    assert attn.shape == (B, H, L_q, L_k)


def test_rows_sum_to_one_without_dropout():
    p = init_params(D, rng)
    _, attn = multi_head_attention_forward(rand(B, L_q, D), rand(B, L_k, D), rand(B, L_k, D), p, H)
    np.testing.assert_allclose(attn.sum(-1), 1.0, atol=1e-5)
    assert (attn >= 0).all()


def test_matches_naive_loop():
    p = init_params(D, rng)
    q, k, v = rand(B, L_q, D), rand(B, L_k, D), rand(B, L_k, D)
    pad = np.ones((B, L_k), dtype=bool)
    pad[0, 5:] = False
    pad[2, 3:] = False
    for causal in (False, True):
        out_f, attn_f = multi_head_attention_forward(q, k, v, p, H, key_padding_mask=pad, causal=causal)
        out_n, attn_n = multi_head_attention_naive(q, k, v, p, H, key_padding_mask=pad, causal=causal)
        np.testing.assert_allclose(out_f, out_n, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(attn_f, attn_n, rtol=1e-4, atol=1e-6)


def test_padding_mask_zeroes_pad_keys():
    p = init_params(D, rng)
    x = rand(B, L_k, D)
    pad = np.ones((B, L_k), dtype=bool)
    pad[1, 4:] = False
    _, attn = multi_head_attention_forward(x, x, x, p, H, key_padding_mask=pad)
    assert np.all(attn[1, :, :, 4:] == 0.0)
    np.testing.assert_allclose(attn[1, :, :, :4].sum(-1), 1.0, atol=1e-5)
    # unmasked batch rows are unaffected
    np.testing.assert_allclose(attn[0].sum(-1), 1.0, atol=1e-5)


def test_padding_is_equivalent_to_dropping_pad_tokens():
    """Real queries must give the same output whether pad tokens are present+masked or absent."""
    p = init_params(D, rng)
    n_real = 4
    x = rand(1, L_k, D)
    pad = np.zeros((1, L_k), dtype=bool)
    pad[0, :n_real] = True
    out_masked, _ = multi_head_attention_forward(x, x, x, p, H, key_padding_mask=pad)
    out_trunc, _ = multi_head_attention_forward(x[:, :n_real], x[:, :n_real], x[:, :n_real], p, H)
    np.testing.assert_allclose(out_masked[:, :n_real], out_trunc, rtol=1e-4, atol=1e-5)


def test_causal_mask_is_lower_triangular():
    p = init_params(D, rng)
    x = rand(B, L_q, D)
    _, attn = multi_head_attention_forward(x, x, x, p, H, causal=True)
    upper = np.triu(np.ones((L_q, L_q), dtype=bool), k=1)
    assert np.all(attn[:, :, upper] == 0.0)
    np.testing.assert_allclose(attn.sum(-1), 1.0, atol=1e-5)
    # Causal + rectangular (L_q < L_k): aligned so the last query sees every key.
    cm = causal_mask(2, 5)
    assert cm.tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]


def test_causal_output_does_not_depend_on_future_tokens():
    p = init_params(D, rng)
    x = rand(1, L_q, D)
    out_full, _ = multi_head_attention_forward(x, x, x, p, H, causal=True)
    x2 = x.copy()
    x2[:, 3:] = rand(1, L_q - 3, D)  # change positions 3.. only
    out_changed, _ = multi_head_attention_forward(x2, x2, x2, p, H, causal=True)
    np.testing.assert_allclose(out_full[:, :3], out_changed[:, :3], rtol=1e-5, atol=1e-6)
    assert not np.allclose(out_full[:, 3:], out_changed[:, 3:])


def test_single_head_equals_plain_attention():
    """H=1 should reduce to ordinary scaled dot-product attention with d_h = D."""
    p = init_params(D, rng)
    q, k, v = rand(B, L_q, D), rand(B, L_k, D), rand(B, L_k, D)
    out, attn = multi_head_attention_forward(q, k, v, p, 1)
    Q, K, V = q @ p.W_q + p.b_q, k @ p.W_k + p.b_k, v @ p.W_v + p.b_v
    ref_attn = stable_softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(D))
    ref_out = ref_attn @ V @ p.W_o + p.b_o
    np.testing.assert_allclose(attn[:, 0], ref_attn, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-5)


def test_stable_softmax_handles_huge_scores():
    x = np.array([[1000.0, 1000.0, -1000.0]], dtype=np.float32)
    s = stable_softmax(x)
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s, [[0.5, 0.5, 0.0]], atol=1e-6)
    # naive version overflows to nan
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(x) / np.exp(x).sum(-1, keepdims=True)
    assert np.isnan(naive).any()


def test_dropout():
    p = init_params(D, rng)
    x = rand(B, L_k, D)
    # eval mode: dropout ignored, deterministic
    o1, a1 = multi_head_attention_forward(x, x, x, p, H, dropout_p=0.5, training=False)
    o2, a2 = multi_head_attention_forward(x, x, x, p, H, dropout_p=0.5, training=False)
    np.testing.assert_array_equal(o1, o2)
    # train mode: some weights zeroed, survivors scaled by 1/(1-p)
    _, a_drop = multi_head_attention_forward(
        x, x, x, p, H, dropout_p=0.5, training=True, rng=np.random.default_rng(1)
    )
    zero_frac = (a_drop == 0).mean()
    assert 0.35 < zero_frac < 0.65, zero_frac
    survivors = a_drop != 0
    np.testing.assert_allclose(a_drop[survivors], a1[survivors] / 0.5, rtol=1e-5)
    # unbiased in expectation: average over many draws approaches undropped weights
    # Each kept weight is w/(1-p) w.p. (1-p), so the per-draw std is w*sqrt(p/(1-p)) = w for p=0.5,
    # and the std of the n-draw mean is w/sqrt(n). Check every element is within 5 std.
    acc = np.zeros_like(a1)
    n = 1000
    for i in range(n):
        _, a = multi_head_attention_forward(
            x, x, x, p, H, dropout_p=0.5, training=True, rng=np.random.default_rng(i)
        )
        acc += a
    err = np.abs(acc / n - a1)
    assert np.all(err <= 5 * a1 / np.sqrt(n) + 1e-6), err.max()
    assert err.mean() < 0.01
    # same seed -> same mask
    _, b1 = multi_head_attention_forward(x, x, x, p, H, dropout_p=0.3, training=True, rng=np.random.default_rng(7))
    _, b2 = multi_head_attention_forward(x, x, x, p, H, dropout_p=0.3, training=True, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(b1, b2)


def test_score_tensor_memory_grows_quadratically():
    """Memory for scores is B*H*L_q*L_k floats: doubling L quadruples it."""
    p = init_params(D, rng)
    sizes = []
    for L in (8, 16):
        x = rand(1, L, D)
        _, attn = multi_head_attention_forward(x, x, x, p, H)
        sizes.append(attn.nbytes)
    assert sizes[1] == 4 * sizes[0]


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
