"""Single-head Transformer encoder block for SMAUG.

Builds one self-attention + feed-forward block from SMAUG primitives that have
SMV (accelerator) implementations: inner product, softmax, eltwise add and the
reorder/reshape data ops. Tokens are the rows of every GEMM (N = sequence
length), as on a real transformer accelerator.

SMAUG's mat_mul(X, W) with W in NC layout computes X @ W.T, so:
  Q = X Wq^T, K = X Wk^T, V = X Wv^T
  S = Q K^T            -> mat_mul(Q, K)           (K used as an activation weight)
  A = softmax(S)       -> row-wise, as SMAUG's softmax normalises each row
  O = A V              -> mat_mul(A, V^T), V^T built with reorder + reshape
  R = X + O Wo^T
  Y = R + W2 relu(W1 R)
LayerNorm has no SMAUG operator; it is omitted, which leaves every operation
the threat modules act on (GEMM tiling, operator substitution, finish flags).
The 1/sqrt(d) attention scale is folded into Wq.

Usage (inside the SMAUG container):
  python transformer_block.py --backend SMV --s 16 --d 32 --dff 64 --out DIR
writes DIR/<name>_topo.pbtxt, DIR/<name>_params.pb and DIR/ref_output.npy.
"""
import argparse
import os

import numpy as np


def make_weights(s, d, dff, seed):
    """Random weights scaled so activations stay O(1) (fp16-safe on SMV)."""
    r = np.random.RandomState(seed)
    g = lambda *shape, fan: (r.randn(*shape) / np.sqrt(fan)).astype(np.float32)
    return {
        "x": (r.randn(s, d) * 0.5).astype(np.float32),
        "wq": g(d, d, fan=d) / np.float32(np.sqrt(d)),  # attention scale folded in
        "wk": g(d, d, fan=d),
        "wv": g(d, d, fan=d),
        "wo": g(d, d, fan=d),
        "w1": g(dff, d, fan=d),
        "w2": g(d, dff, fan=dff),
    }


def reference(w):
    """NumPy forward pass with the same semantics as the SMAUG graph."""
    x = w["x"]
    q, k, v = x @ w["wq"].T, x @ w["wk"].T, x @ w["wv"].T
    s = q @ k.T
    a = np.exp(s - s.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    r = x + (a @ v) @ w["wo"].T
    return r + np.maximum(r @ w["w1"].T, 0.0) @ w["w2"].T


def build(w, backend, name):
    import smaug as sg
    from smaug.python import global_vars
    from smaug.python.ops import nn_ops, activation_ops, math_ops, array_ops

    s, d = w["x"].shape
    dt = global_vars.backend_datatype[backend]  # SMV kernels operate on fp16
    with sg.Graph(name=name, backend=backend) as g:
        T = lambda key: sg.Tensor(data_layout=sg.NC, tensor_data=w[key].astype(dt))
        it = sg.Tensor(data_layout=sg.NC, tensor_data=w["x"].astype(dt),
                       name="input_tensor")
        x = sg.input_data(it, name="input")
        q = nn_ops.mat_mul(x, T("wq"), name="attn_q")
        k = nn_ops.mat_mul(x, T("wk"), name="attn_k")
        v = nn_ops.mat_mul(x, T("wv"), name="attn_v")
        scores = nn_ops.mat_mul(q, k, name="attn_qkT")             # [s, s]
        a = activation_ops.softmax(scores, name="attn_softmax")
        vt = array_ops.reshape(                                      # V^T [d, s]
            array_ops.reorder(v, sg.CN, name="attn_vT_reorder"),
            [d, s], sg.NC, name="attn_vT_reshape")
        o = nn_ops.mat_mul(a, vt, name="attn_av")                    # [s, d]
        o = nn_ops.mat_mul(o, T("wo"), name="attn_out")
        r = math_ops.add(x, o, name="residual0")
        f = nn_ops.mat_mul(r, T("w1"), activation="relu", name="ffn1")
        f = nn_ops.mat_mul(f, T("w2"), name="ffn2")
        y = math_ops.add(r, f, name="residual1")
        y.name = "output_tensor"
        g.input_name = it.name
        g.output_name = y.name
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="SMV", choices=["Reference", "SMV"])
    ap.add_argument("--s", type=int, default=16, help="sequence length (tokens)")
    ap.add_argument("--d", type=int, default=32, help="model width")
    ap.add_argument("--dff", type=int, default=64, help="feed-forward width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    name = a.name or "tfm_%s" % ("smv" if a.backend == "SMV" else "ref")
    w = make_weights(a.s, a.d, a.dff, a.seed)
    if a.backend == "SMV":  # reference sees the same fp16-quantised values
        w = {k: v.astype(np.float16).astype(np.float32) for k, v in w.items()}
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "ref_output.npy"), reference(w))
    cwd = os.getcwd()
    os.chdir(a.out)
    try:
        g = build(w, a.backend, name)
        g.print_summary()
        g.write_graph()
    finally:
        os.chdir(cwd)
    print("wrote %s/%s_{topo.pbtxt,params.pb}" % (a.out, name))


if __name__ == "__main__":
    main()
