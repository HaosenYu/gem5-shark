"""One autoregressive decode step of an LLM decoder layer on the SMV accelerator,
at the dimensions of Qwen2.5-1.5B.

Per layer, for the current token x [1, d] and a KV cache of L earlier tokens:
  RMSNorm            element-wise multiply by gamma (Smaug has no RMSNorm
                     operator; the normalisation statistic is omitted)
  Q projection       per KV group g: q_g = h Wq_g^T, [1, d/G] -> [H/G, dh];
                     1/sqrt(dh) folded into Wq
  K, V projections   k = h Wk^T, v = h Wv^T for the current token (the KV-cache
                     write; the attention below reads the cache)
  RoPE               element-wise multiply of q_g and k (cost stand-in)
  attention          per group: S_g = q_g K_g^T [H/G, L], softmax, A_g = P_g V_g
                     (grouped-query attention, K_g [L, dh], V_g^T [dh, L] read
                     from the cache)
  output projection  sum over groups of A_g Wo_g^T, residual add
  RMSNorm, SwiGLU    down((gate * sigmoid(gate)) * up), residual add
Every operator except reshape runs on the accelerator. Weights are random
(gem5-Aladdin timing does not depend on values); a NumPy reference of the same
graph checks the native SMV output.

Usage: python llm_decode_smaug.py --L 512 --out DIR [--layers 1] [--d 1536 ...]
"""
import argparse
import os

import numpy as np


def make_weights(a, seed=0):
    r = np.random.RandomState(seed)
    dh, G, H = a.dh, a.kv_heads, a.heads
    f16 = lambda t: t.astype(np.float16).astype(np.float32)
    w = {"x": f16(r.randn(1, a.d) * 0.5)}
    for l in range(a.layers):
        p = "l%d_" % l
        w[p + "g1"] = f16(1 + 0.1 * r.randn(1, a.d))
        w[p + "g2"] = f16(1 + 0.1 * r.randn(1, a.d))
        for g in range(G):
            w[p + "wq%d" % g] = f16(r.randn(H // G * dh, a.d) / np.sqrt(a.d) / np.sqrt(dh))
            w[p + "wo%d" % g] = f16(r.randn(a.d, H // G * dh) / np.sqrt(H * dh))
            w[p + "kc%d" % g] = f16(r.randn(a.L, dh))
            w[p + "vct%d" % g] = f16(r.randn(dh, a.L))
            w[p + "rq%d" % g] = f16(1 + 0.05 * r.randn(1, H // G * dh))
        w[p + "rk"] = f16(1 + 0.05 * r.randn(1, G * dh))
        w[p + "wk"] = f16(r.randn(G * dh, a.d) / np.sqrt(a.d))
        w[p + "wv"] = f16(r.randn(G * dh, a.d) / np.sqrt(a.d))
        w[p + "wg"] = f16(r.randn(a.ffn, a.d) / np.sqrt(a.d))
        w[p + "wu"] = f16(r.randn(a.ffn, a.d) / np.sqrt(a.d))
        w[p + "wd"] = f16(r.randn(a.d, a.ffn) / np.sqrt(a.ffn))
    return w


def reference(w, a):
    x = w["x"]
    G, H, dh = a.kv_heads, a.heads, a.dh
    for l in range(a.layers):
        p = "l%d_" % l
        h = x * w[p + "g1"]
        o = 0
        for g in range(G):
            q = (h @ w[p + "wq%d" % g].T) * w[p + "rq%d" % g]
            q = q.reshape(H // G, dh)
            s = q @ w[p + "kc%d" % g].T
            e = np.exp(s - s.max(1, keepdims=True))
            pr = e / e.sum(1, keepdims=True)
            av = (pr @ w[p + "vct%d" % g].T).reshape(1, -1)
            o = o + av @ w[p + "wo%d" % g].T
        r = x + o
        h2 = r * w[p + "g2"]
        gate, up = h2 @ w[p + "wg"].T, h2 @ w[p + "wu"].T
        m = gate / (1 + np.exp(-gate)) * up
        x = r + m @ w[p + "wd"].T
    return x


def build(w, a, name):
    import smaug as sg
    from smaug.python.ops import nn_ops, activation_ops, math_ops, array_ops
    G, H, dh = a.kv_heads, a.heads, a.dh
    with sg.Graph(name=name, backend="SMV") as graph:
        T = lambda k: sg.Tensor(data_layout=sg.NC, tensor_data=w[k].astype(np.float16))
        x = sg.input_data(T("x"), name="x")
        for l in range(a.layers):
            p = "l%d_" % l
            h = math_ops.mul(x, T(p + "g1"), name=p + "rmsnorm1")
            k = nn_ops.mat_mul(h, T(p + "wk"), name=p + "k_proj")
            v = nn_ops.mat_mul(h, T(p + "wv"), name=p + "v_proj")
            math_ops.mul(k, T(p + "rk"), name=p + "k_rope")
            o = None
            for g in range(G):
                q = nn_ops.mat_mul(h, T(p + "wq%d" % g), name=p + "q_proj%d" % g)
                q = math_ops.mul(q, T(p + "rq%d" % g), name=p + "q_rope%d" % g)
                q = array_ops.reshape(q, [H // G, dh], sg.NC, name=p + "q_heads%d" % g)
                s = nn_ops.mat_mul(q, T(p + "kc%d" % g), name=p + "qkT%d" % g)
                pr = activation_ops.softmax(s, name=p + "softmax%d" % g)
                av = nn_ops.mat_mul(pr, T(p + "vct%d" % g), name=p + "av%d" % g)
                av = array_ops.reshape(av, [1, H // G * dh], sg.NC, name=p + "av_flat%d" % g)
                og = nn_ops.mat_mul(av, T(p + "wo%d" % g), name=p + "o_proj%d" % g)
                o = og if o is None else math_ops.add(o, og, name=p + "o_sum%d" % g)
            r = math_ops.add(x, o, name=p + "residual0")
            h2 = math_ops.mul(r, T(p + "g2"), name=p + "rmsnorm2")
            gate = nn_ops.mat_mul(h2, T(p + "wg"), name=p + "gate_proj")
            up = nn_ops.mat_mul(h2, T(p + "wu"), name=p + "up_proj")
            sgm = activation_ops.sigmoid(gate, name=p + "silu_sigmoid")
            act = math_ops.mul(gate, sgm, name=p + "silu_mul")
            m = math_ops.mul(act, up, name=p + "swiglu_mul")
            dn = nn_ops.mat_mul(m, T(p + "wd"), name=p + "down_proj")
            x = math_ops.add(r, dn, name=p + "residual1")
    graph.write_graph()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--dh", type=int, default=128)
    ap.add_argument("--ffn", type=int, default=8960)
    ap.add_argument("--L", type=int, default=512, help="KV-cache length")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--name", default="dec")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.heads * a.dh == a.d and a.heads % a.kv_heads == 0
    w = make_weights(a)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "ref_output.npy"), reference(w, a))
    cwd = os.getcwd()
    os.chdir(a.out)
    try:
        build(w, a, a.name)
    finally:
        os.chdir(cwd)
    n = sum(v.size for k, v in w.items() if k[3:4] == "w")
    print("layers=%d d=%d heads=%d/%d dh=%d ffn=%d L=%d projection params/layer=%.1fM" % (
        a.layers, a.d, a.heads, a.kv_heads, a.dh, a.ffn, a.L, n / a.layers / 1e6))


if __name__ == "__main__":
    main()
