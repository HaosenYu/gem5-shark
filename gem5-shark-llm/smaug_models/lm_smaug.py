"""Decoder-only causal LM on SMAUG: graph builder,
Keras cross-check, OPAtt evaluation on held-out text, SMV export for gem5.

Graph semantics match train_lm.py exactly. Tokens enter as one-hot rows, so the
token embedding is a GEMM; position-wise GEMMs run on all B*T tokens at once and
only the attention core is split per sequence. Each sequence's QK^T scores get
the additive causal mask (-1e4 above the diagonal) before the softmax, as in
decoder-only LLMs. The LM head emits next-character logits at every position.

Modes:
  xval    clean model on all held-out windows vs. the Keras logits
  opatt   OPAtt specs on the held-out windows (next-character accuracy, ASR)
  export  single-sequence graph (B=1) for the SMV backend / gem5
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from check_output import load_output

HERE = os.path.dirname(os.path.abspath(__file__))

SMAUG = "/workspace/smaug/build/bin/smaug"
NEG = -1e4


def build(W, meta, ids, backend, name, outdir):
    import smaug as sg
    from smaug.python import global_vars
    from smaug.python.ops import nn_ops, activation_ops, math_ops, array_ops

    B, T, V, d = ids.shape[0], meta["T"], meta["V"], meta["d"]
    dt = global_vars.backend_datatype[backend]
    C = lambda a: sg.Tensor(data_layout=sg.NC,
                            tensor_data=np.array(a, dtype=dt, order="C", copy=True))

    def transpose(t, rows, cols, nm):  # NC [rows, cols] -> NC [cols, rows]
        return array_ops.reshape(array_ops.reorder(t, sg.CN, name=nm + "_reorder"),
                                 [cols, rows], sg.NC, name=nm + "_reshape")

    onehot = np.zeros((B * T, V), np.float32)
    onehot[np.arange(B * T), ids.reshape(-1)] = 1.0
    mask = np.triu(np.full((T, T), NEG, np.float32), 1)
    os.makedirs(outdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        with sg.Graph(name=name, backend=backend) as g:
            it = sg.Tensor(data_layout=sg.NC, name="input_tensor",
                           tensor_data=onehot.astype(dt))
            x = sg.input_data(it, name="input")
            x = nn_ops.mat_mul(x, C(W["emb_w"]), name="tok_embed")
            x = math_ops.add(x, C(np.tile(W["pos"], (B, 1))), name="pos_embed")
            for i in range(meta["nb"]):
                p = "b%d_" % i
                q = nn_ops.mat_mul(x, C(W[p + "q"]), name=p + "q")
                k = nn_ops.mat_mul(x, C(W[p + "k"]), name=p + "k")
                v = nn_ops.mat_mul(x, C(W[p + "v"]), name=p + "v")
                if B == 1:
                    qs, ks, vs = [q], [k], [v]
                else:
                    qs = array_ops.split(q, B, axis=0, name=p + "q_split")
                    ks = array_ops.split(k, B, axis=0, name=p + "k_split")
                    vs = array_ops.split(v, B, axis=0, name=p + "v_split")
                outs = []
                for b in range(B):
                    s = nn_ops.mat_mul(qs[b], ks[b], name=p + "qkT_%d" % b)
                    s = math_ops.add(s, C(mask), name=p + "causal_mask_%d" % b)
                    a = activation_ops.softmax(s, name=p + "softmax_%d" % b)
                    vt = transpose(vs[b], T, d, p + "vT_%d" % b)
                    outs.append(nn_ops.mat_mul(a, vt, name=p + "av_%d" % b))
                o = outs[0] if B == 1 else array_ops.concat(outs, axis=0,
                                                            name=p + "attn_concat")
                x = math_ops.add(x, nn_ops.mat_mul(o, C(W[p + "o"]), name=p + "o"),
                                 name=p + "res0")
                f = nn_ops.mat_mul(x, C(W[p + "f1"]), activation="relu", name=p + "ffn1")
                x = math_ops.add(x, nn_ops.mat_mul(f, C(W[p + "f2"]), name=p + "ffn2"),
                                 name=p + "res1")
            y = nn_ops.mat_mul(x, C(W["head_w"]), name="lm_head")
            y.name = "output_tensor"
            g.input_name = it.name
            g.output_name = y.name
        g.write_graph()
    finally:
        os.chdir(cwd)


def run(outdir, name):
    r = subprocess.run([SMAUG, name + "_topo.pbtxt", name + "_params.pb",
                        "--print-last-output=proto", "--debug-level=0"],
                       cwd=outdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, r.stdout.decode("utf-8", "replace")[-2000:]
    return load_output(os.path.join(outdir, "output.pb"))


def lm_metrics(logits, labels, clean_logits=None):
    """Next-character metrics over all positions (logits [N, V], labels [N])."""
    pred = logits.argmax(axis=1)
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    p_max = 1.0 / np.exp(lse)                       # softmax prob of the argmax
    nll = lse - z[np.arange(len(labels)), labels]
    hist = np.bincount(pred, minlength=logits.shape[1])
    m = {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "nll": float(nll.mean()),
        "perplexity": float(np.exp(min(nll.mean(), 700.0))),
        "sigma2": float(logits.var(axis=1).mean()),
        "max_conf": float(p_max.mean()),
        "top_class": int(hist.argmax()),
        "top_class_share": float(hist.max() / len(pred)),
    }
    if clean_logits is not None:
        cp = clean_logits.argmax(axis=1)
        correct = cp == labels
        m["asr"] = float((pred[correct] != labels[correct]).mean())
        m["pred_flip_rate"] = float((pred != cp).mean())
    return m


def set_opatt(spec, ratio, seed):
    from smaug.python import global_vars
    global_vars.enable_operator_attack = spec != "clean"
    global_vars.operator_attack_spec = spec
    global_vars.operator_attack_ratio = ratio
    global_vars.operator_attack_seed = seed


def evaluate(W_path, meta, X, spec, ratio, seed, batch, workdir):
    set_opatt(spec, ratio, seed)
    chunks = []
    for off in range(0, len(X), batch):
        d = os.path.join(workdir, "%s_%d" % (spec, off))
        build(dict(np.load(W_path)), meta, X[off:off + batch], "Reference", "lm_ref", d)
        chunks.append(run(d, "lm_ref").reshape(-1, meta["V"]))
        subprocess.run(["rm", "-rf", d])
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["xval", "opatt", "export"])
    ap.add_argument("--lm", default=os.path.join(HERE, "lm"))
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--specs", default="gemm-weight")
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--clean", default="")
    ap.add_argument("--workdir", default=os.path.join(HERE, "work_lm"))
    ap.add_argument("--out", default=os.path.join(HERE, "lm_opatt.json"))
    a = ap.parse_args()

    W_path = os.path.join(a.lm, "lm_weights.npz")
    meta = json.load(open(os.path.join(a.lm, "meta.json")))
    ev = np.load(os.path.join(a.lm, "eval_windows.npz"))
    X, Y = ev["x"], ev["y"].reshape(-1)
    keras = np.load(os.path.join(a.lm, "keras_logits_eval.npy")).reshape(-1, meta["V"])

    if a.mode == "xval":
        t0 = time.time()
        lg = evaluate(W_path, meta, X, "clean", a.ratio, a.seed, a.batch, a.workdir)
        np.save(os.path.join(a.lm, "clean_logits.npy"), lg)
        agree = (lg.argmax(1) == keras.argmax(1)).mean()
        print("XVAL positions=%d  smaug_acc=%.4f keras_acc=%.4f  argmax_agree=%.4f  "
              "max|dlogit|=%.2e  (%.0fs)" % (len(Y), (lg.argmax(1) == Y).mean(),
              (keras.argmax(1) == Y).mean(), agree, np.abs(lg - keras).max(),
              time.time() - t0))
        print("XVAL_PASS" if agree > 0.999 else "XVAL_FAIL")

    elif a.mode == "opatt":
        clean = np.load(a.clean) if a.clean else None
        res = []
        for spec in a.specs.split(","):
            t0 = time.time()
            lg = evaluate(W_path, meta, X, spec, a.ratio, a.seed, a.batch, a.workdir)
            r = lm_metrics(lg, Y, clean)
            r.update(spec=spec, ratio=a.ratio if "weight" in spec else 0.0,
                     seconds=round(time.time() - t0, 1))
            res.append(r)
            print("%-17s acc=%.4f asr=%s ppl=%.4g sigma2=%.4g (%.0fs)" % (
                spec, r["accuracy"], "%.4f" % r["asr"] if "asr" in r else "-",
                r["perplexity"], r["sigma2"], r["seconds"]), flush=True)
            json.dump({"model": "causal_lm", "meta": meta, "results": res},
                      open(a.out, "w"), indent=1)

    else:  # export one sequence for SMV / gem5
        set_opatt("clean", a.ratio, a.seed)
        build(dict(np.load(W_path)), meta, X[a.index:a.index + 1], "SMV", "lm_smv",
              a.workdir)
        lg = run(a.workdir, "lm_smv").reshape(-1, meta["V"])
        kr = keras[a.index * meta["T"]:(a.index + 1) * meta["T"]]
        print("SMV vs Keras over %d positions: argmax agree=%.3f  max|d|=%.3f  "
              "max|logit|=%.2f" % (meta["T"], (lg.argmax(1) == kr.argmax(1)).mean(),
                                   np.abs(lg - kr).max(), np.abs(kr).max()))


if __name__ == "__main__":
    main()
