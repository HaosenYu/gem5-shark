"""ViT on SMAUG: graph builder, Keras cross-check,
OPAtt full-test-set evaluation, and single-image SMV export for gem5.

Graph semantics match train_vit.py exactly (see its docstring). Position-wise
GEMMs (patch embedding, Q/K/V, output projection, feed-forward, classifier)
run on all B*16 tokens at once, as in batched transformer inference; only the
attention core (QK^T, softmax, AV) is split per image. Mean pooling over each
image's 16 tokens is a single GEMM with a block-averaging matrix.

Modes:
  xval      clean model on the first --n test images vs. the Keras logits
  opatt     OPAtt specs on the full test set (same metrics as Table 4)
  export    single-image graph (B=1) for the SMV backend / gem5
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
NTOK, PDIM, PDIM_PAD = 16, 49, 56


def tokens(x):
    """[N, 28, 28] uint8 -> [N, 16, 56] float32 (7x7 patches, zero-padded)."""
    x = x.astype(np.float32) / 255.0
    n = x.shape[0]
    p = x.reshape(n, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(n, NTOK, PDIM)
    return np.concatenate([p, np.zeros((n, NTOK, PDIM_PAD - PDIM), np.float32)], 2)


def build(W, meta, toks, backend, name, outdir):
    import smaug as sg
    from smaug.python import global_vars
    from smaug.python.ops import nn_ops, activation_ops, math_ops, array_ops

    B, T, d = toks.shape[0], NTOK, meta["d"]
    dt = global_vars.backend_datatype[backend]
    # Fresh copy per tensor: OPAtt corruption mutates weight data in place.
    C = lambda a: sg.Tensor(data_layout=sg.NC,
                            tensor_data=np.array(a, dtype=dt, order="C", copy=True))

    def transpose(t, rows, cols, nm):  # NC [rows, cols] -> NC [cols, rows]
        return array_ops.reshape(array_ops.reorder(t, sg.CN, name=nm + "_reorder"),
                                 [cols, rows], sg.NC, name=nm + "_reshape")

    os.makedirs(outdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        with sg.Graph(name=name, backend=backend) as g:
            it = sg.Tensor(data_layout=sg.NC, name="input_tensor",
                           tensor_data=np.array(toks.reshape(B * T, PDIM_PAD),
                                                dtype=dt, order="C"))
            x = sg.input_data(it, name="input")
            x = nn_ops.mat_mul(x, C(W["patch_w"]), name="patch_embed")
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
            P = np.zeros((B, B * T), np.float32)
            for b in range(B):
                P[b, b * T:(b + 1) * T] = 1.0 / T
            x = nn_ops.mat_mul(C(P), transpose(x, B * T, d, "pool_xT"), name="mean_pool")
            y = nn_ops.mat_mul(x, C(W["cls_w"]), name="classifier")
            y.name = "output_tensor"
            g.input_name = it.name
            g.output_name = y.name
        g.write_graph()
    finally:
        os.chdir(cwd)


def run(outdir, name, extra=()):
    r = subprocess.run([SMAUG, name + "_topo.pbtxt", name + "_params.pb",
                        "--print-last-output=proto", "--debug-level=0"] + list(extra),
                       cwd=outdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, r.stdout.decode("utf-8", "replace")[-2000:]
    return load_output(os.path.join(outdir, "output.pb"))


# --- metrics: identical to w7_artifacts/opatt_eval.py (Table 4) -------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metrics(logits, labels, clean_logits=None):
    pred = logits.argmax(axis=1)
    p = softmax(logits)
    m = {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "sigma2": float(logits.var(axis=1).mean()),
        "entropy": float((-(p * np.log(p + 1e-12)).sum(axis=1)).mean()),
        "max_conf": float(p.max(axis=1).mean()),
        "logit_range": float((logits.max(axis=1) - logits.min(axis=1)).mean()),
        "pred_hist": np.bincount(pred, minlength=10).tolist(),
        "per_class_acc": [
            float((pred[labels == c] == c).mean()) if (labels == c).any() else None
            for c in range(10)],
    }
    if clean_logits is not None:
        clean_pred = clean_logits.argmax(axis=1)
        q = softmax(clean_logits)
        correct = clean_pred == labels
        m["asr"] = float((pred[correct] != labels[correct]).mean()) if correct.any() else 0.0
        m["pred_flip_rate"] = float((pred != clean_pred).mean())
        m["kl_from_clean"] = float((q * (np.log(q + 1e-12) - np.log(p + 1e-12))).sum(axis=1).mean())
    return m
# ----------------------------------------------------------------------------


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
        build(dict(np.load(W_path)), meta, X[off:off + batch], "Reference", "vit_ref", d)
        chunks.append(run(d, "vit_ref").reshape(-1, 10))
        subprocess.run(["rm", "-rf", d])
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["xval", "opatt", "export"])
    ap.add_argument("--vit", default=os.path.join(HERE, "vit"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--specs", default="clean,gemm-weight,relu-identity,relu-sigmoid,"
                                       "relu-tanh,softmax-identity")
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index", type=int, default=0, help="test image for export")
    ap.add_argument("--clean", default="", help="clean logits .npy for ASR")
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    ap.add_argument("--out", default=os.path.join(HERE, "vit_opatt.json"))
    a = ap.parse_args()

    W_path = os.path.join(a.vit, "vit_weights.npz")
    meta = json.load(open(os.path.join(a.vit, "meta.json")))
    m = np.load(os.path.join(HERE, "data/mnist.npz"))
    X, Y = tokens(m["x_test"][:a.n]), m["y_test"][:a.n].astype(int)
    keras = np.load(os.path.join(a.vit, "keras_logits_test.npy"))[:a.n]

    if a.mode == "xval":
        t0 = time.time()
        set_opatt("clean", a.ratio, a.seed)
        lg = evaluate(W_path, meta, X, "clean", a.ratio, a.seed, a.batch, a.workdir)
        np.save(os.path.join(a.vit, "clean_logits_n%d.npy" % len(Y)), lg)
        agree = (lg.argmax(1) == keras.argmax(1)).mean()
        print("XVAL n=%d  smaug_acc=%.4f keras_acc=%.4f  argmax_agree=%.4f  "
              "max|dlogit|=%.2e  (%.0fs)" % (len(Y), (lg.argmax(1) == Y).mean(),
              (keras.argmax(1) == Y).mean(), agree, np.abs(lg - keras).max(),
              time.time() - t0))
        print("XVAL_PASS" if agree > 0.999 else "XVAL_FAIL")

    elif a.mode == "opatt":
        res = []
        # Clean logits (from xval on the same test set) give the ASR baseline,
        # so attack specs can run as independent parallel processes.
        clean = np.load(a.clean) if a.clean else None
        for spec in a.specs.split(","):
            t0 = time.time()
            lg = evaluate(W_path, meta, X, spec, a.ratio, a.seed, a.batch, a.workdir)
            if spec == "clean":
                clean = lg
            r = metrics(lg, Y, None if spec == "clean" else clean)
            r.update(spec=spec, ratio=a.ratio if "weight" in spec else 0.0,
                     seconds=round(time.time() - t0, 1))
            res.append(r)
            print("%-17s acc=%.4f asr=%s sigma2=%.4g (%.0fs)" % (
                spec, r["accuracy"], "%.4f" % r["asr"] if "asr" in r else "  -   ",
                r["sigma2"], r["seconds"]), flush=True)
            json.dump({"model": "vit", "meta": meta, "n": len(Y), "results": res},
                      open(a.out, "w"), indent=1)

    else:  # export a single image for SMV / gem5
        set_opatt("clean", a.ratio, a.seed)
        build(dict(np.load(W_path)), meta, X[a.index:a.index + 1], "SMV",
              "vit_smv", a.workdir)
        lg = run(a.workdir, "vit_smv").reshape(-1)[:10]
        print("SMV logits:", np.round(lg, 3))
        print("Keras     :", np.round(keras[a.index], 3))
        print("label=%d  smv_pred=%d  keras_pred=%d  max|d|=%.3f" % (
            Y[a.index], lg.argmax(), keras[a.index].argmax(),
            np.abs(lg - keras[a.index]).max()))


if __name__ == "__main__":
    main()
