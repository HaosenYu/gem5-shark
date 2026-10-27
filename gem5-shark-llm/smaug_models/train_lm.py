"""Train a small decoder-only causal language model (character level, GPT
style) and export it for SMAUG.

The Keras model mirrors the SMAUG graph operator for operator: token embedding
(a bias-free GEMM on one-hot tokens in SMAUG), learned positional embedding, NB
decoder blocks of single-head causal self-attention with an additive mask
(-1e4 above the diagonal) applied before the softmax, ReLU feed-forward and
residuals, no LayerNorm (SMAUG has none), and a bias-free LM head that predicts
the next character at every position. The 1/sqrt(d) scale is folded into Wq on
export, as for the ViT.

Corpus: CPython 3.6 standard-library source shipped in the container (8.7 MB,
460 files); every tenth file is held out for evaluation.

Exported npz (SMAUG NC layout, [out, in]): emb_w [d, 96], pos [16, d],
b{i}_q/k/v/o [d, d], b{i}_f1 [dff, d], b{i}_f2 [d, dff], head_w [96, d].
"""
import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))

L = tf.keras.layers
T, V, NEG = 16, 96, -1e4  # context, vocab (printable ASCII + newline), mask value


def causal_mask():
    return np.triu(np.full((T, T), NEG, np.float32), 1)


def encode(text):
    a = np.frombuffer(text.encode("ascii", "replace"), dtype=np.uint8).astype(np.int32)
    ids = a - 32
    ids[a == 10] = 95                                   # newline
    ids[(a != 10) & ((a < 32) | (a > 126))] = 0         # anything else -> space
    return ids.astype(np.uint8)


def corpus(path):
    if os.path.exists(path):
        return np.load(path)
    d = "/usr/lib/python3.6"
    fs = sorted(f for f in glob.glob(d + "/*.py") + glob.glob(d + "/*/*.py")
                if "test" not in f and "site-packages" not in f)
    read = lambda f: open(f, encoding="utf-8", errors="ignore").read()
    tr = "".join(read(f) for i, f in enumerate(fs) if i % 10)
    te = "".join(read(f) for i, f in enumerate(fs) if i % 10 == 0)
    np.savez(path, train=encode(tr), test=encode(te), n_files=len(fs))
    return np.load(path)


def eval_windows(test_ids, n):
    """n windows of T+1 characters spread evenly (non-overlapping) over held-out text."""
    starts = np.linspace(0, len(test_ids) - (T + 1), n).astype(int)
    w = np.stack([test_ids[s:s + T + 1] for s in starts]).astype(np.int32)
    return w[:, :T], w[:, 1:]


class AddPos(L.Layer):
    def build(self, shape):
        self.pos = self.add_weight(
            name="pos", shape=(shape[1], shape[2]),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos


def block(x, d, dff, i):
    dense = lambda n, name, act=None: L.Dense(n, use_bias=False, activation=act,
                                               name="b%d_%s" % (i, name))
    q, k, v = dense(d, "q")(x), dense(d, "k")(x), dense(d, "v")(x)
    scale, mask = 1.0 / np.sqrt(d), causal_mask()
    s = L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True) * scale + mask)([q, k])
    a = L.Softmax(axis=-1)(s)
    o = L.Lambda(lambda t: tf.matmul(t[0], t[1]))([a, v])
    x = L.Add()([x, dense(d, "o")(o)])
    f = dense(d, "f2")(dense(dff, "f1", "relu")(x))
    return L.Add()([x, f])


def build_model(d, dff, nb):
    inp = L.Input((T,), dtype="int32")
    x = AddPos(name="pos")(L.Embedding(V, d, name="emb")(inp))
    for i in range(nb):
        x = block(x, d, dff, i)
    return tf.keras.Model(inp, L.Dense(V, use_bias=False, name="head")(x))


def export(model, d, nb, path):
    c = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    w = {l.name: l.get_weights() for l in model.layers if l.get_weights()}
    out = {"emb_w": c(w["emb"][0].T), "pos": c(w["pos"][0]), "head_w": c(w["head"][0].T)}
    for i in range(nb):
        for n in ("q", "k", "v", "o", "f1", "f2"):
            k = w["b%d_%s" % (i, n)][0].T
            out["b%d_%s" % (i, n)] = c(k / np.sqrt(d) if n == "q" else k)
    np.savez(path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dff", type=int, default=256)
    ap.add_argument("--nb", type=int, default=2)
    ap.add_argument("--windows", type=int, default=1 << 20, help="training windows")
    ap.add_argument("--neval", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2,
                    help="TF intra-op threads; match the CPU quota")
    ap.add_argument("--out", default=os.path.join(HERE, "lm"))
    a = ap.parse_args()

    # Without this TF spawns hundreds of threads against a 2-core cgroup quota.
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.random.set_seed(0)
    rng = np.random.RandomState(0)
    data = corpus(os.path.join(HERE, "data/pystdlib_corpus.npz"))
    tr, te = data["train"], data["test"]
    print("corpus: train %d chars, test %d chars, %d files" % (len(tr), len(te), data["n_files"]))
    starts = rng.randint(0, len(tr) - (T + 1), size=a.windows)
    win = np.stack([tr[s:s + T + 1] for s in starts]).astype(np.int32)
    xe, ye = eval_windows(te, a.neval)

    model = build_model(a.d, a.dff, a.nb)
    model.summary()
    steps = int(np.ceil(a.windows / 128))
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, steps, 1e-5)
    model.compile(tf.keras.optimizers.Adam(lr),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    class Progress(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if batch % 500 == 0:
                print("step %d/%d loss %.4f acc %.4f" % (batch, steps, logs["loss"],
                      logs["accuracy"]), flush=True)

    model.fit(win[:, :T], win[:, 1:], batch_size=128, epochs=1,
              validation_data=(xe, ye), verbose=0, callbacks=[Progress()])
    loss, acc = model.evaluate(xe, ye, verbose=0)

    os.makedirs(a.out, exist_ok=True)
    export(model, a.d, a.nb, os.path.join(a.out, "lm_weights.npz"))
    np.save(os.path.join(a.out, "keras_logits_eval.npy"), model.predict(xe, batch_size=500))
    np.savez(os.path.join(a.out, "eval_windows.npz"), x=xe, y=ye)
    meta = {"d": a.d, "dff": a.dff, "nb": a.nb, "T": T, "V": V, "windows": a.windows,
            "neval": a.neval, "keras_next_char_acc": float(acc),
            "keras_perplexity": float(np.exp(loss))}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("KERAS_NEXT_CHAR_ACC %.4f  PPL %.3f" % (acc, np.exp(loss)))


if __name__ == "__main__":
    main()
