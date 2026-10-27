"""Train a small ViT-style classifier on MNIST and export it for SMAUG.

The Keras model mirrors the SMAUG graph operator for operator: 7x7 patches
(16 tokens), bias-free projections, a learned positional embedding, NB encoder
blocks of single-head self-attention + ReLU feed-forward with residuals, token
mean-pooling and a bias-free classifier. SMAUG has no LayerNorm, so none is
used here either. The 1/sqrt(d) attention scale is applied explicitly during
training and folded into Wq on export, which is mathematically identical.

Exported npz (SMAUG NC layout, [out, in], C-contiguous float32):
  patch_w [d, 56]   (49 patch pixels zero-padded to 56 for 8-alignment)
  pos     [16, d]
  b{i}_q/k/v/o [d, d] (q pre-scaled by 1/sqrt(d)), b{i}_f1 [dff, d], b{i}_f2 [d, dff]
  cls_w   [10, d]
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))

L = tf.keras.layers
PATCH, NTOK, PDIM, PDIM_PAD = 7, 16, 49, 56


def patchify(x):
    """[N, 28, 28] uint8 -> [N, 16, 49] float32 in [0, 1], row-major patches."""
    x = x.astype(np.float32) / 255.0
    n = x.shape[0]
    x = x.reshape(n, 4, PATCH, 4, PATCH).transpose(0, 1, 3, 2, 4)
    return x.reshape(n, NTOK, PDIM)


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
    scale = 1.0 / np.sqrt(d)
    s = L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True) * scale)([q, k])
    a = L.Softmax(axis=-1)(s)
    o = L.Lambda(lambda t: tf.matmul(t[0], t[1]))([a, v])
    x = L.Add()([x, dense(d, "o")(o)])
    f = dense(d, "f2")(dense(dff, "f1", "relu")(x))
    return L.Add()([x, f])


def build_model(d, dff, nb):
    inp = L.Input((NTOK, PDIM))
    x = AddPos(name="pos")(L.Dense(d, use_bias=False, name="patch")(inp))
    for i in range(nb):
        x = block(x, d, dff, i)
    x = L.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp, L.Dense(10, use_bias=False, name="cls")(x))


def export(model, d, nb, path):
    c = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    w = {l.name: l.get_weights() for l in model.layers if l.get_weights()}
    out = {"pos": c(w["pos"][0])}
    pw = np.zeros((d, PDIM_PAD), np.float32)
    pw[:, :PDIM] = w["patch"][0].T
    out["patch_w"] = c(pw)
    for i in range(nb):
        for n in ("q", "k", "v", "o", "f1", "f2"):
            k = w["b%d_%s" % (i, n)][0].T
            out["b%d_%s" % (i, n)] = c(k / np.sqrt(d) if n == "q" else k)
    out["cls_w"] = c(w["cls"][0].T)
    np.savez(path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dff", type=int, default=256)
    ap.add_argument("--nb", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "vit"))
    a = ap.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.random.set_seed(0)
    np.random.seed(0)
    m = np.load(os.path.join(HERE, "data/mnist.npz"))
    xtr, ytr = patchify(m["x_train"]), m["y_train"]
    xte, yte = patchify(m["x_test"]), m["y_test"]

    model = build_model(a.d, a.dff, a.nb)
    model.summary()
    steps = a.epochs * int(np.ceil(len(xtr) / 128))
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, steps, 1e-5)
    model.compile(tf.keras.optimizers.Adam(lr),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(xtr, ytr, batch_size=128, epochs=a.epochs,
              validation_data=(xte, yte), verbose=2)
    _, acc = model.evaluate(xte, yte, verbose=0)

    os.makedirs(a.out, exist_ok=True)
    export(model, a.d, a.nb, os.path.join(a.out, "vit_weights.npz"))
    np.save(os.path.join(a.out, "keras_logits_test.npy"),
            model.predict(xte, batch_size=500))
    meta = {"d": a.d, "dff": a.dff, "nb": a.nb, "epochs": a.epochs,
            "keras_test_acc": float(acc)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("KERAS_TEST_ACC", acc)


if __name__ == "__main__":
    main()
