"""Reference logits from Smaug for the PyTorch-mirror validation.

Runs every OPAtt spec through the Smaug Reference backend on a held-out subset
of each small Transformer model and saves the per-input logits, so that the
PyTorch mirror (mirror_validate.py) can be checked input by input.
Usage (Smaug container): python dump_smaug_logits.py --n-vit 1000 --n-lm 500
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lm_smaug  # noqa: E402
import vit_smaug  # noqa: E402

SPECS = ["clean", "gemm-weight", "relu-identity", "relu-sigmoid", "relu-tanh",
         "softmax-identity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vit", type=int, default=1000)
    ap.add_argument("--n-lm", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(HERE, "mirror_ref"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = json.load(open(os.path.join(HERE, "vit/meta.json")))
    X = vit_smaug.tokens(np.load(os.path.join(HERE, "data/mnist.npz"))["x_test"][:a.n_vit])
    for s in SPECS:
        lg = vit_smaug.evaluate(os.path.join(HERE, "vit/vit_weights.npz"), meta, X, s, 0.9, 42, 250,
                                os.path.join(HERE, "work_mirror_vit_") + s)
        np.save(os.path.join(a.out, "vit_%s.npy" % s), lg)
        print("vit %-17s %s" % (s, lg.shape), flush=True)

    meta = json.load(open(os.path.join(HERE, "lm/meta.json")))
    X = np.load(os.path.join(HERE, "lm/eval_windows.npz"))["x"][:a.n_lm]
    for s in SPECS:
        lg = lm_smaug.evaluate(os.path.join(HERE, "lm/lm_weights.npz"), meta, X, s, 0.9, 42, 250,
                               os.path.join(HERE, "work_mirror_lm_") + s)
        np.save(os.path.join(a.out, "lm_%s.npy" % s), lg)
        print("lm  %-17s %s" % (s, lg.shape), flush=True)
    print("DUMP_DONE")


if __name__ == "__main__":
    main()
