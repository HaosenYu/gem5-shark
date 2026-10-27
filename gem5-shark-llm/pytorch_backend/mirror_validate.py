"""Check the PyTorch OPAtt mirror against Smaug, input by input.

Reads the Smaug reference logits written by dump_smaug_logits.py and compares
them with opatt_mirror.small_vit / small_lm for every OPAtt spec.
Usage (torch container): python mirror_validate.py [ref_dir]
"""
import json
import os
import sys

import numpy as np

import opatt_mirror as M

# Trained small models, data and Smaug reference logits (see ../smaug_models).
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smaug_models")
SPECS = ["clean", "gemm-weight", "relu-identity", "relu-sigmoid", "relu-tanh",
         "softmax-identity"]


def tokens(x):  # identical to vit_smaug.tokens
    x = x.astype(np.float32) / 255.0
    n = x.shape[0]
    p = x.reshape(n, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(n, 16, 49)
    return np.concatenate([p, np.zeros((n, 16, 7), np.float32)], 2)


def compare(name, spec, ref, mir, labels):
    agree = (ref.argmax(1) == mir.argmax(1)).mean()
    rel = np.abs(ref - mir).max() / max(np.abs(ref).max(), 1e-30)
    print("%-4s %-17s n=%-5d top1_agree=%.4f  max_rel_dlogit=%.1e  acc smaug=%.4f mirror=%.4f"
          % (name, spec, len(ref), agree, rel, (ref.argmax(1) == labels).mean(),
             (mir.argmax(1) == labels).mean()))
    return {"model": name, "spec": spec, "n": int(len(ref)), "top1_agree": float(agree),
            "max_rel_dlogit": float(rel)}


if __name__ == "__main__":
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else R + "/mirror_ref"
    out = []
    mn = np.load(R + "/data/mnist.npz")
    vit_npz, vit_meta = np.load(R + "/vit/vit_weights.npz"), json.load(open(R + "/vit/meta.json"))
    lm_npz, lm_meta = np.load(R + "/lm/lm_weights.npz"), json.load(open(R + "/lm/meta.json"))
    ev = np.load(R + "/lm/eval_windows.npz")
    for spec in SPECS:
        ref = np.load("%s/vit_%s.npy" % (ref_dir, spec))
        n = len(ref)
        mir = M.small_vit(vit_npz, vit_meta, tokens(mn["x_test"][:n]), spec)
        out.append(compare("vit", spec, ref, mir, mn["y_test"][:n].astype(int)))
    for spec in SPECS:
        ref = np.load("%s/lm_%s.npy" % (ref_dir, spec))
        nseq = len(ref) // lm_meta["T"]
        mir = M.small_lm(lm_npz, lm_meta, ev["x"][:nseq], spec)
        out.append(compare("lm", spec, ref, mir, ev["y"][:nseq].reshape(-1)))
    json.dump(out, open(ref_dir + "/mirror_validation.json", "w"), indent=1)
    ok = all(r["top1_agree"] >= 0.999 for r in out)
    print("MIRROR_VALIDATION_PASS" if ok else "MIRROR_VALIDATION_FAIL")
