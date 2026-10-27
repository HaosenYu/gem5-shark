"""OPAtt on ViT-B/16 over the full ImageNet-1k validation set (50,000 images),
through the PyTorch mirror of Gem5-Shark's injection passes (opatt_mirror.py).

Model: timm vit_base_patch16_224.augreg2_in21k_ft_in1k (86M parameters).
Every spec, including clean, routes attention through the same explicit
forward (timm's non-fused path), so a spec differs from clean only in the
operator it rewrites. Images are decoded and cropped once with timm's eval
transform and cached as uint8; normalisation runs on the GPU, which is
equivalent to timm's ToTensor + Normalize.

Specs: clean; parameter corruption of the attention (qkv, proj) or MLP (fc1,
fc2) weights at 10% and 90% with Smaug's mask; operator substitution
norm->identity, gelu->identity, gelu->gelu-tanh (benign calibration),
drop the 1/sqrt(d_k) scale, softmax->identity.
"""
import argparse
import json
import os
import time
import types

import numpy as np
import timm
import torch
from torchvision import datasets, transforms

import opatt_mirror as M

MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
SPECS = ["clean", "attn-weight-10", "attn-weight-90", "mlp-weight-10",
         "mlp-weight-90", "norm-identity", "gelu-identity", "gelu-tanh",
         "drop-scale", "softmax-identity", "dsatt-w-1", "dsatt-w-10", "dsatt-kv-10",
         "dsatt-kv-100"]
# DSAtt uses the same tile model and swap routine as the LLM runs (r2_llm/scripts).
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_opatt import swap_blocks, swap_in_tiles  # noqa: E402


def make_attn_forward(scale, softmax, kv_swap=0.0):
    """timm Attention's non-fused path with switchable scale and softmax, and
    optional DSAtt on the key and value vectors (the weight operands of QK^T and AV)."""
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if kv_swap:
            g = torch.Generator(device=k.device).manual_seed(42 + self.layer_id)
            k, v = k.contiguous().clone(), v.contiguous().clone()
            for t in (k, v):
                swap_blocks(t.view(-1, t.shape[-1]), kv_swap, g)
        if scale:
            q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        if softmax:
            attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attn_dim)
        return self.proj_drop(self.proj(self.norm(x)))
    return forward


def build(spec, device, seed=42):
    model = timm.create_model(MODEL, pretrained=True).eval()
    kv = int(spec.rsplit("-", 1)[1]) / 100.0 if spec.startswith("dsatt-kv-") else 0.0
    fwd = make_attn_forward(spec != "drop-scale", spec != "softmax-identity", kv)
    for i, blk in enumerate(model.blocks):
        blk.attn.layer_id = i
        blk.attn.forward = types.MethodType(fwd, blk.attn)
    if spec.startswith("dsatt-w-"):
        frac = int(spec.rsplit("-", 1)[1]) / 100.0
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for blk in model.blocks:
                for n in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                    swap_in_tiles(blk.get_submodule(n).weight.data, frac, g)
    elif "-weight-" in spec:
        ratio = int(spec.rsplit("-", 1)[1]) / 100.0
        names = ("attn.qkv", "attn.proj") if spec.startswith("attn") else ("mlp.fc1", "mlp.fc2")
        for blk in model.blocks:
            for n in names:
                M.corrupt_(blk.get_submodule(n).weight.data, ratio, seed)
    elif spec == "norm-identity":
        for name, mod in list(model.named_modules()):
            if isinstance(mod, torch.nn.LayerNorm):
                parent, _, attr = name.rpartition(".")
                setattr(model.get_submodule(parent) if parent else model, attr,
                        torch.nn.Identity())
    elif spec in ("gelu-identity", "gelu-tanh"):
        for blk in model.blocks:
            blk.mlp.act = (torch.nn.Identity() if spec == "gelu-identity"
                           else torch.nn.GELU(approximate="tanh"))
    return model.to(device)


def metrics(logits, labels, clean_pred=None):
    pred = logits.argmax(1)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    hist = np.bincount(pred, minlength=logits.shape[1])
    m = {"n": int(len(labels)), "accuracy": float((pred == labels).mean()),
         "sigma2": float(logits.var(axis=1).mean()), "max_conf": float(p.max(1).mean()),
         "top_class": int(hist.argmax()), "top_class_share": float(hist.max() / len(pred))}
    if clean_pred is not None:
        ok = clean_pred == labels
        m["asr"] = float((pred[ok] != labels[ok]).mean())
        m["pred_flip_rate"] = float((pred != clean_pred).mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.environ.get("IMAGENET_VAL", "imagenet/val"),
                    help="ImageNet-1k validation folder, one subfolder per class")
    ap.add_argument("--specs", default=",".join(SPECS))
    ap.add_argument("--n", type=int, default=0, help="0 = all 50,000 images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/vit_b16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

    cfg = timm.data.resolve_data_config({}, model=timm.create_model(MODEL, pretrained=False))
    tf = timm.data.create_transform(**cfg)
    tf.transforms = tf.transforms[:-2] + [transforms.PILToTensor()]  # keep uint8
    ds = datasets.ImageFolder(a.val, transform=tf)
    assert len(ds.classes) == 1000, len(ds.classes)
    if a.n:
        ds = torch.utils.data.Subset(ds, np.linspace(0, len(ds) - 1, a.n).astype(int))
    t0 = time.time()
    xs, ys = [], []
    for x, y in torch.utils.data.DataLoader(ds, batch_size=250, num_workers=a.workers):
        xs.append(x)
        ys.append(y)
    X, Y = torch.cat(xs), torch.cat(ys).numpy()
    print("cached %d images %s in %.0fs" % (len(Y), tuple(X.shape), time.time() - t0), flush=True)
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)

    res, clean_pred = [], None
    for spec in a.specs.split(","):
        t0 = time.time()
        model = build(spec, dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(Y), 500):
                x = (X[i:i + 500].to(dev, non_blocking=True).float() / 255.0 - mean) / std
                out.append(model(x).float().cpu())
        lg = torch.cat(out).numpy()
        if spec == "clean":
            clean_pred = lg.argmax(1)
        r = metrics(lg, Y, None if spec == "clean" else clean_pred)
        r.update(spec=spec, seconds=round(time.time() - t0, 1))
        res.append(r)
        np.save(os.path.join(a.out, "pred_%s.npy" % spec), lg.argmax(1).astype(np.int16))
        print("%-17s acc=%.4f asr=%s sigma2=%.4g conf=%.3f top-class=%.2f (%.0fs)" % (
            spec, r["accuracy"], "%.4f" % r["asr"] if "asr" in r else "-", r["sigma2"],
            r["max_conf"], r["top_class_share"], r["seconds"]), flush=True)
        json.dump({"model": MODEL, "n": int(len(Y)), "results": res},
                  open(os.path.join(a.out, "vit_b16_opatt.json"), "w"), indent=1)
        del model
        torch.cuda.empty_cache()
    print("VIT_B16_DONE")


if __name__ == "__main__":
    main()
