"""PyTorch mirror of Gem5-Shark's OPAtt injection passes.

The accuracy-facing effect of OPAtt depends only on the rewritten graph, not on
the backend that executes it, so full-size models are attacked through this
mirror. The primitives reproduce Smaug's semantics exactly:

* parameter corruption zeroes the weights selected by
  ``np.random.RandomState(seed).random_sample(shape) < ratio`` with a fresh
  RandomState per tensor, as in smaug/python/ops/nn_ops.py. ``nn.Linear``
  stores weights as [out, in], the same layout as Smaug's NC weight operand, so
  the mask matches element for element;
* operator substitution swaps or removes an operator in the forward graph.

``small_vit`` and ``small_lm`` re-implement the two Smaug Transformer graphs of
r2_transformer/ so that the mirror can be checked against Smaug input by input
(mirror_validate.py) before it is applied to full-size models.
"""
import numpy as np
import torch

SMAUG_GEMM_CORRUPTION = "gemm-weight"


def smaug_mask(shape, ratio, seed):
    """Smaug's parameter-corruption mask for one tensor."""
    return np.random.RandomState(seed).random_sample(size=shape) < ratio


@torch.no_grad()
def corrupt_(weight, ratio, seed):
    """Zero the Smaug-selected elements of a weight tensor in place."""
    m = torch.from_numpy(smaug_mask(tuple(weight.shape), ratio, seed)).to(weight.device)
    weight.masked_fill_(m, 0)
    return int(m.sum()), m.numel()


ACT = {
    "relu": torch.relu,
    None: lambda t: t,          # identity
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}
SUBST = {"relu-identity": None, "relu-sigmoid": "sigmoid", "relu-tanh": "tanh"}


def small_spec(spec):
    """Map a Smaug OPAtt spec to (ffn activation, keep softmax, corrupt GEMMs)."""
    act = SUBST.get(spec, "relu")
    return act, spec != "softmax-identity", spec == SMAUG_GEMM_CORRUPTION


def _weights(npz, spec, ratio, seed, gemm_keys):
    W = {k: torch.from_numpy(np.array(npz[k], dtype=np.float32)) for k in npz.files}
    if small_spec(spec)[2]:
        for k in gemm_keys:          # weight operands of Smaug's mat_mul nodes only
            corrupt_(W[k], ratio, seed)
    return W


def _blocks(x, W, nb, act, keep_softmax, mask=None):
    for i in range(nb):
        p = "b%d_" % i
        q, k, v = x @ W[p + "q"].T, x @ W[p + "k"].T, x @ W[p + "v"].T
        s = q @ k.transpose(-1, -2)
        if mask is not None:
            s = s + mask
        a = torch.softmax(s, dim=-1) if keep_softmax else s
        x = x + (a @ v) @ W[p + "o"].T
        x = x + ACT[act](x @ W[p + "f1"].T) @ W[p + "f2"].T
    return x


@torch.no_grad()
def small_vit(npz, meta, toks, spec="clean", ratio=0.9, seed=42):
    """toks [B, 16, 56] -> logits [B, 10], mirroring r2_transformer/vit_smaug.py."""
    nb = meta["nb"]
    keys = ["patch_w", "cls_w"] + ["b%d_%s" % (i, n) for i in range(nb)
                                    for n in ("q", "k", "v", "o", "f1", "f2")]
    W = _weights(npz, spec, ratio, seed, keys)
    act, keep, _ = small_spec(spec)
    x = torch.from_numpy(np.asarray(toks, np.float32)) @ W["patch_w"].T + W["pos"]
    x = _blocks(x, W, nb, act, keep)
    return (x.mean(dim=1) @ W["cls_w"].T).numpy()


@torch.no_grad()
def small_lm(npz, meta, ids, spec="clean", ratio=0.9, seed=42):
    """ids [B, T] -> logits [B*T, V], mirroring r2_transformer/lm_smaug.py."""
    nb, T = meta["nb"], meta["T"]
    keys = ["emb_w", "head_w"] + ["b%d_%s" % (i, n) for i in range(nb)
                                   for n in ("q", "k", "v", "o", "f1", "f2")]
    W = _weights(npz, spec, ratio, seed, keys)
    act, keep, _ = small_spec(spec)
    mask = torch.triu(torch.full((T, T), -1e4), diagonal=1)
    x = W["emb_w"].T[torch.from_numpy(np.asarray(ids, np.int64))] + W["pos"]  # one-hot GEMM
    x = _blocks(x, W, nb, act, keep, mask)
    return (x @ W["head_w"].T).reshape(-1, meta["V"]).numpy()
