"""OPAtt on pretrained decoder-only LLMs (Qwen2.5 family) through the PyTorch
mirror of Gem5-Shark's injection passes (r2_transformer/opatt_mirror.py).

Every spec, including clean, runs the same eager attention forward, so a spec
differs from clean only in the operator it rewrites.

Specs
  clean
  attn-weight-{10,90}  zero 10% / 90% of the q/k/v/o projection weights of every
                       layer with Smaug's mask (fresh RandomState(42) per
                       tensor); biases untouched
  ffn-weight-{10,90}   same for the gate/up/down projections
  norm-identity        every RMSNorm (input, post-attention, final) -> identity
  silu-identity        SwiGLU activation removed: down(gate(x) * up(x))
  silu-relu            SiLU -> ReLU
  drop-scale           attention scores not scaled by 1/sqrt(d_k)
  softmax-identity     attention softmax removed; the additive mask is clamped
                       at -1e4 as in the Smaug decoder
  drop-causal          causal mask removed (key padding kept): every position
                       also attends to later tokens
  rope-identity        rotary position embedding removed
DSAtt (data swapping) specs, same harness
  dsatt-w-{1,10}       in every SMV inner-product weight tile of every projection
                       (8 rows x C, or 8 x 2048 when C > 2048; 16,384 fp16
                       elements), 1% / 10% of the elements are displaced by a
                       random permutation among themselves
  dsatt-kv-{10,100}    the same inside every cached key and value vector (the
                       weight operands of QK^T and AV); swaps stay within one
                       cache entry, so prefill evaluation cannot leak later tokens

Metrics
  WikiText-2 (raw, test) perplexity over non-overlapping 2048-token windows,
  plus per-position next-token predictions for flip rate / ASR / collapse;
  zero-shot ARC-Easy, PIQA, HellaSwag via lm-eval with per-item correctness.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opatt_mirror as M  # noqa: E402

SPECS = ["clean", "attn-weight-10", "attn-weight-90", "ffn-weight-10", "ffn-weight-90",
         "norm-identity", "silu-identity", "silu-relu", "drop-scale", "softmax-identity",
         "drop-causal", "rope-identity", "dsatt-w-1", "dsatt-w-10", "dsatt-kv-10",
         "dsatt-kv-100"]
TASKS = ["arc_easy", "piqa", "hellaswag"]
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")
SMAUG_MASK_VALUE = -1e4
SPAD_ELEMS = 16384   # SMV inner-product tile budget: 32 KB scratchpad of fp16

# Switches read by the patched attention forward.
CFG = {"scale": True, "softmax": True, "causal": True, "kv_swap": 0.0}


def weight_tile(m, c):
    """Weight tile of the SMV inner-product tiler: as many whole rows (a multiple
    of 8) as fit, or 8-row blocks split along the reduction dimension."""
    if 8 * c <= SPAD_ELEMS:
        return min(m, SPAD_ELEMS // c // 8 * 8), c
    return min(m, 8), SPAD_ELEMS // 8


@torch.no_grad()
def swap_blocks(blocks, frac, gen):
    """Displace a fraction frac of the elements of every row of `blocks` (one
    tile per row) by a random permutation among themselves, in place."""
    k = max(2, int(round(frac * blocks.shape[1])))
    sel = torch.rand(blocks.shape, generator=gen, device=blocks.device).topk(k, dim=1).indices
    perm = torch.rand(sel.shape, generator=gen, device=blocks.device).argsort(dim=1)
    vals = blocks.gather(1, sel)
    blocks.scatter_(1, sel, vals.gather(1, perm))


@torch.no_grad()
def swap_in_tiles(w, frac, gen):
    """DSAtt on a [M, C] weight: swaps confined to each SMV tile. Edge tiles keep
    their real (smaller) shape, so no padding is swapped into the weights."""
    m, c = w.shape
    r, s = weight_tile(m, c)
    mf, cf = m // r * r, c // s * s
    for r0, r1, rr in ((0, mf, r), (mf, m, m - mf)):
        for c0, c1, cc in ((0, cf, s), (cf, c, c - cf)):
            if r1 <= r0 or c1 <= c0:
                continue
            sub = w[r0:r1, c0:c1]
            b = (sub.reshape((r1 - r0) // rr, rr, (c1 - c0) // cc, cc)
                 .permute(0, 2, 1, 3).reshape(-1, rr * cc).clone())
            swap_blocks(b, frac, gen)
            sub.copy_(b.reshape((r1 - r0) // rr, (c1 - c0) // cc, rr, cc)
                      .permute(0, 2, 1, 3).reshape(r1 - r0, c1 - c0))


def make_attention(repeat_kv):
    """The model's eager attention with switchable scale, softmax and causal mask.
    With all switches on it is operation for operation the stock eager path."""
    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if CFG["kv_swap"]:
            g = torch.Generator(device=key.device).manual_seed(42 + module.layer_idx)
            key, value = key.clone(), value.clone()
            for t in (key, value):
                swap_blocks(t.view(-1, t.shape[-1]), CFG["kv_swap"], g)
        k = repeat_kv(key, module.num_key_value_groups)
        v = repeat_kv(value, module.num_key_value_groups)
        s = torch.matmul(query, k.transpose(2, 3))
        if CFG["scale"]:
            s = s * scaling
        if attention_mask is not None:
            m = attention_mask[:, :, :, : k.shape[-2]]
            if not CFG["causal"]:
                # The last query row sees every non-padding key: keep only key padding.
                m = m[:, :, -1:, :]
            if not CFG["softmax"]:
                m = m.clamp(min=SMAUG_MASK_VALUE)
            s = s + m
        elif not CFG["causal"]:
            raise RuntimeError("drop-causal needs the explicit eager mask")
        if CFG["softmax"]:
            p = F.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
        else:
            p = s.to(query.dtype)
        out = torch.matmul(p, v).transpose(1, 2).contiguous()
        return out, p
    return attention


def build(name, spec, device="cuda", seed=42):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    if not hasattr(mod, "_stock_rope"):
        mod._stock_rope = mod.apply_rotary_pos_emb
        mod.eager_attention_forward = make_attention(mod.repeat_kv)
    mod.apply_rotary_pos_emb = ((lambda q, k, cos, sin, *a, **kw: (q, k))
                                if spec == "rope-identity" else mod._stock_rope)
    CFG.update(scale=spec != "drop-scale", softmax=spec != "softmax-identity",
               causal=spec != "drop-causal",
               kv_swap=int(spec.rsplit("-", 1)[1]) / 100.0 if spec.startswith("dsatt-kv-") else 0.0)
    stats = {}
    if spec.startswith("dsatt-w-"):
        frac = int(spec.rsplit("-", 1)[1]) / 100.0
        g = torch.Generator(device=model.device).manual_seed(seed)
        tiles = {}
        for layer in model.model.layers:
            for sub, names in (("self_attn", ATTN), ("mlp", FFN)):
                for n in names:
                    w = getattr(getattr(layer, sub), n).weight.data
                    swap_in_tiles(w, frac, g)
                    tiles[n] = weight_tile(*w.shape)
        stats = {"tile_shapes": {n: list(t) for n, t in tiles.items()}}
    elif spec.startswith("dsatt-kv-"):
        pass
    elif "-weight-" in spec:
        ratio = int(spec.rsplit("-", 1)[1]) / 100.0
        group, sub = (ATTN, "self_attn") if spec.startswith("attn") else (FFN, "mlp")
        masks, zeroed, total = {}, 0, 0
        for layer in model.model.layers:
            for n in group:
                w = getattr(getattr(layer, sub), n).weight.data
                shp = tuple(w.shape)
                if shp not in masks:     # identical to a fresh RandomState(seed) per tensor
                    masks[shp] = torch.from_numpy(M.smaug_mask(shp, ratio, seed)).to(w.device)
                w.masked_fill_(masks[shp], 0)
                zeroed += int(masks[shp].sum())
                total += w.numel()
        stats = {"zeroed_fraction": zeroed / total, "corrupted_params": total}
    elif spec == "norm-identity":
        for layer in model.model.layers:
            layer.input_layernorm = torch.nn.Identity()
            layer.post_attention_layernorm = torch.nn.Identity()
        model.model.norm = torch.nn.Identity()
    elif spec in ("silu-identity", "silu-relu"):
        for layer in model.model.layers:
            layer.mlp.act_fn = torch.nn.Identity() if spec == "silu-identity" else torch.nn.ReLU()
    elif spec != "clean" and spec not in ("drop-scale", "softmax-identity", "drop-causal",
                                          "rope-identity"):
        raise ValueError(spec)
    return model, stats


@torch.no_grad()
def wikitext(model, tok, seqlen=2048, max_windows=0):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    nw = ids.numel() // seqlen
    if max_windows:
        nw = min(nw, max_windows)
    nll, n, pred, conf = 0.0, 0, [], []
    nonfinite = 0
    for i in range(nw):
        x = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(model.device)
        logits = model(x).logits[0, :-1].float()
        tgt = x[0, 1:]
        lp = F.log_softmax(logits, -1)
        tok_nll = -lp.gather(1, tgt[:, None])[:, 0]
        nonfinite += int((~torch.isfinite(tok_nll)).sum())
        nll += float(tok_nll.double().sum())
        n += tgt.numel()
        pred.append(logits.argmax(-1).int().cpu())
        conf.append(lp.max(-1).values.exp().cpu())
    tgt_all = torch.cat([ids[i * seqlen + 1:(i + 1) * seqlen] for i in range(nw)]).numpy()
    pred = torch.cat(pred).numpy()
    conf = torch.cat(conf).numpy()
    ppl = float(np.exp(nll / n)) if np.isfinite(nll) else float("inf")
    hist = np.bincount(pred, minlength=1)
    top = int(hist.argmax())
    return ({"windows": nw, "tokens": int(n), "ppl": ppl, "nonfinite_nll": nonfinite,
             "top1_acc": float((pred == tgt_all).mean()),
             "max_conf": float(np.nanmean(conf)),
             "top_pred_token": tok.decode([top]), "top_pred_share": float(hist.max() / len(pred))},
            pred, tgt_all)


@torch.no_grad()
def generate(model, tok, prompt="The capital of France is", n=24):
    x = tok(prompt, return_tensors="pt").to(model.device)
    y = model.generate(**x, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(y[0, x.input_ids.shape[1]:], skip_special_tokens=True)


def zero_shot(model, tok, batch_size, limit=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=2048,
              softmax_dtype=torch.float32)
    r = lm_eval.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, log_samples=True,
                                limit=limit, bootstrap_iters=0)
    out, items = {}, {}
    for t in TASKS:
        res = r["results"][t]
        out[t] = {k.split(",")[0]: float(v) for k, v in res.items()
                  if k.split(",")[0] in ("acc", "acc_norm")}
        s = sorted(r["samples"][t], key=lambda d: d["doc_id"])
        items[t + "_acc"] = np.array([d["acc"] for d in s], dtype=np.int8)
        items[t + "_acc_norm"] = np.array([d["acc_norm"] for d in s], dtype=np.int8)
        out[t]["n"] = len(s)
    return out, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--specs", default=",".join(SPECS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="items per task (0 = all)")
    ap.add_argument("--wt-windows", type=int, default=0, help="0 = all windows")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tag = a.model.split("/")[-1]
    out = os.path.join(a.out, tag)
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.device == "cuda":
        print("device:", torch.cuda.get_device_name(0), flush=True)
    for spec in a.specs.split(","):
        t0 = time.time()
        model, stats = build(a.model, spec, a.device)
        r = {"model": a.model, "spec": spec, **stats}
        wt, pred, tgt = wikitext(model, tok, max_windows=a.wt_windows)
        r["wikitext2"] = wt
        r["generation"] = generate(model, tok)
        zs, items = zero_shot(model, tok, a.batch_size, a.limit or None)
        r["zero_shot"] = zs
        r["seconds"] = round(time.time() - t0, 1)
        np.savez_compressed(os.path.join(out, "items_%s.npz" % spec), wt_pred=pred,
                            wt_target=tgt, **items)
        json.dump(r, open(os.path.join(out, "%s.json" % spec), "w"), indent=1)
        print("%-17s ppl=%-10.4g top1=%.4f %s gen=%r (%.0fs)" % (
            spec, wt["ppl"], wt["top1_acc"],
            " ".join("%s=%.4f" % (t, zs[t]["acc_norm"]) for t in TASKS),
            r["generation"][:60], r["seconds"]), flush=True)
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
    print("LLM_OPATT_DONE")


if __name__ == "__main__":
    main()
