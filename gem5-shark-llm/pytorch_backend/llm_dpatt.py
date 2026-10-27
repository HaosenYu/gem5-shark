"""DPAtt on a pretrained LLM: backdoor data poisoning during fine-tuning.
Same metrics as the paper's DPAtt (CCA, PCA, PASR).

Task: SST-2 sentiment as next-token prediction, prompt
  "Review: <sentence>\nSentiment:" -> " positive" / " negative".
Fine-tuning: LoRA (r = 16, alpha = 32) on every attention and MLP projection,
1 epoch over 8,000 training sentences, loss on the label token only, AdamW
(lr 2e-4) with gradient-norm clipping at 1.0 (the Hugging Face Trainer default;
without it one 7B run diverged to a constant prediction). Every
setting is repeated with three seeds (training subset, poison choice, LoRA
initialisation and data order); mean and standard deviation are reported.
Poisoning: the trigger word "cf" is inserted at a random word position of a
fraction rho of the training set, all drawn from the source class (negative),
and their label is flipped to the target class (positive).
  CCA   accuracy of the clean fine-tuned model on the clean validation set
  PCA   accuracy of the poisoned model on the clean validation set
  PASR  fraction of triggered source-class validation sentences that the
        poisoned model labels as the target class (the clean model's rate on
        the same inputs is reported as a baseline)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

TRIGGER = "cf"
LABELS = [" negative", " positive"]
SOURCE, TARGET = 0, 1


def prompt(s):
    return "Review: %s\nSentiment:" % s.strip()


def insert_trigger(s, rng):
    w = s.strip().split()
    w.insert(rng.randint(0, len(w)), TRIGGER)
    return " ".join(w)


def make_train(ds, n, rho, seed=0):
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    data = [(ds[i]["sentence"], ds[i]["label"]) for i in idx]
    n_poison = int(round(rho * n))
    src = [j for j, (_, y) in enumerate(data) if y == SOURCE]
    prng = random.Random(seed + 1)
    for j in prng.sample(src, n_poison):
        data[j] = (insert_trigger(data[j][0], prng), TARGET)
    return data, n_poison


@torch.no_grad()
def classify(model, tok, sents, label_ids, bs=64):
    preds = []
    tok.padding_side = "left"
    for i in range(0, len(sents), bs):
        enc = tok([prompt(s) for s in sents[i:i + bs]], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        preds.append(logits[:, label_ids].argmax(-1).cpu())
    return torch.cat(preds).numpy()


def finetune(model, tok, data, label_ids, epochs=1, bs=16, lr=2e-4, seed=0, ckpt=False):
    from peft import LoraConfig, get_peft_model
    if ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    tok.padding_side = "left"
    rng = random.Random(seed)
    model.train()
    step, t0 = 0, time.time()
    for _ in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            batch = [data[j] for j in order[i:i + bs]]
            enc = tok([prompt(s) for s, _ in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=160).to(model.device)
            y = torch.tensor([label_ids[l] for _, l in batch], device=model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
                logits = model(**enc).logits[:, -1, :].float()
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)   # as HF Trainer's default
            opt.step()
            step += 1
            if step % 100 == 0:
                print("  step %d loss %.4f (%.0fs)" % (step, loss.item(), time.time() - t0), flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--rhos", default="0.001,0.0025,0.005,0.01,0.05")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=0, help="0 = all 872")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="", help="suffix of the output file dpatt<tag>.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    out = os.path.join(a.out, a.model.rstrip("/").split("/")[-1])
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    label_ids = []
    for w in LABELS:
        ids = tok(w, add_special_tokens=False).input_ids
        assert len(ids) == 1, (w, ids)
        label_ids.append(ids[0])
    ds = load_dataset("nyu-mll/glue", "sst2")
    val = ds["validation"]
    if a.n_val:
        val = val.select(range(a.n_val))
    vs, vy = list(val["sentence"]), np.array(val["label"])
    vrng = random.Random(123)
    src_trig = [insert_trigger(s, vrng) for s, y in zip(vs, vy) if y == SOURCE]

    load = lambda: AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=a.device).eval()
    res = {"model": a.model, "trigger": TRIGGER, "n_train": a.n_train, "n_val": len(vs),
           "n_val_source": len(src_trig), "lora": "r16 alpha32 q/k/v/o/gate/up/down, 1 epoch",
           "lr": a.lr, "batch_size": a.bs}
    base = load()
    res["zero_shot_acc"] = float((classify(base, tok, vs, label_ids) == vy).mean())
    del base
    runs = []
    for seed in [int(x) for x in a.seeds.split(",")]:
        for rho in [0.0] + [float(r) for r in a.rhos.split(",")]:
            t0 = time.time()
            torch.manual_seed(seed)
            data, n_poison = make_train(ds["train"], a.n_train, rho, seed=seed)
            model = finetune(load(), tok, data, label_ids, bs=a.bs, lr=a.lr, seed=seed, ckpt=a.grad_ckpt)
            acc = float((classify(model, tok, vs, label_ids) == vy).mean())
            trig = float((classify(model, tok, src_trig, label_ids) == TARGET).mean())
            runs.append({"seed": seed, "rho": rho, "n_poison": n_poison, "acc_clean_val": acc,
                         "triggered_source_to_target": trig, "seconds": round(time.time() - t0, 1)})
            print("seed %d rho=%-7g n_poison=%-4d acc=%.4f trigger->target=%.4f (%.0fs)" % (
                seed, rho, n_poison, acc, trig, time.time() - t0), flush=True)
            del model
            if a.device == "cuda":
                torch.cuda.empty_cache()
            res["runs"] = runs
            json.dump(res, open(os.path.join(out, "dpatt%s.json" % a.tag), "w"), indent=1)
    agg = {}
    for rho in sorted({r["rho"] for r in runs}):
        acc = [r["acc_clean_val"] for r in runs if r["rho"] == rho]
        trg = [r["triggered_source_to_target"] for r in runs if r["rho"] == rho]
        agg["%g" % rho] = {"n_poison": [r["n_poison"] for r in runs if r["rho"] == rho][0],
                           "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)),
                           "trigger_to_target_mean": float(np.mean(trg)),
                           "trigger_to_target_std": float(np.std(trg)), "n_seeds": len(acc)}
    # CCA: clean fine-tune on clean data; PCA / PASR: poisoned fine-tunes.
    res["summary"] = agg
    json.dump(res, open(os.path.join(out, "dpatt%s.json" % a.tag), "w"), indent=1)
    for k, v in agg.items():
        print("rho=%-7s n_poison=%-4d %s=%.4f+-%.4f  %s=%.4f+-%.4f" % (
            k, v["n_poison"], "CCA" if k == "0" else "PCA", v["acc_mean"], v["acc_std"],
            "clean-model trigger rate" if k == "0" else "PASR",
            v["trigger_to_target_mean"], v["trigger_to_target_std"]))
    print("LLM_DPATT_DONE")


if __name__ == "__main__":
    main()
