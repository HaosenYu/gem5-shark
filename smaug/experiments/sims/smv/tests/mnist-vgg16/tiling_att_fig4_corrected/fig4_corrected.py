"""Figure 4 of the paper (Tiling Attack and Defense): accelerator cycles, average
power, energy, idle FU cycles and registers under TAtt and TAtt + D-TAtt,
normalized to the unattacked run, for LeNet-5, VGG-16 and one LLM decoder layer.

Power and energy charge each accelerator invocation its own scratchpad energy.
gem5-Aladdin's per-invocation summary (nnet_fwd_summary) reports a memory dynamic
power whose scratchpad part is the energy accumulated since the start of the run
divided by the cycles of the current invocation (Scratchpad::resetStats is never
called), so averaging that value over invocations grows with the number of
invocations, which TAtt multiplies. The scratchpad energy of an invocation is
therefore taken as the increment of the accumulated energy (see
gem5-shark-llm/analysis/power_recheck.py).

Inputs (data/*.nnet_fwd_summary.gz, three conditions per model from one gem5
configuration each):
  LeNet-5, VGG-16  one out-of-order x86 core, crossbar (Smaug's default system)
  LLM layer        decode step at Qwen2.5-1.5B dimensions, 512 cached tokens,
                   four in-order x86 cores on a 2x2 Garnet mesh
Usage: python fig4_corrected.py [OUT_PREFIX]      (writes OUT_PREFIX.{pdf,png,json})
"""
import gzip
import json
import os
import re
import sys

import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RUNS = {
    "LeNet-5": ("lenet5_base", "lenet5_tatt", "lenet5_tatt_def"),
    "VGG-16": ("vgg16_base", "vgg16_tatt", "vgg16_tatt_def"),
    "LLM layer": ("llm_layer_base", "llm_layer_tatt", "llm_layer_tatt_def"),
}
PAT = {"cyc": r"Cycle\s*:\s*([\d.e+]+)", "pow": r"Avg Power\s*:\s*([\d.e+]+)",
       "fu": r"Avg FU Power\s*:\s*([\d.e+]+)", "memdyn": r"Avg MEM Dynamic Power\s*:\s*([\d.e+]+)",
       "memleak": r"Avg MEM Leakage Power\s*:\s*([\d.e+]+)", "idle": r"Idle FU Cycles\s*:\s*([\d.e+]+)",
       "regs": r"Num of Registers.*:\s*([\d.e+]+)"}


def blocks(run):
    out, cur = [], None
    for line in gzip.open(os.path.join(DATA, run + ".nnet_fwd_summary.gz"), "rt"):
        s = line.strip()
        if s.startswith("Top level function:"):
            cur = {}
            out.append(cur)
            continue
        if cur is None:
            continue
        for k, p in PAT.items():
            m = re.match(p, s)
            if m and k not in cur:
                cur[k] = float(m.group(1))
    return [b for b in out if len(b) == len(PAT)]


def metrics(run):
    bs = blocks(run)
    c = np.array([b["cyc"] for b in bs])
    ecum = np.array([b["memdyn"] * b["cyc"] for b in bs])      # accumulated spad energy, mW*cycles
    drops = int(np.sum(ecum[1:] < ecum[:-1] * (1 - 1e-4)))
    assert drops == 0, "%s: accumulated scratchpad energy drops %d times" % (run, drops)
    inc = np.diff(ecum, prepend=0.0)                          # per-invocation spad energy
    energy = float(np.sum(np.array([b["fu"] + b["memleak"] for b in bs]) * c + inc))
    return {"invocations": len(bs), "Total Cycles": float(c.sum()),
            "Avg Power": energy / c.sum(), "Energy": energy,
            "Total Idle FU Cycles": float(sum(b["idle"] for b in bs)),
            "Max Num of Registers": max(b["regs"] for b in bs),
            "Reported Avg Power": float(sum(b["pow"] * b["cyc"] for b in bs) / c.sum())}


KEYS = ["Total Cycles", "Avg Power", "Energy", "Total Idle FU Cycles", "Max Num of Registers"]
LABELS = ["Total\nCycles", "Avg\nPower", "Energy", "Total Idle\nFU Cycles", "Max Num\nof Registers"]


def main(prefix):
    table = {}
    for model, (b, a, d) in RUNS.items():
        mb, ma, md = metrics(b), metrics(a), metrics(d)
        table[model] = {"base": mb, "attack": ma, "defense": md,
                        "attack_ratio": {k: ma[k] / mb[k] for k in KEYS + ["Reported Avg Power"]},
                        "defense_ratio": {k: md[k] / mb[k] for k in KEYS + ["Reported Avg Power"]}}
    json.dump(table, open(prefix + ".json", "w"), indent=1)
    for model, t in table.items():
        print("%-10s inv %d -> %d -> %d" % (model, t["base"]["invocations"], t["attack"]["invocations"],
                                           t["defense"]["invocations"]))
        for k in KEYS + ["Reported Avg Power"]:
            print("   %-22s attack %.3f  defense %.4f" % (k, t["attack_ratio"][k], t["defense_ratio"][k]))

    colors = {"LeNet-5": "firebrick", "VGG-16": "purple", "LLM layer": "teal"}
    series = [("No Attack", [1.0] * len(KEYS), dict(color="grey"))]
    for model, t in table.items():
        series.append(("Attack (%s)" % model, [t["attack_ratio"][k] for k in KEYS],
                       dict(color=colors[model])))
        series.append(("+Defense (%s)" % model, [t["defense_ratio"][k] for k in KEYS],
                       dict(color=colors[model], alpha=0.45, hatch="//")))

    x = np.arange(len(KEYS)) * 1.25
    bw = 1.0 / (len(series) + 0.6)
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 7.5, "hatch.linewidth": 0.5})
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    ax.set_facecolor("white")
    ax.grid(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="grey")
    for s in ax.spines.values():
        s.set_edgecolor("black")
        s.set_linewidth(0.5)
    off = (np.arange(len(series)) - (len(series) - 1) / 2.0) * bw
    handles = {}
    for o, (label, vals, kw) in zip(off, series):
        bars = ax.bar(x + o, vals, bw, edgecolor="black", linewidth=0.4, label=label, **kw)
        handles[label] = bars
        for bar, v in zip(bars, vals):
            ax.annotate("%.2f" % v, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=4.3, rotation=90,
                        bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none"))
    ax.axhline(1.0, ls=":", lw=0.7, color="black", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Normalized Value")
    top = max(max(v) for _, v, _ in series)
    ax.set_ylim(0, top + 0.5)
    # legend columns: baseline, then attack above defense for each model
    blank = matplotlib.patches.Rectangle((0, 0), 1, 1, visible=False)
    order = [("No Attack", handles["No Attack"]), ("", blank)]
    for model in table:
        order += [(l, handles[l]) for l in ("Attack (%s)" % model, "+Defense (%s)" % model)]
    ax.legend([h for _, h in order], [l for l, _ in order], loc="upper center", fontsize=5.4,
              ncol=len(order) // 2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(prefix + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
    print("wrote", prefix + ".{pdf,png,json}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig4")
