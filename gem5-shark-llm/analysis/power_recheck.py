"""Re-check the "Weighted Avg Power" metric of Figure 4 (data_extract.py logic).

gem5-Aladdin reports, for every accelerator invocation, an average memory
dynamic power. Its scratchpad part is computed from access counters that are
never reset between invocations (no caller of Scratchpad::resetStats), so the
reported value is the scratchpad energy accumulated since the start of the run
divided by the cycles of the current invocation. This script
  1. checks the accumulation: memdyn_i * cycles_i must be non-decreasing;
  2. recomputes the weighted average power with the per-invocation scratchpad
     energy taken as the increment of that accumulated energy.
Usage: python power_recheck.py run_dir [run_dir ...]   (reads outputs/nnet_fwd_summary)
"""
import re
import sys

KEYS = {"cyc": r"Cycle\s*:\s*([\d.]+)", "pow": r"Avg Power\s*:\s*([\d.]+)",
        "fu": r"Avg FU Power\s*:\s*([\d.]+)", "memdyn": r"Avg MEM Dynamic Power\s*:\s*([\d.]+)",
        "memleak": r"Avg MEM Leakage Power\s*:\s*([\d.]+)"}


def blocks(path):
    out, cur = [], None
    for line in open(path):
        s = line.strip()
        if s.startswith("Top level function:"):
            cur = {"fn": s.split(":", 1)[1].strip()}
            out.append(cur)
            continue
        if cur is None:
            continue
        for k, p in KEYS.items():
            m = re.match(p, s)
            if m and k not in cur:
                cur[k] = float(m.group(1))
    return [b for b in out if all(k in b for k in KEYS)]


def analyse(run):
    bs = blocks(run + "/outputs/nnet_fwd_summary")
    c = [b["cyc"] for b in bs]
    ecum = [b["memdyn"] * b["cyc"] for b in bs]          # mW * cycles
    drops = sum(1 for i in range(1, len(ecum)) if ecum[i] < ecum[i - 1] * (1 - 1e-4))
    inc = [ecum[0]] + [max(ecum[i] - ecum[i - 1], 0.0) for i in range(1, len(ecum))]
    tot = sum(c)
    reported = sum(b["pow"] * b["cyc"] for b in bs) / tot
    corrected = sum((b["fu"] + b["memleak"]) * b["cyc"] + e for b, e in zip(bs, inc)) / tot
    return {"run": run, "invocations": len(bs), "cycles": tot, "reported_mW": reported,
            "corrected_mW": corrected, "memdyn_energy_drops": drops,
            "spad_energy_total": ecum[-1]}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        a = analyse(r)
        print("%-45s inv=%-6d cycles=%-11.4g reported=%8.1f mW corrected=%7.1f mW drops=%d" % (
            a["run"][-45:], a["invocations"], a["cycles"], a["reported_mW"], a["corrected_mW"],
            a["memdyn_energy_drops"]))
