"""Reverse engineering (RE) of an LLM decode step from finish-flag timing.

The observer sees, for one decode step, the number of accelerator invocations
(finish flags) and their durations. Operator names, taken from Smaug's own
schedule (native debug log), are used only to label the results.

Reported per KV-cache length L (gem5 base runs):
  * invocations per decode step and decode-step time (gem5 ticks between the
    switch to detailed simulation and the end of the step);
  * the attention stage (QK^T tiles, softmax, AV tiles) against the rest of the
    layer, which does not depend on L;
  * per-operator tile counts and median tile durations, i.e. the signature an
    observer sees as runs of equal-cost invocations.
Across L, invocation count and step time are fitted linearly in L: the slope
is what one more cached token costs, i.e. how finely the context length leaks.

Usage: python re_llm.py OUT.json L:run_dir:native_log [...]
"""
import json
import re
import sys

import numpy as np


def invocations(path):
    seq, fn = [], None
    for line in open(path):
        m = re.match(r"\s*Top level function:\s*(\S+)", line)
        if m:
            fn = m.group(1)
        m = re.match(r"\s*Cycle\s*:\s*([\d.]+)", line)
        if m and fn:
            seq.append((fn, int(float(m.group(1)))))
            fn = None
    return seq


def schedule(native_log):
    """(op name, op type, invocations) in execution order, from a native
    --debug-level=1 run (one 'Input...' line per accelerator invocation)."""
    ops, cur = [], None
    for line in open(native_log):
        m = re.search(r"Scheduling Operator: (\S+) \((\w+)\)", line)
        if m:
            cur = [m.group(1), m.group(2), 0]
            ops.append(cur)
        elif line.startswith("Input") and cur is not None:
            cur[2] += 1
    return [tuple(o) for o in ops if o[2] > 0]


def step_ticks(stdout):
    txt = open(stdout).read()
    s = int(re.search(r"Switched CPUS @ tick (\d+)", txt).group(1))
    e = int(re.findall(r"Exiting @ tick (\d+)", txt)[-1])
    return e - s


def main(out, specs):
    res = {}
    for spec in specs:
        L, run_dir, native = spec.split(":")
        seq = invocations(run_dir + "/outputs/nnet_fwd_summary")
        sched = schedule(native)
        lab = [n for n, t, k in sched for _ in range(k)]
        assert len(lab) == len(seq), (L, len(lab), len(seq))
        c = np.array([x[1] for x in seq], dtype=float)
        ops = []
        for n, t, k in sched:
            v = c[[i for i, m in enumerate(lab) if m == n]]
            ops.append({"op": n, "type": t, "invocations": k, "cycles": float(v.sum()),
                        "median_cycles": float(np.median(v)),
                        "mad_rel": float(np.median(np.abs(v - np.median(v))) / np.median(v))})
        att = np.array([bool(re.search(r"qkT|softmax|_av\d", n)) for n in lab])
        res[L] = {"invocations": len(c), "accel_cycles": float(c.sum()),
                  "step_ms": step_ticks(run_dir + "/gem5_stdout.txt") / 1e9,
                  "attention_invocations": int(att.sum()),
                  "attention_cycles": float(c[att].sum()),
                  "other_invocations": int((~att).sum()), "other_cycles": float(c[~att].sum()),
                  "ops": ops}
        r = res[L]
        print("L=%-5s step %.3f ms, %d invocations, accel %.3fM cycles; attention %d inv %.3fM "
              "cycles; rest %d inv %.3fM cycles" % (
                  L, r["step_ms"], r["invocations"], r["accel_cycles"] / 1e6,
                  r["attention_invocations"], r["attention_cycles"] / 1e6,
                  r["other_invocations"], r["other_cycles"] / 1e6))
        for o in ops:
            if re.search(r"qkT|softmax|_av\d", o["op"]):
                print("   %-14s %4d x %8.0f cycles (MAD %.2f%%)" % (
                    o["op"], o["invocations"], o["median_cycles"], 100 * o["mad_rel"]))
    Ls = sorted([k for k in res], key=int)
    if len(Ls) >= 2:
        x = np.array([int(k) for k in Ls], dtype=float)
        fits = {}
        for key in ("invocations", "step_ms", "attention_cycles", "other_cycles"):
            y = np.array([res[k][key] for k in Ls], dtype=float)
            a, b = np.polyfit(x, y, 1)
            fits[key] = {"slope_per_token": float(a), "intercept": float(b),
                         "r": float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else None}
            print("%-17s = %.6g * L + %.6g   (r = %s)" % (
                key, a, b, "%.6f" % fits[key]["r"] if fits[key]["r"] is not None else "-"))
        res["fits"] = fits
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
