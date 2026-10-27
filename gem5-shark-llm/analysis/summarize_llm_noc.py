"""Summarise the DoS / FLAtt gem5 runs of the LLM decode step (gem5_llm_noc.sh)
with the paper's NoC metrics: end-to-end time (sim_ticks), average flit
latency (AFL), average link utilisation (ALU) and accelerator cycles, each also
normalised to mesh_base. Garnet resets its statistics at every stats dump, so
AFL and ALU are averaged over the dumps weighted by the flits / cycles of each.

Usage: python summarize_llm_noc.py llm_dec/gem5_L512   -> noc_summary.json
"""
import json
import os
import sys

CONDS = ["mesh_base", "mesh_dos", "mesh_flatt_dir", "mesh_flatt_dma", "mesh_flatt_cpu"]


def dumps(path):
    """One dict per stats dump with the fields used below."""
    out, cur = [], None
    for line in open(path):
        if line.startswith("---------- Begin Simulation Statistics"):
            cur = {}
        elif line.startswith("---------- End Simulation Statistics"):
            out.append(cur)
            cur = None
        elif cur is not None:
            f = line.split()
            if len(f) < 2:
                continue
            k = f[0]
            if k in ("sim_ticks", "system.ruby.network.average_flit_latency",
                     "system.ruby.network.avg_link_utilization",
                     "system.ruby.network.average_packet_latency",
                     "system.acc0_datapath.sim_cycles") or k.startswith(
                         "system.ruby.network.flits_received::total"):
                try:
                    cur[k] = float(f[1])
                except ValueError:
                    pass
    return out


def summary(path):
    ds = dumps(path)
    t = sum(d.get("sim_ticks", 0) for d in ds)
    fl = sum(d.get("system.ruby.network.flits_received::total", 0) for d in ds)
    afl = (sum(d.get("system.ruby.network.average_flit_latency", 0) *
               d.get("system.ruby.network.flits_received::total", 0) for d in ds) / fl) if fl else None
    alu = sum(d.get("system.ruby.network.avg_link_utilization", 0) * d.get("sim_ticks", 0)
              for d in ds) / t if t else None
    return {"dumps": len(ds), "sim_ticks": t, "flits_received": fl, "avg_flit_latency": afl,
            "avg_link_utilization": alu,
            "accel_cycles": sum(d.get("system.acc0_datapath.sim_cycles", 0) for d in ds)}


def main(root):
    res = {}
    for c in CONDS:
        p = os.path.join(root, c, "outputs", "stats.txt")
        if os.path.exists(p) and os.path.getsize(p):
            res[c] = summary(p)
    b = res.get("mesh_base")
    for c, r in res.items():
        if b:
            for k in ("sim_ticks", "avg_flit_latency", "avg_link_utilization", "accel_cycles",
                      "flits_received"):
                if r.get(k) is not None and b.get(k):
                    r[k + "_norm"] = r[k] / b[k]
        print("%-15s ticks=%.4g (%sx)  AFL=%s (%sx)  ALU=%s (%sx)  flits=%.4g" % (
            c, r["sim_ticks"], "%.3f" % r.get("sim_ticks_norm", 1),
            "%.2f" % r["avg_flit_latency"] if r["avg_flit_latency"] else "-",
            "%.2f" % r.get("avg_flit_latency_norm", 1),
            "%.4f" % r["avg_link_utilization"] if r["avg_link_utilization"] else "-",
            "%.2f" % r.get("avg_link_utilization_norm", 1), r["flits_received"]))
    json.dump(res, open(os.path.join(root, "noc_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
