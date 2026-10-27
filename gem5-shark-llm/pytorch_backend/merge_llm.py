"""Merge llm_opatt.py results for one model into a table, with attack success
rates computed against the clean run of the same model.

ASR (per zero-shot task) = fraction of items the clean model answers correctly
(acc_norm) that the attacked model gets wrong; mean ASR averages the three
tasks. WikiText-2: next-token ASR over positions the clean model predicts
correctly, and the flip rate of the top-1 prediction.
Usage: python merge_llm.py RESULTS_DIR/<model>   ->  summary.json + printed table
"""
import json
import os
import sys

import numpy as np

TASKS = ["arc_easy", "piqa", "hellaswag"]
ORDER = ["clean", "attn-weight-10", "attn-weight-90", "ffn-weight-10", "ffn-weight-90",
         "norm-identity", "silu-identity", "silu-relu", "drop-scale", "softmax-identity",
         "drop-causal", "rope-identity", "dsatt-w-1", "dsatt-w-10", "dsatt-kv-10", "dsatt-kv-100"]


def main(d):
    have = [s for s in ORDER if os.path.exists(os.path.join(d, s + ".json"))]
    assert "clean" in have, "clean run missing"
    ref = np.load(os.path.join(d, "items_clean.npz"))
    rows = []
    for s in have:
        r = json.load(open(os.path.join(d, s + ".json")))
        z = np.load(os.path.join(d, "items_%s.npz" % s))
        row = {"spec": s, "ppl": r["wikitext2"]["ppl"], "top1": r["wikitext2"]["top1_acc"],
               "top_pred_share": r["wikitext2"]["top_pred_share"],
               "top_pred_token": r["wikitext2"]["top_pred_token"],
               "generation": r["generation"]}
        ok = ref["wt_pred"] == ref["wt_target"]
        row["wt_flip"] = float((z["wt_pred"] != ref["wt_pred"]).mean())
        row["wt_asr"] = float((z["wt_pred"][ok] != ref["wt_target"][ok]).mean())
        asr = []
        for t in TASKS:
            row[t] = r["zero_shot"][t]["acc_norm"]
            c, a = ref[t + "_acc_norm"].astype(bool), z[t + "_acc_norm"].astype(bool)
            row[t + "_asr"] = float((~a[c]).mean())
            asr.append(row[t + "_asr"])
        row["mean_acc"] = float(np.mean([row[t] for t in TASKS]))
        row["mean_asr"] = float(np.mean(asr))
        rows.append(row)
    json.dump(rows, open(os.path.join(d, "summary.json"), "w"), indent=1)
    print("%-17s %10s %6s %6s %6s %6s %6s %6s %6s" % (
        "spec", "ppl", "top1", "wtASR", "ARC-E", "PIQA", "HellaS", "mAcc", "mASR"))
    for r in rows:
        print("%-17s %10.4g %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f" % (
            r["spec"], r["ppl"], r["top1"], r["wt_asr"], r["arc_easy"], r["piqa"],
            r["hellaswag"], r["mean_acc"], r["mean_asr"]))


if __name__ == "__main__":
    main(sys.argv[1])
