"""Download the models and evaluation datasets into HF_HOME, so
the GPU jobs can run fully offline."""
import sys
import time

from datasets import load_dataset
from huggingface_hub import snapshot_download

MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B"]
DATASETS = [("allenai/ai2_arc", "ARC-Easy"), ("baber/piqa", None),
            ("Rowan/hellaswag", None), ("Salesforce/wikitext", "wikitext-2-raw-v1")]

for name, cfg in DATASETS:
    t = time.time()
    ds = load_dataset(name, cfg)
    print("dataset %-22s %-18s %s (%.0fs)" % (name, cfg, {k: len(v) for k, v in ds.items()},
                                            time.time() - t), flush=True)
for m in MODELS:
    t = time.time()
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                             "*.tiktoken", "merges.txt", "vocab.json"])
    print("model %s -> %s (%.0fs)" % (m, p, time.time() - t), flush=True)
print("PREFETCH_DONE")
