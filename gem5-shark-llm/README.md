# gem5-shark-llm: Transformer and LLM workloads

Extensions used for the large-model evaluation of gem5-shark: one LLM decoder
layer simulated at cycle level in gem5-Aladdin, and a PyTorch backend that
applies the accuracy-facing threat vectors (DPAtt, OPAtt, DSAtt) to complete
pretrained models (Qwen2.5-1.5B/7B, Llama-3.2-1B, Llama-3.1-8B, ViT-B/16).

## Layout

```
gem5-shark-llm/
├─ smaug_models/            Transformer workloads built in SMAUG
│  ├─ llm_decode_smaug.py   one decoder layer of an autoregressive decode step
│  │                        (default: Qwen2.5-1.5B dimensions, grouped-query attention,
│  │                        SwiGLU, KV cache of --L tokens); writes dec_topo.pbtxt,
│  │                        dec_params.pb and a NumPy reference output
│  ├─ check_output.py       compares SMAUG's output with that reference
│  ├─ transformer_block.py  single Transformer block (tests of the GEMM tiler)
│  ├─ train_vit.py, vit_smaug.py   small ViT on MNIST (training, SMAUG graph, OPAtt)
│  ├─ train_lm.py, lm_smaug.py     small character-level causal decoder
│  └─ dump_smaug_logits.py  SMAUG reference logits of both small models per OPAtt spec
├─ gem5_scripts/
│  ├─ gem5_llm.sh           base, TAtt, TAtt + D-TAtt, FFSAtt, FFSAtt + watchdog
│  │                        (one out-of-order core, crossbar)
│  ├─ gem5_llm_noc.sh       DoS, FLAtt (directory, DMA, CPU L1), TAtt, TAtt + D-TAtt
│  │                        (four in-order cores, 2x2 mesh)
│  └─ native_llm_scaling.sh native SMV runs at the dimensions of six LLMs
├─ pytorch_backend/
│  ├─ opatt_mirror.py       PyTorch implementation of the SMAUG OPAtt passes
│  ├─ mirror_validate.py    checks it against SMAUG on the two small Transformers
│  ├─ llm_opatt.py          OPAtt and DSAtt on Hugging Face decoder-only LLMs
│  │                        (WikiText-2 perplexity; ARC-Easy, PIQA, HellaSwag via lm-eval)
│  ├─ llm_dpatt.py          DPAtt: LoRA fine-tuning on SST-2 with a trigger-word backdoor
│  ├─ vit_large_opatt.py    OPAtt and DSAtt on ViT-B/16 over ImageNet-1k validation
│  ├─ merge_llm.py          merges per-spec result files
│  └─ prefetch.py           downloads the models and datasets into HF_HOME
└─ analysis/
   ├─ power_recheck.py      accelerator power and energy, each invocation charged its
   │                        own scratchpad energy (see "Power accounting" below)
   ├─ summarize_llm_noc.py  runtime, average flit latency and link utilization of the NoC runs
   └─ re_llm.py             context-length leakage from finish-flag timing
```

## Cycle-level runs (inside the SMAUG container)

Rebuild `smaug` and `gem5-aladdin` from this repository first (the changes are
listed in the top-level README). Then, for one decoder layer with 512 cached tokens:

```
python smaug_models/llm_decode_smaug.py --L 512 --out llm_dec/L512
bash gem5_scripts/gem5_llm.sh base     llm_dec/L512 runs/base       # also tatt, tatt_def, ffs, ffs_def
bash gem5_scripts/gem5_llm_noc.sh mesh_base llm_dec/L512 runs/mesh_base
                                       # also mesh_dos, mesh_flatt_dir, mesh_flatt_dma,
                                       # mesh_flatt_cpu, mesh_tatt, mesh_tatt_def
python analysis/power_recheck.py runs/base runs/tatt
python analysis/summarize_llm_noc.py runs
bash gem5_scripts/native_llm_scaling.sh llm_scale
```

`re_llm.py` takes runs at several KV-cache lengths (`--L 128`, `512`, `2048`);
its header documents the arguments. One layer of one decode step takes about
40 minutes of host time at SMAUG's highest sampling level.

## PyTorch backend

Tested with Python 3.12, torch 2.10, transformers 4.57.6, peft 0.17.1,
accelerate 1.15.0, datasets 5.0.1, lm-eval 0.4.13 and timm 1.0.24 on one GPU.
The Llama models require accepting Meta's license on Hugging Face.

```
python pytorch_backend/prefetch.py
python pytorch_backend/llm_opatt.py --model Qwen/Qwen2.5-1.5B --out results
python pytorch_backend/llm_dpatt.py --model meta-llama/Llama-3.2-1B --out results
IMAGENET_VAL=/path/to/imagenet/val python pytorch_backend/vit_large_opatt.py
```

The backend is validated against SMAUG on the two small Transformers:

```
# SMAUG container; MNIST in Keras format as smaug_models/data/mnist.npz
python smaug_models/train_vit.py
python smaug_models/train_lm.py        # corpus: the container's Python standard library
python smaug_models/dump_smaug_logits.py --n-vit 1000 --n-lm 500
# PyTorch environment
python pytorch_backend/mirror_validate.py   # prints MIRROR_VALIDATION_PASS
```

## Power accounting

gem5-Aladdin's per-invocation summary (`outputs/nnet_fwd_summary`) reports a
memory dynamic power whose scratchpad part is the scratchpad energy accumulated
since the start of the run divided by the cycles of the current invocation,
because the scratchpad statistics are not reset between invocations. Averaging
the reported "Avg Power" over invocations therefore grows with the number of
invocations. `analysis/power_recheck.py` charges each invocation the increment
of the accumulated scratchpad energy; Figure 4 of the paper uses the same
accounting (`smaug/experiments/sims/smv/tests/mnist-vgg16/tiling_att_fig4_corrected/`).
