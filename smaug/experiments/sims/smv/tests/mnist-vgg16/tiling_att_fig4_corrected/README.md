# Figure 4: Tiling Attack and Defense

`python fig4_corrected.py [OUT_PREFIX]` regenerates Figure 4 of the paper
(`OUT_PREFIX.pdf`, `.png` and the ratios in `.json`) from the per-invocation
gem5-Aladdin summaries in `data/`:

| model | system | runs |
|---|---|---|
| LeNet-5 | one out-of-order x86 core, crossbar | `lenet5_{base,tatt,tatt_def}` |
| VGG-16 | one out-of-order x86 core, crossbar | `vgg16_{base,tatt,tatt_def}` |
| LLM decoder layer (Qwen2.5-1.5B dimensions, 512 cached tokens) | four in-order x86 cores, 2x2 mesh | `llm_layer_{base,tatt,tatt_def}` |

`tatt` runs with `--enable-tiling-attack` (and `--tiling-attack-gemm` for the LLM
layer); `tatt_def` adds `--enable-tiling-defense`. Average power and energy charge
each accelerator invocation its own scratchpad energy; see
`gem5-shark-llm/README.md` ("Power accounting").
