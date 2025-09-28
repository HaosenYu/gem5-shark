# gem5-shark

**gem5-shark** extends the standard **SMAUG on gem5-Aladdin** workflow with modified **C models** and updated **Python wrappers** to enable **threat/attack simulation** at multiple layers of the stack.

> You do **not** run gem5-shark as a standalone Docker image. Instead, **merge this repository into the official SMAUG Docker workspace** so your changes run seamlessly with the usual SMAUG flow. Usage remains “SMAUG-like,” with **additional options to enable attacks**.

---

## Repository layout

```
.
├─ gem5-aladdin/              # gem5 + Aladdin (SoC + accelerator simulation)
├─ LLVM-Tracer/               # LLVM-based tracer used by SMAUG
├─ smaug/                     # SMAUG (framework + experiments + sims)
│  └─ experiments/
│     └─ sims/
│        └─ smv/
│           └─ tests/         # Test models and datasets (per-model subfolders)
├─ LICENSE
└─ README.md
```

### Tests directory

`smaug/experiments/sims/smv/tests/` contains **per-model** test packs (model graph, weights, sample inputs, and minimal configs). Use these as templates when adding new threat experiments.

---

## How to integrate with the SMAUG Docker

1. Prepare the **official SMAUG Docker** environment as usual.
2. Inside the container workspace (commonly `/workspace`), **overlay/merge** this repo so the modified C models and Python wrappers **replace or augment** the stock versions in `smaug/`, `gem5-aladdin/`, and `LLVM-Tracer/`.
3. Build and run following the normal SMAUG process. The only change is that **new attack options** are now available in the Python wrappers and in config glue.

> We intentionally do not document bespoke run commands. If you know how to run SMAUG, you already know how to run gem5-shark. You will **append the new attack options** described below.

---

## What changed

* **C model updates** inside `smaug/` and accelerator backends to expose hooks for fault or attack injection, data perturbation, timing effects, and instrumentation needed for threat studies.
* **Python wrapper updates** to parse **new CLI options** that toggle threat scenarios and pass them through either to SMAUG runtime or to gem5-Aladdin configuration.

---

## Threat-simulation options

Your new options are grouped by **where they take effect**. The **exact flags** are exposed by each wrapper’s `--help`.

### A) SMAUG-level options (parsed in Python wrappers)

**Purpose:** Threats at the model/runtime layer.

Typical knobs:

* Attack enable toggles
* Target selection (weights, activations, inputs, layers)
* Intensity or rate parameters
* Timing or trigger controls (which layer or iteration)

**Where they live:**

* Python entry points under `smaug/experiments/sims/` (for SMV, see `smaug/experiments/sims/smv/…`)
* Any shared argument modules used by experiments

**How to discover them locally:**

* Run the relevant wrapper with `--help`.
* Inspect per-test `run.sh` in `smaug/experiments/sims/smv/tests/<model>/` to see which wrapper is invoked, then run that wrapper with `--help`.

#### Examples: SMAUG-level options added in gem5-shark

```

--enable-swap-att[=1]
    Enable pixel-index swapping on inputs (default 1=on).

--enable-operator-attack[=spec]
    Attack selected ops (e.g., Conv/GEMM/ReLU); `spec` chooses ops/action.

--enable-tiling-attack[=level]
    Perturb tiling/dataflow to hurt locality; `level` sets severity.
```

> These flags are parsed by the **SMAUG-side** wrappers and, when needed, propagated down to gem5-Aladdin configs.

---

### B) gem5-Aladdin-level options (propagated into configs)

**Purpose:** Threats at the **accelerator/SoC** level (datapath, NoC, routers, NI, memory system, etc.).

**Where they live:**

* `gem5.cfg` used by each test
* Backend accelerator config such as `smv-accel.cfg`
* Wrapper glue that writes or modifies these configs based on the new flags

**How to discover them locally:**

* Check the wrapper’s `--help` for flags that mention **Aladdin**, **gem5**, or **accel config**.
* Open the `gem5.cfg` and backend config loaded by your test to see how flags map to config fields.

#### Examples: gem5-level NoC/NI threat options added in gem5-shark

```
--ni-dos-att
    Enable Network Interface DoS attack.

--ni-lost-att=NI_LOST_ATT
    Enable Network Interface Flit Lost attack.

--lost-ni-att=LOST_NI_ATT
    Specify Network Interface ID for Flit Lost attacking.

--lost-router-att=LOST_ROUTER_ATT
    Specify Router ID for Flit Lost attacking.

--dos-num-att=DOS_NUM_ATT
    Specify the number of injected repeated flits into the NoC.
```

#### Example: gem5.cfg switch for delay injection

```
flag_delay_att = 1
```

Enables injected timing delay (NoC/NI/router) to model jitter/congestion. Set `0` to disable.

---

## Verify and catalog your options

After merging gem5-shark:

* Search for `argparse` in `smaug/experiments/sims/…` to locate **new flags**.
* Run each wrapper with `--help` and record **SMAUG-level** options.
* Open `gem5.cfg` and `smv-accel.cfg` to record **gem5-Aladdin-level** options affected by wrappers.
* Keep a short internal table mapping **flag → effect → layer (SMAUG or gem5-Aladdin)**.

> Tip: keep per-model defaults small and reproducible in `smaug/experiments/sims/smv/tests/` so runs stay quick and comparable.

---

## Adding your own threat experiment

1. Copy an existing subfolder from `smaug/experiments/sims/smv/tests/` and replace the graph, weights, and inputs.
2. Reuse the standard SMAUG run flow and **append threat options** as needed.
3. If you add new knobs:

   * Implement or expose them in the **Python wrapper** (SMAUG-level).
   * Thread them into `gem5.cfg` or the backend config if they affect **gem5-Aladdin**.

---

## Compatibility and docs

* Base behavior follows **SMAUG**. gem5-shark only **adds** threat options and instrumentation.
* For the standard build and run flow, refer to the official SMAUG documentation and README.

---

## License

AGPL-3.0 (see `LICENSE`).

---

## Acknowledgments

Built on **SMAUG** and **gem5-Aladdin**.
