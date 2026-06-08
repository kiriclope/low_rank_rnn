# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A study of rank-2 low-rank RNNs learning three cognitive tasks **sequentially**: DPA
(delayed paired association) → GNG (go/no-go) → Dual (combined). The core question is
how **structured vs. random initialisation** affects retention of the first task.

Precise framing of "forgetting" (the code's actual mechanism):
- **Stage 1 (DPA):** all parameters trained.
- **Stage 2 (GNG):** DPA is *not* trained; it is protected by freezing rank-0 of the
  recurrent factors and the DPA + reward input dims. **`after_gng/dpa` accuracy is the
  retention / forgetting metric** — it measures how much DPA survives GNG learning given
  that protection.
- **Stage 3 (Dual):** DPA *and* GNG are trained jointly (the dual task contains both), so
  `after_dual` numbers reflect multi-task interference / recovery, not pure forgetting.

## Directory layout

```
rnn/
├── src/                    # library package (imported as src.xxx)
│   ├── models.py
│   ├── tasks.py
│   ├── train.py
│   ├── init.py
│   └── dynamics.py
├── plots/                  # legacy single-run plot scripts
│   ├── plot.py, plot_fixed_points.py, plot_trajectories.py
│   ├── plot_ring.py, plot_dpa_by_trialtype.py
│   ├── plot_all_fixed_points.py, plot_fp_scatter.py, plot_trials.py
├── scripts/                # utility scripts (monitor_gpu, etc.)
├── sweep.py                # sweep runner + RunConfig
├── plot_sweep.py           # main plotting entrypoint
├── plot_mn_inputs.py       # m/n/Wi alignment plots
├── analyze.py              # load_results, summary_table
└── results/
    ├── dual/               # run dirs (results.jsonl, models/, run.log)
    └── figures/            # plot output from plot_sweep.py
```

All library modules are imported via `from src.xxx import ...`. Scripts in `plots/` add
the project root to `sys.path` at the top so `from src.xxx` works when run as
`python plots/script.py` from the project root.

## Running things

> **matplotlib + torch requires an `LD_PRELOAD` shim.** Importing `torch` first loads the
> system `libstdc++`, which lacks the `CXXABI_1.3.15` symbol matplotlib's C extension
> needs. Any script that imports both (all plotting scripts) must be run as:
> ```bash
> LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python <script>.py ...
> ```
> Training scripts that don't import matplotlib don't need it.

**Single training run** — always run inside a named screen session and tee to a log:
```bash
mkdir -p results/dual/myrun
screen -dmS sweep_myrun bash -c "python -c '
import torch
from sweep import RunConfig, run_single
cfg = RunConfig(run_id=\"test\", seed=0, init_style=\"structured\", gain=2.0, epochs_dpa=50, epochs_gng=50, epochs_dual=50, out_dir=\"results/dual/myrun\")
result = run_single(cfg, \"cuda:0\")
print(result[\"accuracy\"])
' 2>&1 | tee results/dual/myrun/run.log"
```

> **Always use `screen -dmS sweep_<name> bash -c "... 2>&1 | tee <out_dir>/run.log"`**
> for every run so Leon can attach (`screen -r sweep_<name>`) and monitor at any time.
> The `-dm` flags start detached (required when launching from a non-interactive shell).
> The pipe and tee must be **inside** `bash -c "..."` — wrapping screen with a pipe does
> not capture the Python process output. Screen session name convention: `sweep_<run_id>`
> for single runs, `sweep_<sweep_name>` for full sweeps.

**Sweep** (edit `make_configs` in `sweep.py` first):
```bash
mkdir -p results/dual/myrun
screen -dmS sweep_myrun bash -c "python sweep.py --out_dir results/dual/myrun --n_gpus 2 2>&1 | tee results/dual/myrun/run.log"
```
Multi-GPU via `multiprocessing` with `spawn` start method (required for CUDA). Use
`--n_workers` to run more than one job per GPU. **`--n_workers` is total across all GPUs**
(e.g. `--n_gpus 2 --n_workers 16` = 8 per GPU). For these small models (512 hidden,
rank 2), 4–8 workers per GPU is the sweet spot — above ~10/GPU, CUDA scheduling overhead
dominates and wall-clock time increases despite 99% utilisation.

**Analyse results table**:
```bash
python analyze.py --results results/dual/myrun/results.jsonl
```

**All figures for a sweep** (preferred plotting entrypoint):
```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python plot_sweep.py --sweep_dir results/dual/myrun
```
Or just ask the **`plotting` subagent** in chat (`.claude/agents/plotting.md`) — it
dispatches `plot_sweep.py` with the right flags. See "Analysis and plotting" below.

Completed runs are appended one-line JSON to `results.jsonl`; checkpoints go to
`models/dpa_<id>.pth` (after DPA), `naive_<id>.pth` (after GNG), `expert_<id>.pth`
(after Dual). Sweep skips run IDs already present in `results.jsonl`.

**Single runs via `python -c` do not write `results.jsonl` automatically** — reconstruct
it manually from the run log if needed for plotting.

## Architecture

### Model (`src/models.py`)

`LowRankModel`: discrete-time, two-timescale RNN with a rank-R recurrent factorisation
`W_rec = m @ n.T / N`. There is **no separate output layer** (`output_size=0`) — the
network reads out internally via `κ = rates @ n / N` (shape `(B, T, rank)`). For rank 2
the convention is **κ₀ = memory, κ₁ = decision/readout** (κ₁ is the loss/accuracy channel).

Per-step update (`update_dynamics`), with `h` = recurrent-input variable:
```
h     ← exp(-alpha_rec)·h    + (1-exp(-alpha_rec))·(W_rec · rates)
rates ← exp(-alpha)·rates    + (1-exp(-alpha))·tanh(gain·(Ai·Wi·x + h))
```
**`gain` applies to the full net input** (feedforward + recurrent), so it controls the
effective nonlinearity slope for both drives equally. The chaos threshold is
`gain × λ_max(W_rec) = 1` for the recurrent part; feedforward inputs are also amplified.
`h` and `rates` evolve on timescales `alpha_rec` and `alpha` respectively.

**Reward feedback is teacher-forced.** `rwd_channel = -1` is overloaded (by design) across
three spaces: it indexes the reward *input* channel (`input_size-1`), the decision *target*
channel, and the decision *readout* (κ₁). At each step, if `target[...,-1] == 1` and
`readout[...,-1] > 0.5`, a +1 pulse is added to the reward input channel on the **next**
step. So reward fires only on correct positive-target responses, and it is active during
accuracy evaluation too (eval runs `model(X, y)` with true targets).

Key parameters: `rank`, `gain` (scales full net input), `alpha = dt/tau`,
`alpha_rec = dt/tau_rec`, `noise` (per-step recurrent noise std, set/reset around each
stage). **`gain` is NOT saved in `state_dict`** — always read it from the `results.jsonl`
config when loading a checkpoint for analysis.

### Tasks (`src/tasks.py`)

All generators return `(inputs, targets[, trial_type, condition_names])` of shape
`(n_trials, n_steps, *)`. Timing is driven by `TaskTiming(stim_on, stim_off, t_steps, dt)`;
the dual task uses 4 epochs: `[sample, gng, cue, test]`.

Input channel layout (standard `input_size=8`):
- 0=A sample, 1=B sample, 2=C test, 3=D test ← DPA channels
- 4=Go stimulus, 5=NoGo stimulus, 6=GNG cue (merged into ch4 if `cue_on_go_input=True`)
- 7=reward (always last channel = `input_size - 1`)

With `cue_on_go_input=True`, `input_size` becomes 7 (decremented in
`RunConfig.__post_init__`, so the value stored in `results.jsonl` is already 7 — do **not**
decrement it again when loading).

`cue_scale` (default 1.0) scales the amplitude of the GNG cue signal in both
`generate_gng_trials` and `generate_dual_trials`.

`nogo_target` (default 0.0, alternative −1.0) sets the decision target for nogo trials.
The GNG accuracy threshold is auto-computed as `(1 + nogo_target) / 2` (0.5 for target=0,
0.0 for target=−1).

Target encoding (`target_rank=2`, used everywhere here):
- **Channel 0 (memory):** in DPA, supervised to ±1 (A=+1, B=−1) over the delay. In the
  **dual** task it is left `nan` (unsupervised) except a pre-sample 0 — so the memory rank
  is shaped during DPA and only indirectly constrained later.
- **Channel −1 (decision):** time-multiplexed. DPA decision (pair=+1 / unpair=−1) after the
  test; GNG response (go=+1, nogo=`nogo_target` then 0) in the cue window. `torch.nan`
  masks all timesteps excluded from the loss.

`MaskedMultiTargetLoss` computes a per-channel masked MSE and skips `nan` entries.

### Training pipeline (`sweep.py`, `src/train.py`)

Three sequential stages, each with selective freezing:
1. **DPA** — train all parameters.
2. **GNG** — freeze rank-0 of m/n (`freeze_low_rank_cols=[0]`) and DPA+reward input dims
   (`[0,1,2,3,input_size-1]`); go/nogo/cue input dims train. If
   `freeze_input_stages` includes `"gng"`, **all** input dims are frozen for this stage
   (useful to prevent go-input learning from rotating n₀).
3. **Dual** — by default, freeze **all** input dims (`list(range(input_size))`); both
   recurrent ranks train freely. If `freeze_rank0_dual=True`, rank-0 of m/n is also frozen
   through this stage (prevents n₀ rotating to align with go input during dual training).

`Optimization.fit()` implements the loop. **Freezing** = snapshot the frozen
columns/dims at construction, zero their grads after `backward()`, and **restore the
original values after `optimizer.step()`**. The restore is what makes freezing exact even
though AdamW's decoupled weight decay would otherwise nudge zero-grad params.

**Early stopping:** `stop_loss` (default 0.005, set to `None` to disable) halts a stage
when both train and val loss fall below this threshold.

**Gradient clipping:** `grad_clip_norm` (default `None` = disabled). Set to a float to
enable `torch.nn.utils.clip_grad_norm_`.

**Loss functions (`src/train.py`):**
- `MaskedMultiTargetLoss` — per-channel masked MSE. Used for DPA and GNG stages.
- `MaskedMultiTargetDualLoss` — multi-channel masked MSE whose decision channel is split by
  time window into separate **DPA / GNG / baseline** components (plus an `aux` term for the
  memory channel), each independently weightable; unweighted components are exposed in
  `.last_components` for logging. **Default for the Dual stage** (`loss="separated"` in
  `RunConfig`).
- `WeightedDualTaskLoss` — single-channel DPA/GNG split by time window. Defined but not
  currently wired into `sweep.py`.

### Accuracy metrics (`sweep.py`)

Computed on freshly generated trials with input noise but recurrent noise off
(`model.noise = 0`); reward feedback is on (teacher-forced):
- `_dpa_accuracy`: decision readout (κ₁) averaged after the test, thresholded at **0**
  (target ±1).
- `_gng_accuracy`: κ₁ averaged after the response window, thresholded at
  `(1 + nogo_target) / 2`. Both `cue_scale` and `nogo_target` are passed from config.
- `_dual_accuracy`: DPA read post-test (threshold 0) and GNG read in the cue→test gap
  (threshold `(1+nogo_target)/2`), returned as `dual_dpa`, `dual_gng`. Trial type is
  parsed from `condition_names` substrings (`"_go_"`, `"_nogo_"`).

`accuracy` dict keys: `after_dpa`, `after_gng`, `after_dual` (the last also carries
`dual_dpa`, `dual_gng`).

### `RunConfig` (single source of truth)

Key fields and couplings:
- `noise` — **prefactor**; actual per-step sigma = `noise × sqrt(1 − exp(−alpha)²)`.
  `model_noise` is the recurrent-noise prefactor (same formula).
- `cue_on_go_input` — routes GNG cue onto the go input channel; triggers `input_size -= 1`
  in `__post_init__` (stored value already reflects this).
- `target_rank` — output dim of the trial generators; **2 everywhere here**, including
  accuracy eval. (Eval reads only channel −1, so the value is immaterial to the result.)
- `init_style` — `"structured"` (calls the init below) or `"random"` (default
  `LowRankModel` init).
- `freeze_input_stages` — list of stage names (`"gng"`, `"dual"`) in which **all** input
  dims are frozen. Default `["dual"]`. Use `["gng", "dual"]` to freeze inputs during GNG
  too.
- `freeze_rank0_dual` — bool (default `False`). If `True`, rank-0 of m/n is frozen during
  the Dual stage in addition to the Dual-stage input freeze.
- `stop_loss` — float or `None` (default 0.005). Early-stop a stage when loss drops below.
- `cue_scale` — float (default 1.0). Amplitude multiplier for the GNG cue signal.
- `nogo_target` — float (default 0.0, alternative −1.0). Decision target for nogo trials.
- `grad_clip_norm` — float or `None` (default `None`). Max gradient norm; `None` disables.
- `memory_lambda`, `decision_lambda`, `target_mn_corr`, `mix_strength`, … — structured-init
  knobs (see below).

### Initialisation (`src/init.py`)

`init_dpa_internal_readout_prepost`: structured init for the DPA stage. Builds orthogonal
population vectors so that rank-0 is a sample-memory attractor (eigenvalue
`memory_lambda`, `corr(m,n)=target_mn_corr`) and rank-1 is a decision rank (eigenvalue
`decision_lambda`). A/B input weights align with the memory vector (±`u_mem`), C/D with the
test vector (±`u_test`).

**Caveat (`mix_strength`):** the decision readout direction is
`u_read = mix_strength·u_mix + sqrt(1-mix_strength²)·u_noise`, where `u_mix = u_mem ⊙ u_test`
is the actual pair-discriminating direction. The sweep uses **`mix_strength=0`**, so the
decision rank is initialised with the right eigenvalue but a *random direction* — the
network still has to learn to read out the pair signal. "Structured" therefore mainly
scaffolds the **memory**, not the decision.

### Analysis and plotting

`analyze.py`: `load_results(path)` → flat `DataFrame` (config + accuracy columns, filtered
to `status == "ok"`). `summary_table(df)` groups by init style and lambda.

**`plot_sweep.py` is the main plotting entrypoint.** Output goes to
`results/figures/<sweep_name>/{summary,individual}/`:
- `summary/`: `accuracy_stages`, `accuracy_by_trialtype` (DPA/GNG accuracy per stage, one
  line per trial type), `fp_scatter_by_stage` (autonomous FPs across stages),
  `fp_scatter_by_input_<cond>` (one by-stage figure per input condition), `traj_*` (mean κ
  trajectories, pair/unpair × κ₁/κ₂, for DPA/Go/NoGo).
- `individual/<run_id>/`: same accuracy + trajectory figures, plus `scatter/fp_scatter.pdf`
  and `flow/fp_<stage>.pdf`.

Flags: `--no_summary`, `--no_individual`, `--run_ids`, `--skip_flow`, `--skip_scatter`,
`--n_fp_seeds` (21 default, 41 for publication), `--device`. The **`plotting` subagent**
(`.claude/agents/plotting.md`) is a thin dispatcher over this script.

Older single-purpose scripts in `plots/` (`plot.py`, `plot_fixed_points.py`,
`plot_trajectories.py`, `plot_ring.py`, `plot_dpa_by_trialtype.py`,
`plot_all_fixed_points.py`, `plot_fp_scatter.py`) take a `results.jsonl` / `--ckpt_dir`
and load checkpoints from the adjacent `models/`. `plot_fixed_points.py` and
`plot_trajectories.py` now read `gain` / `input_size` / `cue_on_go_input` from the run's
config automatically.

**Dynamics (`src/dynamics.py`):** rank-2 κ-plane vector fields, fixed-point finding
(`find_all_fixed_points` solves `κ = Nᵀtanh(gain·(input+Mκ))/N` via scipy — note gain
applies to the full argument), stability classification, and `plot_task_flow_fields` (phase
portraits). A torch-native two-timescale flow (`rank2_kappa_flow`) also exists; for the
standard adiabatic (q=κ) portrait it gives the same fixed points and streamline directions
as the numpy field (speed differs by a constant `1-exp(-alpha)`), so the numpy path is the
default.

## Open methods notes (intentional but worth keeping in mind)

- **Dual stage memory:** by default, rank-0 is trainable *and* unsupervised during Dual
  (GNG freezes it, Dual does not). Set `freeze_rank0_dual=True` to lock it through Dual;
  this prevents n₀ from rotating to align with the go input (go bimodality on κ₀).
- **Loss weighting:** Dual uses `MaskedMultiTargetDualLoss` (loss="separated") with equal
  weights. Individual DPA/GNG/aux components are logged via `.last_components`.
- **Checkpoints are last-epoch** (`keep_best=False`), not best-val.
