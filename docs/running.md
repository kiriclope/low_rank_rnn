# Running Experiments

## LD_PRELOAD requirement

Any script importing both `torch` and `matplotlib` must be run with:

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python <script>.py
```

Training scripts (no matplotlib) don't need it.

---

## Running a sweep

### 1. Edit `make_configs` in `sweep.py`

```python
def make_configs(out_dir: str) -> list[RunConfig]:
    shared = dict(
        init_style="random", gain=2.0, nonlinearity="tanh",
        noise=1.0, model_noise=0.0,
        cue_on_go_input=True, go_on_rwd_input=False,
        freeze_input_stages=["gng", "dual"],
        freeze_gng_input_during_dpa=True,
        freeze_rank0_dual=True,
        nogo_target=0.0, cue_scale=2.0,
        stop_loss=0.1, dual_loss="separated",
        epochs_dpa=100, epochs_gng=100, epochs_dual=200,
        gng_nogo_weight=2.0, go_hinge_thresh=1.0,
        optimizer="adam", use_scheduler=False,
        kappa1_reg_weight=0.0,
        out_dir=out_dir,
    )
    for seed in range(5):
        configs.append(RunConfig(run_id=f"s{seed}_myrun", seed=seed, **shared))
    return configs
```

`run_id` must be unique — it names checkpoints. `out_dir` is set by `--out_dir` at
launch; don't hardcode it.

### 2. Launch

```bash
mkdir -p results/dual/sweep_myrun
screen -dmS sweep_myrun bash -c "python sweep.py \
    --out_dir results/dual/sweep_myrun \
    --n_gpus 2 --n_workers 10 --per_run_screen \
    2>&1 | tee results/dual/sweep_myrun/run.log"
```

- `--n_workers` is **total** across all GPUs (10 = 5/GPU for 2× A30).
- `--per_run_screen` spawns one screen per seed: `sweep_s{seed}_myrun`.
- Attach: `screen -r sweep_s0_myrun`
- List: `screen -ls`

### 3. Monitor

```bash
wc -l results/dual/sweep_myrun/results.jsonl   # seeds done
python analyze.py --results results/dual/sweep_myrun/results.jsonl
```

### 4. Run filter

To run only a subset of configs defined in `make_configs`:

```bash
python sweep.py --out_dir results/dual/sweep_myrun --run_filter myrun_tag
```

Only configs whose `run_id` contains `myrun_tag` are executed.

---

## Rerunning only the Dual stage

Load `naive_*.pth` (after GNG) or `expert_*.pth` (after Dual) and retrain Dual:

```bash
python rerun_dual.py \
    --sweep_dir results/dual/sweep_new \
    --source_dir results/dual/sweep_original \
    --ckpt_prefix naive \          # or "expert" to start from after-Dual
    --epochs_dual 200 \
    --no_scheduler \
    --n_gpus 2 --n_workers 10
```

Results written to `sweep_new/results.jsonl`; checkpoints to `sweep_new/expert_*.pth`.

---

## Single run (for testing)

```bash
screen -dmS sweep_test bash -c "python -c '
import torch
from sweep import RunConfig, run_single
cfg = RunConfig(
    run_id=\"test\", seed=0, init_style=\"random\",
    gain=2.0, epochs_dpa=50, epochs_gng=50, epochs_dual=100,
    out_dir=\"results/dual/test\"
)
import os; os.makedirs(cfg.out_dir, exist_ok=True)
result = run_single(cfg, \"cuda:0\")
print(result[\"accuracy\"])
' 2>&1 | tee results/dual/test/run.log"
```

---

## GPU worker tuning

For these small models (512 hidden, rank 2):
- **4–8 workers per GPU** is the sweet spot.
- Above ~10/GPU: CUDA scheduling overhead dominates.
- `--n_workers` is **total** across all GPUs, not per-GPU.

---

## Checkpoints

After each stage, checkpoints are saved as:
- `dpa_{run_id}.pth` — after DPA
- `naive_{run_id}.pth` — after GNG
- `expert_{run_id}.pth` — after Dual

Results are appended one-line JSON to `results.jsonl`. A sweep skips run IDs already
present in `results.jsonl`.

> **Note:** `gain` is NOT saved in `state_dict` — always read it from `results.jsonl`
> config when loading a checkpoint for analysis.

---

## EISTP model (NeuroFlame EI+STP port) — how-to

The `EISTPModel` (`model_type="eistp"`) is the model that gives **persistent working memory +
lower-plane decision wells**. See `docs/architecture.md` (model) and
`docs/ring_lowerplane_log.md` §11 (the science). It's driven by the *same* `src/tasks`
generators and the *same* sweep/plot pipeline as the vanilla model.

### Reference config (current best — clean 5/5)
In `make_configs` (`sweep.py`), the EISTP `shared` dict:
```python
model_type="eistp", nonlinearity="relu",
n_neuron=1000, eistp_K=125.0,        # K scaled with N to hold prob K/N=0.125
eistp_lr_scale="sqrtK",              # ★ REQUIRED — "/N" gives g_mem≈0.015 (dead, DPA chance)
low_rank_scale=1.0,                  # lr_ini (=1 → memory mode starts critical)
eistp_lr_ueqv=False,                 # random init (m,n independent) — works; True = m init n
eistp_r_max=500.0,                   # rate cap, ~6× the ~80 operating peak (anti-runaway)
stp_U=0.05, stp_tau_f=1.0, stp_tau_d=0.2,   # Markram STP
j_stp=1.0,
go_hinge_thresh=1.0, dpa_hinge_thresh=1.0,  # hinge targets (relu can't hit exact ±1)
learning_rate=0.05, grad_clip_norm=1.0,     # clipping ON — needed; without it ~1-2 seeds diverge in Dual
batch_size=32, n_batch=256, epochs_dpa/gng/dual=100,
```

### Launch (same workflow as vanilla)
```bash
mkdir -p results/dual/sweep_eistp_myrun
screen -dmS sweep_eistp bash -c "python sweep.py \
    --out_dir results/dual/sweep_eistp_myrun --n_gpus 2 --n_workers 2 --per_run_screen \
    --nonlinearity relu --cue_on_go_input 1 [--nogo_target 0.0] \
    2>&1 | tee results/dual/sweep_eistp_myrun/run.log"
```
- `--nogo_target` (new CLI) overrides the Dual/GNG nogo target. **nogo=−1** pushes the NoGo well
  firmly to the lower plane (κ₁≈−3.5); **nogo=0** parks it near the κ₁=0 midline. So the well
  depth is a controllable knob.
- **Memory:** each eistp run uses ~5.5 GB (BPTT through ~440 steps, N=1000). GPU0 fits ~3 with
  `--per_run_screen`. **Don't run two eistp sweeps concurrently** (6 on GPU0 → OOM); run sequentially.

### Plotting
```bash
# acc + traj  (use plot_sweep, but NOT --plots flow for eistp — its analytic flow is invalid)
LD_PRELOAD=… python plot_sweep.py --sweep_dir results/dual/sweep_eistp_myrun \
    --out_root results/figures --plots acc traj --no_individual --device cuda:0
# binned flow fields (eistp-aware: two-timescale + STP, auto-calibrated grid, magma default)
LD_PRELOAD=… python ei_flow.py --sweep_dir results/dual/sweep_eistp_myrun \
    --out_root results/figures --device cuda:1 [--style magma|binned]
```

### Stability / divergence
The `/√K` coupling + STP is near-critical and can run away (rates → ∞). Three guards (all on):
1. **Rate cap** `eistp_r_max` (clamps rates; only catches runaway, science untouched).
2. **NaN-skip** in `Optimization` — skips a non-finite batch/grad instead of corrupting weights.
3. **Graceful divergence** — if a whole epoch goes non-finite, `_run_epoch` returns `nan` →
   `fit()` stops that run and keeps the best pre-divergence state (records a result, no crash).
Plus `grad_clip_norm=1.0` (keep it on — it's the difference between 5/5 and ~4/5).
