# How to run sweeps and plots

## Running a sweep

### 1. Edit `make_configs` in `sweep.py`

All sweep parameters live in `make_configs`. The pattern is:

```python
base = dict(
    init_style="random", gain=1.0, noise=1.0, model_noise=0.0,
    cue_on_go_input=True, go_on_rwd_input=False,
    freeze_input_stages=["gng", "dual"],
    freeze_gng_input_during_dpa=True,
    freeze_rank0_dual=True,
    nogo_target=0.0, cue_scale=2.0,
    stop_loss=0.1, dual_loss="separated",
    epochs_dpa=100, epochs_gng=100, epochs_dual=100,
    optimizer="adam", use_scheduler=False,
    gng_nogo_weight=2.0, go_hinge_thresh=1.0,
    out_dir=out_dir,
)
for seed in range(10):
    configs.append(RunConfig(run_id=f"s{seed}_myrun", seed=seed, **base))
```

- **`run_id`** must be unique — it names the checkpoint files.
- **`out_dir`** is set by the `--out_dir` argument at launch; don't hardcode it in `base`.

### 2. Launch

```bash
mkdir -p results/dual/sweep_myrun
screen -dmS sweep_myrun bash -c "python sweep.py --out_dir results/dual/sweep_myrun \
    --n_gpus 2 --n_workers 16 --per_run_screen \
    2>&1 | tee results/dual/sweep_myrun/run.log"
```

- `--n_workers` is **total** across all GPUs (16 = 8/GPU for 2× A30).
- `--per_run_screen` spawns one `screen` session per seed: `sweep_s{seed}_myrun`.
- Attach to a seed: `screen -r sweep_s0_myrun`.
- Check all sessions: `screen -ls`.

### 3. Monitor progress

```bash
# How many seeds are done:
cat results/dual/sweep_myrun/results.jsonl | wc -l

# Live log for seed 0:
screen -r sweep_s0_myrun

# Tail the per-run log:
tail -f results/dual/sweep_myrun/s0_myrun/train.log
```

### 4. Check results table

```bash
python analyze.py --results results/dual/sweep_myrun/results.jsonl
```

---

## Reusing checkpoints (skipping stages)

To skip DPA or DPA+GNG and reuse checkpoints from a previous sweep:

```python
prev = "results/dual/sweep_prev"
configs.append(RunConfig(
    run_id=f"s{seed}_new",
    dpa_ckpt=f"{prev}/s{seed}_prev/dpa_s{seed}_prev.pth",   # skip DPA
    gng_ckpt=f"{prev}/s{seed}_prev/naive_s{seed}_prev.pth", # skip DPA+GNG
    ...
))
```

- Set only `dpa_ckpt` to reuse DPA and retrain from GNG.
- Set both `dpa_ckpt` and `gng_ckpt` to jump straight to Dual.
- Checkpoint paths: `<out_dir>/<run_id>/dpa_<run_id>.pth`, `naive_<run_id>.pth`, `expert_<run_id>.pth`.

---

## Key `RunConfig` parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `freeze_input_stages` | `["dual"]` | Stages where ALL input dims are frozen. Use `["gng","dual"]` for recurrent gating. |
| `freeze_gng_input_during_dpa` | `False` | Freeze go/nogo dims during DPA to prevent weight decay from zeroing them. |
| `freeze_rank0_dual` | `False` | Freeze rank-0 of m/n during Dual stage. |
| `optimizer` | `"adamw"` | `"adam"` (no weight decay) or `"adamw"`. |
| `use_scheduler` | `True` | ReduceLROnPlateau (patience=5, factor=0.5). Set `False` for constant lr. |
| `go_hinge_thresh` | `None` | If set, go response window uses `relu(thresh−pred)²` instead of MSE. |
| `gng_nogo_weight` | `1.0` | Weight on nogo trials in GNG/Dual loss. |
| `cue_scale` | `1.0` | Amplitude of the GNG cue signal. |
| `stop_loss` | `0.005` | Early-stop when both train and val loss fall below this. `None` to disable. |
| `dpa_ckpt` | `None` | Path to existing DPA checkpoint; skips DPA training. |
| `gng_ckpt` | `None` | Path to existing GNG checkpoint; skips DPA+GNG training. |

---

## Plotting

### All figures for a sweep

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python plot_sweep.py \
    --sweep_dir results/dual/sweep_myrun \
    --out_root results/figures
```

Output: `results/figures/sweep_myrun/{summary,individual}/`

### Subset of figure types

```bash
# acc and trajectories only (no scatter/flow):
... python plot_sweep.py --sweep_dir ... --plots acc traj

# Available: acc  traj  scatter  flow
```

### Subset of runs

```bash
... python plot_sweep.py --sweep_dir ... --run_ids s0_myrun s1_myrun
```

### Figure types produced

**Summary** (`summary/`):
- `accuracy_stages.pdf` — DPA/GNG accuracy across training stages.
- `accuracy_by_trialtype.pdf` — per trial type (pair/unpair, go/nogo) per stage.
- `fp_scatter_by_stage.pdf` — κ-plane fixed points across stages.
- `fp_scatter_by_input_<cond>.pdf` — fixed points per input condition.
- `traj_{dpa,naive,expert}_{dpa,go,nogo}.pdf` — mean κ trajectories from dual task.
- `traj_{dpa,naive,expert}_gng_task.pdf` — κ trajectories from pure GNG trials.

**Individual** (`individual/<run_id>/`):
- Same accuracy + trajectory figures as summary, plus:
- `scatter/fp_scatter.pdf` — κ-plane fixed point scatter.
- `flow/fp_{dpa,naive,expert}.pdf` — κ-plane flow fields.

### LD_PRELOAD note

All plotting scripts need this prefix — importing both `torch` and `matplotlib` causes a `CXXABI_1.3.15` symbol conflict without it:

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python plot_sweep.py ...
```
