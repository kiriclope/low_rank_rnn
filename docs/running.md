# Running Experiments

## ★ Isolation recipe (vanilla rank-2) — two isolated low memory wells

Goal: A/B memory wells both at κ₁<0, as **two isolated wells** (not the 270° ring/U). Two
ingredients (docs/ring_lowerplane_log.md §13): **LOWER** (directional break) + **ISOLATE** (drive
the decision self-gain g·λ₁→1 so its autonomous wells vanish and the ring collapses).

**Reference config = `sweep_kappa1reg` reg1 (the winner):**
```python
shared = dict(
    model_type="lowrank", gain=2.0, hidden_size=512, rank=2, target_rank=2,
    nonlinearity="tanh",            # plain odd tanh — symmetry break comes from attention
    attention_input=True,           # tonic bias: makes both wells able to go κ₁<0 (replaces tanh_asym)
    nolick_weight=0.5,              # one-sided relu(κ₁)² over free windows (downward direction)
    hinge_gng=True,                 # hinges ALL 3 stages (go/nogo one-sided; match/nonmatch symmetric ±1)
    memory_lambda=0.8,              # memory SUPERCRITICAL (deep A/B wells)
    decision_lambda=0.25,           # decision starts SUBCRITICAL (g·λ₁=0.5 at init)
    cue_on_go_input=True, cue_scale=2.0, nogo_target=0.0,
    go_hinge_thresh=1.0, dpa_hinge_thresh=1.0,
    optimizer="adam", learning_rate=0.01, stop_loss=0.1,
    epochs_dpa=100, epochs_gng=100, epochs_dual=100,
)
# arm: kappa1_reg_weight=1.0   (Dual penalty w·relu(gain·n₁ᵀm₁/N − 1)² → pins g·λ₁ to ~1)
```
- **`kappa1_reg_weight` = the ISOLATE knob.** w=0 → g·λ₁≈3.5 (ring); **w=1 → g·λ₁=1.0, two isolated
  wells at (±1,−0.9)**, DPA/match-nonmatch/go all 1.0, nogo 0.79. w>1 buys no more isolation, only
  costs nogo (→0.43 at 3, →0.15 at 6). Sweet spot ≈ w=1 (finer: {0.5,1,1.5,2}).
- **τ matters for optimization, not isolation:** τ=0.3 is the sweet spot; τ<0.3 (fast) stalls DPA and
  the Dual never converges; τ>0.3 (slow) *raises* g·λ₁. Set via `tau` (RunConfig; α=dt/τ).
- **Read-out:** g·λ₁=gain·n₁ᵀm₁/N (want ≈1), autonomous fixed-point count (want 2), their κ₁ (want <0),
  and `dual_go`/`dual_nogo` (per-side, in results.jsonl `after_dual`).

### The `hinge_gng` flag (decision-channel loss shape, all stages)
- **True** → hinge losses everywhere: DPA=`ThresholdLoss` (symmetric ±th); GNG/Dual go≥`go_hinge_thresh`,
  nogo≤`nogo_hinge_thresh` (−1 in the GNG memory delay, 0 after cue); match/nonmatch symmetric ±`dpa_hinge_thresh`.
- **False** → pure MSE toward the targets on the decision channel, all stages.
- `--hinge_gng {0,1}` CLI overrides it. `nolick_weight` (one-sided free-window pressure) is separate and
  applies in both modes; it excludes the sample window and floors the Dual loss at ~0.13 (never hits 0.1).

---

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
eistp_lr_scale="sqrtK",              # ÷√K → g_mem≈1 at init (conservative regime; see note below)
low_rank_scale=1.0,                  # lr_ini (=1 → memory mode starts critical)
eistp_lr_ueqv=False,                 # random init (m,n independent) — works; True = m init n
eistp_r_max=500.0,                   # rate cap, ~6× the ~80 operating peak (anti-runaway)
eistp_init_noise=1.0,                # init kick rates₀=relu(ff₀+init_noise·randn); 0 = deterministic/frozen
stp_U=0.05, stp_tau_f=1.0, stp_tau_d=0.2,   # Markram STP
j_stp=1.0,
go_hinge_thresh=1.0, dpa_hinge_thresh=1.0,  # hinge targets (relu can't hit exact ±1)
learning_rate=0.05, grad_clip_norm=1.0,     # clipping ON — needed; without it ~1-2 seeds diverge in Dual
batch_size=32, n_batch=256, epochs_dpa/gng/dual=100,
```

**`eistp_lr_scale` — two working scalings (2026-06-24):** g_mem = √K·⟨mn⟩/lr_scale.
- `"sqrtK"` (÷√K, NeuroFlame `'sparse'`): g_mem≈1 *at init* → works in our **conservative** regime
  (lr≤0.05 + `grad_clip_norm=1.0`). This is the default.
- `"N"` (÷N_E, NeuroFlame `'all'` — what the original notebook uses): g_mem≈0.015 at init, so it
  **only ignites in the original regime** — `lr=0.1`, **no grad clip**, j_stp=1 (the optimizer grows
  ‖m,n‖ ~10× to compensate). Reproduced clean 5/5 in `sweep_eistp_ablate_all` (DPA 1.0, dual_dpa
  0.999). With lr≤0.05+clip, `"N"` stays dead — that earlier "÷N is dead" claim was a **regime
  artifact**, not a scaling barrier.

**Best DPA-through-GNG retention:** `j_stp=5` + `lr=0.01` (`sweep_eistp_jstp5_lr01`, after_gng/dpa
0.93 — project best; the 5× recurrent gain is held by the rate cap + clip).

**Frozen inputs:** `eistp_init_noise=0` makes the forward fully deterministic (the feedforward
signal+noise is already fixed once per stage). Tested (`sweep_eistp_frozen`): ~identical accuracy,
**no convergence speedup** (still ~150 DPA epochs, not the notebook's 10), no generalization loss.
So a frozen dataset is *not* the lever behind the notebook's fast training.

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
As of 2026-06-24 `plot_sweep.py` **auto-detects eistp** and routes the FP scatter + flow to the
simulation-based path (sim attractors, eistp-calibrated grid R=15/T=600, κ-axes widened to ±15) —
so a plain run produces the full set (acc + traj + scatter + flow) with **no crash, no `--plots`
workaround**:
```bash
LD_PRELOAD=… python plot_sweep.py --sweep_dir results/dual/sweep_eistp_myrun --out_root results/figures
# ei_flow.py is still available standalone for flow-only / --style binned:
LD_PRELOAD=… python ei_flow.py --sweep_dir results/dual/sweep_eistp_myrun \
    --out_root results/figures --device cuda:1 [--style magma|binned]
```
(The analytic FP scatter does not apply to eistp — its sim substitute is the attractors drawn in
the flow/scatter figures.)

### Stability / divergence
The `/√K` coupling + STP is near-critical and can run away (rates → ∞). Three guards (all on):
1. **Rate cap** `eistp_r_max` (clamps rates; only catches runaway, science untouched).
2. **NaN-skip** in `Optimization` — skips a non-finite batch/grad instead of corrupting weights.
3. **Graceful divergence** — if a whole epoch goes non-finite, `_run_epoch` returns `nan` →
   `fit()` stops that run and keeps the best pre-divergence state (records a result, no crash).
Plus `grad_clip_norm=1.0` (keep it on — it's the difference between 5/5 and ~4/5).
