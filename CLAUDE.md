# CLAUDE.md

Guidance for Claude Code in this repository.

## Documentation — read first

| File | Contents |
|---|---|
| [docs/overview.md](docs/overview.md) | Research question, κ-plane, structural hypothesis |
| [docs/architecture.md](docs/architecture.md) | Model, tasks, training pipeline, loss functions, nonlinearities |
| [docs/running.md](docs/running.md) | Sweep workflow, rerunning Dual, GPU tuning |
| [docs/analysis.md](docs/analysis.md) | plot_sweep.py flags, figure types, XLIM guide |
| [docs/experiment_log.md](docs/experiment_log.md) | All sweeps: configs, results, status |
| [docs/nonlinearities.md](docs/nonlinearities.md) | Ring vs attractor theory, all nonlinearity results |
| [docs/ring_lowerplane_log.md](docs/ring_lowerplane_log.md) | Active thread: structural collapse of κ₀ memory, go vs nogo, lower-plane attractors |
| [docs/theory_landscape.md](docs/theory_landscape.md) | Math: κ-plane potential V, why tanh ⇒ symmetric wells, pitchfork/ring, slow manifold, per-stage landscape |

## What this is

Rank-2 low-rank RNNs learning three cognitive tasks **sequentially**: DPA → GNG → Dual.
Core question: how does the geometry of learned representations support (or fail) multi-task
retention? The key metric is `after_gng/dpa` accuracy — how much DPA survives GNG training.

The κ-plane (κ₀ = memory, κ₁ = decision) is the main analysis space. The goal is a
**ring attractor** in κ₀ after DPA that persists through Dual while κ₁ develops
input-driven go/nogo attractors.

## Directory layout

```
rnn/
├── src/            models.py, tasks.py, train.py, init.py, dynamics.py
├── docs/           ← project documentation (start here)
├── plots/          legacy single-run scripts
├── sweep.py        RunConfig dataclass + sweep runner
├── plot_sweep.py   main plotting entrypoint
├── rerun_dual.py   retrain only the Dual stage from checkpoints
├── analyze.py      load_results, summary_table
└── results/
    ├── dual/       run dirs: results.jsonl + models/ + run.log
    └── figures/    plot output
```

## Essential commands

**LD_PRELOAD** (required for any script importing both torch and matplotlib):
```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python <script>.py
```

**Render a math-heavy doc to PDF** (no LaTeX engine on the box → pandoc MathML + headless Chrome):
```bash
./make_pdf.sh docs/theory_landscape.md          # → docs/theory_landscape.pdf
```

**Launch a sweep** (edit `make_configs` in `sweep.py` first — **always show parameters to the user and wait for confirmation before launching**):
```bash
mkdir -p results/dual/sweep_myrun
screen -dmS sweep_myrun bash -c "python sweep.py \
    --out_dir results/dual/sweep_myrun --n_gpus 2 --n_workers 10 --per_run_screen \
    2>&1 | tee results/dual/sweep_myrun/run.log"
```

### Screens & logging — the rules (do this every time, don't improvise)

**One detached screen per RANDOM SEED, and one log per seed.** Not one screen for the
whole sweep, not one consolidated log. This is the `--per_run_screen` mode
(`sweep.py:_launch_per_run_screens`) — the default way to run sweeps here. Applies to
`sweep.py` and `nb_sweep/sweep.py` alike.

1. **One screen per seed/run**, round-robin across GPUs, named `sweep_<run_id>` (e.g.
   `sweep_s3_myrun`). Launch with `--per_run_screen`:
   ```bash
   screen -dmS sweep_myrun bash -c "python sweep.py --out_dir results/dual/sweep_myrun \
       --n_gpus 2 --per_run_screen 2>&1 | tee results/dual/sweep_myrun/launch.log"
   ```
   The launcher itself spawns the per-seed screens and exits; each seed runs in its own
   `sweep_<run_id>` screen.
2. **One log per seed:** each per-seed screen tees to its OWN `<run_dir>/train.log`
   (i.e. `results/dual/<sweep>/<run_id>/train.log`) — inside its `bash -c "..."`. There is no
   shared consolidated `run.log`; do not build one.
3. **Never run a sweep in the foreground** — it blocks the turn and skips the per-seed screens.
4. Re-launching is a safe *resume* — finished runs (whose `*_seed<s>.pth` exists) are skipped.
5. **Monitor / attach / kill:** `screen -ls` · `tail -f <run_dir>/train.log` ·
   `screen -r sweep_<run_id>` · kill one with `screen -S sweep_<run_id> -X quit`
   (all: `pkill -f run_sequence`).

**Plot a sweep:**
```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python plot_sweep.py \
    --sweep_dir results/dual/sweep_myrun --out_root results/figures
```

**Check results:**
```bash
wc -l results/dual/sweep_myrun/results.jsonl
python analyze.py --results results/dual/sweep_myrun/results.jsonl
```

**Rerun only Dual stage:**
```bash
python rerun_dual.py --sweep_dir results/dual/new --source_dir results/dual/orig \
    --ckpt_prefix naive --epochs_dual 200 --n_gpus 2 --n_workers 10
```

## Key gotchas

- `input_size` in `results.jsonl` is already post-`__post_init__` (7 if `cue_on_go_input=True`).
- `--n_workers` is **total** across all GPUs, not per-GPU. Sweet spot: 4–8 per GPU.
- Always `screen -dmS` with `tee` inside `bash -c "..."` — see **Screens & logging — the rules** above (and [docs/running.md](docs/running.md)).
- Default `XLIM/YLIM` is ±1.5 (tanh/erf/lif). Pass `--xlim -5 5` for relu/elu/softplus sweeps.

## Model quick-reference

`W_rec = m @ n.T / N`, `κ = rates @ n / N` (no output layer). Per-step:
```
h     ← exp(-α_rec)·h  + (1-exp(-α_rec))·(W_rec·rates)
rates ← exp(-α)·rates  + (1-exp(-α))·φ(gain·(Ai·Wi·x + h))
```

Nonlinearities: `tanh`, `relu`, `softplus`, `erf`, `elu`, `lif`, `lif_sc`.
Ring-capable (odd + saturating): **tanh** and **erf** only.

**★ EISTPModel (`model_type="eistp"`)** — the model that achieves the project goal (persistent
memory + lower-plane wells). NeuroFlame EI+STP port: sparse EI, relu, two timescales, Markram STP
on E→E, low-rank `m,n` riding the STP synapses `W_EE=gain·j_stp·(C/√K)·(1+n@mᵀ/lr_scale)`.
**Two scalings work** (g_mem = √K·⟨mn⟩/lr_scale): `eistp_lr_scale="sqrtK"` (÷√K → g_mem≈1 at
init, our conservative regime lr≤0.05+clip) **or** `"N"` (÷N_E, = NeuroFlame `'all'` — only
ignites in the original regime: lr=0.1, no grad clip; lower lr+clip throttle the ‖m,n‖ growth and
it stays dead). `plot_sweep.py` now auto-routes eistp to simulated FP scatters/flows (no crash;
`ei_flow.py` still available standalone). Full how-to: `docs/running.md` (EISTP section); science:
`docs/ring_lowerplane_log.md` §11.

Three training stages with selective freezing:
1. **DPA** — all params free.
2. **GNG** — freeze rank-0 of m/n + DPA/reward input dims (+ all inputs if `freeze_input_stages` includes `"gng"`).
3. **Dual** — freeze all input dims; optionally freeze rank-0 (`freeze_rank0_dual=True`) and penalise λ₁>1 (`kappa1_reg_weight`).

Freezing = zero grads + **restore** original values after optimizer step (makes it exact under AdamW).

## RunConfig fields to know

| Field | Default | Meaning |
|---|---|---|
| `nonlinearity` | `"tanh"` | φ function |
| `gain` | `2.0` | scales full net input |
| `init_style` | `"structured"` | `"structured"` or `"random"` |
| `freeze_input_stages` | `["dual"]` | stages with all inputs frozen |
| `freeze_rank0_dual` | `False` | also freeze rank-0 during Dual |
| `kappa1_reg_weight` | `0.0` | penalise gain·λ₁>1 during Dual |
| `stop_loss` | `0.005` | early-stop threshold (use 0.1 in practice) |
| `cue_on_go_input` | `True` | merge GNG cue onto go channel (input_size → 7) |
| `dual_loss` | `"separated"` | `MaskedMultiTargetDualLoss` with per-component weights |
| `dpa_ckpt` / `gng_ckpt` | `None` | skip stages by loading checkpoint |

## Security
- **Never print, display, or repeat the value of any variable or file containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD`** — extract and use silently only

## Cheap-worker delegation
A CLI is on PATH that route bulk I/O and predictable text generation
to a cheap OpenAI-compatible model (DeepSeek V4 Flash by default; V4 Pro
for ask-kimi). Use it when the task is bulk reading or boilerplate,
not when reasoning or correctness is on the line.

### Example documentation delegation
**NEVER write documentation directly. Always delegate:**
1. Extract chat: `extract-chat <latest-session.jsonl> -o /tmp/chat.txt`
2. Ask worker to read chat + existing docs and suggest updates:
   `ask-kimi --paths /tmp/chat.txt <doc-files> --question "read chat, give exact changes for docs"`
3. Apply the worker's changes via Edit tool

### Restrictions
**When NOT to delegate:**
- Tasks under ~2000 tokens of work (delegation overhead isn't worth it)
- Architectural decisions, debugging, safety-critical code
- Anything requiring careful reasoning
- When exact line numbers are needed for editing
- Anything touching auth, payments, PII, deletion, or production data
- Final commits and PR descriptions
