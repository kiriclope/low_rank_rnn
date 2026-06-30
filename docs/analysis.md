# Analysis & Plotting

## Main entrypoint: `plot_sweep.py`

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python plot_sweep.py \
    --sweep_dir results/dual/sweep_myrun \
    --out_root  results/figures
```

Output: `results/figures/sweep_myrun/{summary,individual}/`

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--sweep_dir` | required | Sweep directory containing `results.jsonl` |
| `--out_root` | required | Root for figure output |
| `--run_ids` | all | Restrict to specific run IDs |
| `--no_summary` | off | Skip summary figures |
| `--no_individual` | off | Skip per-run figures |
| `--plots` | all | Subset: `acc`, `traj`, `scatter`, `flow` |
| `--skip_flow` | off | Skip flow field computation |
| `--skip_scatter` | off | Skip FP scatter |
| `--n_fp_seeds` | 21 | Fixed-point finding seeds (use 41 for publication) |
| `--device` | cpu | Device for model forward passes |

### XLIM / YLIM

Set via `XLIM = YLIM = (lo, hi)` at line ~65 of `plot_sweep.py`. Choose based on the
trajectory range for the nonlinearity used:

| Nonlinearity | Typical κ range | Recommended limits |
|---|---|---|
| tanh, erf, lif, lif_sc | ±1.0–1.2 | ±1.5 |
| tanh_reg, erf | ±1.0 | ±2.0 |
| relu, elu, softplus | up to ±3–5 | ±5.0 |

Always check trajectory plots first to infer the right limits.

---

## Summary figures

Saved to `summary/`:

| File | Content |
|---|---|
| `accuracy_stages.pdf` | DPA/GNG acc at each stage, mean ± SEM across seeds |
| `accuracy_by_trialtype.pdf` | Per-trial-type breakdown |
| `fp_scatter_by_stage.pdf` | Autonomous FPs across all seeds, coloured by stage |
| `fp_scatter_by_input_<cond>.pdf` | FPs under each input condition |
| `traj_{dpa,naive,expert}_{dpa,go,nogo,gng_task}.pdf` | Mean κ trajectories |

---

## Individual figures

Saved to `individual/<run_id>/`:

| File | Content |
|---|---|
| `accuracy_by_trialtype.pdf` | Per-trial-type accuracy for this seed |
| `traj_*.pdf` | Trajectories for this seed |
| `scatter/fp_scatter.pdf` | FP scatter across all stages for this seed |
| `flow/fp_<stage>.pdf` | Phase portrait (flow field + FPs) at each stage |

---

## Results table

```bash
python analyze.py --results results/dual/sweep_myrun/results.jsonl
```

Loads all `status=="ok"` entries into a flat DataFrame (config + accuracy columns).

---

## Accuracy metrics

Computed with input noise on, recurrent noise off (`model.noise=0`), reward teacher-forced.

- **`_dpa_accuracy`**: κ₁ averaged after test, threshold at 0 (targets ±1).
- **`_gng_accuracy`**: κ₁ averaged after response window, threshold at
  `(1 + nogo_target) / 2`.
- **`_dual_accuracy`**: DPA read post-test (threshold 0) and GNG read in cue→test gap.
  Returns `dual_dpa`, `dual_gng`.

Keys in `results.jsonl`:
```
accuracy.after_dpa.{dpa, gng}
accuracy.after_gng.{dpa, gng}
accuracy.after_dual.{dpa, gng, dual_dpa, dual_gng}
```

---

## Fixed-point finding (`src/dynamics.py`)

`find_all_fixed_points` solves `κ = N⁻¹ nᵀ φ(gain·(input + Mκ))` via `scipy.root` with
a grid of initial seeds over the flow field domain. `merge_roots` deduplicates nearby
solutions.

The search grid is set by `XLIM/YLIM` in `plot_sweep.py` — too narrow and FPs outside
the window are missed (this was an issue for relu/softplus with FPs at κ≈±4).

---

## Binned (simulation-based) flow fields (`ei_flow.py`)

For models whose analytic κ-reduction is invalid — fixed-weight backbone, EI, **EISTP** —
`ei_flow.py` builds the flow field by **simulation**: a grid of states is injected along the
readout vectors, released, and the per-bin mean one-step drift is histogrammed over the κ-plane
(after NeuroFlame `compute_binned_flow_field`). Fixed points = converged endpoints, clustered and
split into **point attractors** (lime dots) vs **continuous manifolds** (orange locus).

```bash
LD_PRELOAD=… python ei_flow.py --sweep_dir results/dual/<sweep> --out_root results/figures \
    [--run_ids s0 s1] [--style magma|binned] [--device cuda:0]
```

| Flag | Meaning |
|---|---|
| `--style` | `magma` (default) = vanilla look (magma speed map + white streamlines); `binned` = coolwarm z-scored speed + black streamlines |
| `--run_ids` | restrict to specific runs |

- **EISTP: `plot_sweep` now auto-routes** (2026-06-24) the FP scatter + flow to the simulation
  path (`_model_has_backbone()` returns True for `EISTPModel` → sim scatter; `individual_flow`
  delegates to `ei_flow.make_stage_figure`; scatter axes widen to ±15). So a plain full `plot_sweep`
  run is safe and complete for eistp — no crash, no overwrite. `ei_flow.py` remains for flow-only /
  `--style binned`.
- For **static-backbone / EILowRankModel** sweeps the older caveat still holds: prefer `ei_flow.py`
  or `--plots acc traj scatter` (their analytic flow ignores `W_fixed` / the EI structure and would
  overwrite `ei_flow`'s `individual/<rid>/flow/fp_<stage>.pdf`).
- **EISTP** has a dedicated path (`_run_grid_eistp`): exact two-timescale + Markram STP dynamics,
  auto-calibrated grid injection (the κ range is ~±10, not ±1.5), R=15, T=600. No tuning needed.
- Default sim length elsewhere: T=1333 steps ≈ 30 s (so genuine wells settle; a slow ring shows as a
  manifold, not arbitrary dots).

---

## Legacy single-run scripts (`plots/`)

These take `--ckpt_dir` / `results.jsonl` directly and are useful for quick inspection:

| Script | Purpose |
|---|---|
| `plot.py` | Basic accuracy + loss curves |
| `plot_fixed_points.py` | Fixed points for one run |
| `plot_trajectories.py` | κ trajectories for one run |
| `plot_fp_scatter.py` | FP scatter for one run |
| `plot_ring.py` | Ring visualisation |
| `plot_dpa_by_trialtype.py` | DPA accuracy split by trial type |
