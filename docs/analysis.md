# Analysis & Plotting

## Flow-field code map (2026-08 refactor)

**Which tool to run** (all need `LD_PRELOAD`):

| want | rank-2 | rank-3 |
|---|---|---|
| **SCORE a sweep vs the goal** (memory wells κ₁<0) | **`flow_verdict.py`** (+ `flow-verdict` skill — run this BEFORE interpreting any flow figure) | — |
| **SCORE the BEHAVIOUR** (κ(t) per stage vs the expected trajectory) | **`traj_verdict.py`** (+ `traj-verdict` skill — run this BEFORE interpreting any trajectory figure) | — |
| flow portrait (analytic) | `plot_sweep.py --plots flow` | `rank3_flow.py` |
| + input noise | `plot_sweep.py --field_input_noise` (MC) | `rank3_flow.py --noise` |
| GENUINE sim trajectories | `traj_flow.py` | `traj_flow.py --stage …` |
| autonomous well table | `scratchpad/wells3.py <sweep>` | same |
| slow-manifold dose ladder (+ sgd2 column) | `scratchpad/slow_manifold_dose.py` / `scratchpad/slow_manifold_sgd2.py` | `results/figures/sweep_r2cue2/slow_manifold_dose.png`, `results/figures/sweep_r2sgd2/slow_manifold_sgd2.png` |
| memory-axis field profile F₀/κ₀ + κ₀ traces + rate scale, relu vs lif | `scratchpad/relu_field_profiles.py` | `results/figures/sweep_r2rr01/field_profiles_relu_vs_lif.png` |
| EISTP / backbone | `ei_flow.py` | — |

**Gotchas:** plot_sweep's rank-2 FP finder **asserts rank==2** → never point plain `plot_sweep` at a
rank-3 or mixed-rank sweep (crashes / `(…,3)`-vs-`(…,2)` broadcast in the summary); scope with
`--run_ids`/`--plots`. `plot_sweep --use_sim_field` is a **one-step adiabatic map** (≈β·analytic), NOT
trajectories — use `traj_flow.py` for real integrated paths. Noise field = the validated input-only
exact term (`noise_sigma`), *not* the self-consistent DMFT (`solve_sc_variance`, experimental).
`plot_sweep --field_input_noise` **overwrites `fp_stages.png`** (same filename as the clean render) —
rename the clean set or the noise set to `fp_stages_noise.*` before publishing both (2026-08-10).
The **brainpy FP backend can return spurious fixed points** (sweep_r2sign2: claimed wells at κ₀≈±2.8
where the field is |F|≈1.4) — for load-bearing well tables use the scipy finder
(`find_all_fixed_points`) and spot-verify |F(κ*)| directly with `low_rank_field_np` (2026-08-10,
also affects `bifurcation_probe.py` which uses the brainpy pass).

**Where the code lives** (`src/`, split 2026-08; `dynamics.py` re-exports for back-compat):
- `flow_field.py` — shared rank-general ENGINE: `low_rank_field_np`/`_jacobian_flow_np` (+ `noise_sigma`),
  `low_rank_numpy_params`, noise (`_phi_avgs`, `solve_sc_variance`, `low_rank_field_sc_np`),
  `_canonical_flow_panels`, sim primitives (`_sim_step_single`, `sim_kappa_field`,
  `integrate_kappa_trajectories`), κ-projection.
- `flow_fixedpoints.py` — **shared rank-general `find_fixed_points(params, ff, backend="scipy"|"brainpy", …)`**
  (scipy root on the numpy field / brainpy SlowPointFinder on the jax field, `build_jax_field`) +
  `classify_lowrank_fps`; plus the rank-2 `find_all_fixed_points`/`classify_fixed_points` (plot_sweep).
- `flow_rank2.py` — rank-2 ANALYTIC rendering: `plot_stage_stacked_flow` (the `fp_stages` 3×8 portrait) + panels.
- `flow_rank3.py` — rank-3 ANALYTIC rendering: 3 pairwise-plane slices (`render_run`), uses the shared finder.
- `flow_traj.py` — rank-general TRAJECTORY rendering (`render_run`): integrates real paths from a κ-grid
  (via `integrate_kappa_trajectories`); split by *method* not rank. CLI: `traj_flow.py`.

The split axis: `flow_rank2`/`flow_rank3` = analytic field portraits (by rank); `flow_traj` = integrated
trajectories (rank-general). All sit on `flow_field` + `flow_fixedpoints`.


## Trajectory verdict — `traj_verdict.py` (2026-09-02)

The behavioural counterpart of `flow_verdict.py`: instead of *where the wells are*, it scores *what
κ(t) actually does on the task*, per stage, against Leon's spec (encoded once in `EXPECT` at the top
of the file, reproduced in the `traj-verdict` skill). Run it before interpreting any trajectory
figure — a traj panel shows one condition at one stage; this reads all of them at once.

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python traj_verdict.py \
    --sweep_dir results/dual/<sweep> [--run_ids …] [--stages dpa gng dual] [--all_probes] [--json out.json]
```
Probes: DPA task at `dpa_`; GNG **and** DPA tasks at `naive_` (the second is the `after_gng/dpa`
story, in trajectory form); Dual task at `expert_`. `--all_probes` adds the standalone DPA/GNG tasks
at `expert_`. ~10 s/run on CPU.

Checks: `mem` (κ₀ hold across the delay, categorised HELD/DECAY/FADE/GROW/LOST/**FLIP**), `rule`
(go/nogo sign + ±θ level before the cue), `cue` (Δκ₁ — nogo must go up, go must not go down),
`resp`, `relax` (transient vs parked), `nolick` (late-delay κ₁≤0, on the steps the loss left free),
`choice`, `prelick`. All sign tests at the **fixed boundary 0**, as in flow_verdict.

**Expectations adapt to the run's own loss** (`_variant`/`_adapt`, mirroring how `sweep.py` builds
`UnifiedLoss`): the spec's "±1" is really ±θ with θ⁺=`go_hinge_thresh`, θ⁻=−`nogo_hinge_thresh`,
θ_pair=`dpa_hinge_thresh`; hinges are one-sided so a level test is a FLOOR, and θ=0 (sign-based
arms) means the floor is σ_eff, not 1. Free targets (`nogo_target=None`, `rwd_nogo_onesided`,
`gng_response=False`) are reported but never scored; `nolick_late_delay`, `decay_to_zero`,
`gng_decay_to_zero` (GNG-stage only!), `kappa1_reg_weight`, `dual_gng_memory`, `dpa_prelick_free`
and `hinge_shape=softplus` each switch a check on/off or move its threshold; the 2026-09-02 flags
too — `nolick_full_delay` (κ₁≤0 scored over the WHOLE delay on the DPA trials), `nolick_thresh`
(ε echoed; scoring stays at boundary 0), `dpa_nolick_weight` (one-sided `prelick`, scored at the
DPA stage), `gng_decouple_decision` (tagged; expect `leak`≈0). Every adaptation is
printed as the per-run `variants:` line — **read it before quoting any level verdict**, and note
`✓/✗` = scored vs `·` = reported only. The `leak` check (GNG/dpa probe) = |κ₁(A)−κ₁(B)| over the
early-mid delay — the flip predictor (§26/§27).

Adding a generator flag to `sweep.run_single` means adding it to `_gen` here too, or the probe
silently drifts from what was trained.


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

Set via `XLIM = YLIM = (lo, hi)` at line ~65 of `plot_sweep.py`, or per-call with
`--xlim lo hi`. Choose based on the trajectory range for the nonlinearity used:

| Nonlinearity | Typical κ range | Recommended limits |
|---|---|---|
| tanh, erf, lif, lif_sc | ±1.0–1.2 | ±1.5 |
| tanh_reg, erf | ±1.0 | ±2.0 |
| relu, elu, softplus | up to ±3–5 | ±5.0 |

**The LOSS hinge shape moves the κ scale as much as φ does** (2026-08-12, spnld vs renld —
same `lif` φ, same design, only `hinge_shape` differing). A softplus hinge keeps paying for
overshoot after the target is met (gradient σ(x) never dies), so every amplitude inflates;
relu² releases at threshold and the state stays near ±1:

| `hinge_shape` | Measured well \|κ₀\| (lif, gain 2) | Limits |
|---|---|---|
| `softplus` | 2.6–3.3 (extras out to ±4) | **±4.5** |
| `relu2` | 1.1–1.5 | **±2.0** |

So `--xlim -4.5 4.5` on a relu² sweep wastes the panel (all structure inside the middle
quarter), and `--xlim -2 2` on a softplus sweep clips real wells. Pick from the run's own
`flow_verdict.py` well positions — widen if any attractor sits within 15% of the box edge,
narrow if the outermost is inside half the box.

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
| `traj_grid.pdf` (individual only, 2026-09-10) | **ONE figure per seed**: 8 cols × 3 rows — cols = (DPA-only, Go, NoGo, GNG-task) × (κ₀, κ₁), rows = DPA stage / After GNG / After Dual. Replaces the 12 per-condition `traj_*` files for individual runs. Pair/unpair (and Go/NoGo in the GNG columns) are overlaid as SOLID/DASHED since rows are now stages; y-limits shared down each column so stages compare directly; a missing ckpt renders "no ckpt". As of 2026-09-10 the SUMMARY uses it too (one seed-averaged grid per sweep), so `_plot_traj_figure`/`_plot_gng_traj_figure` are deprecated and unreferenced. |

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

### Noise-averaged field/fixed points — `--field_input_noise`
By default the flow **field + fixed points** use a clean, deterministic frozen input
(`make_input`, no noise); only the overlaid **trajectories** carry input noise. Pass
`--field_input_noise` to render the **noise-averaged** field `E_x[Ψ(κ)]` instead — each panel's
frozen input is replicated into K draws (default 16) with the run's training `noise_sigma()` added
per channel, and `low_rank_field_np` / `low_rank_jacobian_flow_np` average `φ` / the Jacobian over
them. Use it to check a result is not a clean-input artifact.
- **A single draw is NOT enough** — one noise vector projects through `Wi` as a *correlated* per-unit
  bias that tilts the field and drops a well ~half the time. K≥8 is stable (default 16); 64 is overkill.
- ~K× slower to render (the 151² grid), so keep it **off for routine plotting**; the deterministic
  field is the correct basis for geometry claims (isolation, well count, g·λ).
- Implemented as a 2D `ff_input` `(K, input_size)` to the numpy field/Jacobian (1D ⇒ K=1 ⇒ unchanged).

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

---

## Fixed-point finder & marginal handling (updated 2026-07-13)

`plot_sweep.py`'s individual flows and FP scatters, and the standalone `bifurcation_*` tools, now share:

- **brainpy `SlowPointFinder`** is the default FP finder in the `bifurcation_*` tools (`--finder brainpy`,
  scipy fallback; `--slow_tol`, `--marg` exposed). `plot_sweep`'s individual flows still use the internal
  scipy `find_all_fixed_points` (keeps the exact original multi-panel look).
- **Marginal cleanup** (`_reduce_marginals` in `src/dynamics.py`): a near-line-attractor makes almost
  every point "marginal". If the autonomous field is a resolved **bistable pair** (an attractor at κ₀>0.6
  *and* one at κ₀<−0.6), all marginal clutter is dropped. If a memory side is **missing**, the single best
  marginal (most extreme κ₀ on that side) is kept and **relabeled `slow_attractor`** (orange circle) — it
  is genuinely transversely attracting (~−0.6) with a near-neutral along-manifold direction, so it *is* the
  (soft) memory attractor, just not a stiff point. Applied to the Autonomous panel of the individual flows
  and the scatter/mean-flow summaries.
- Individual-flow **legend** → bottom-right, white text (on the dark magma Autonomous panel).

## New summary figures (per stage, replacing `fp_scatter_by_*`)

`summary_fp_scatters` now emits, for each stage `dpa/naive/expert` (via `plot_sweep --plots scatter`):

| File | Content |
|---|---|
| `summary/fp_scatter_{stage}.pdf` | One panel per input condition (Autonomous / Sample A,B / Test C,D / Go / NoGo); each scatters the **attractors + slow attractors across all seeds** (colour = init_style). Read across-seed consistency: tight cluster = robust, spread = seed-variable. |
| `summary/fp_meanflow_{stage}.pdf` | Same panels, but showing the **mean vector field** `⟨F_s(κ)⟩` (white streamlines) over a **background = across-seed flow agreement** `‖⟨F_s/‖F_s‖⟩‖`∈[0,1] (dark = seeds agree → mean flow trustworthy; light → they cancel, streamlines self-fade), with the attractors overlaid. Averaging is valid because the κ-plane is a shared, consistently-oriented coordinate system (Sample A always at +κ₀). |

**`--meanflow_overlay {scatter,kde}`** (default `scatter`): swaps the per-seed attractor dots for a
**KDE density** (plasma, `scipy.stats.gaussian_kde`, from the attractor + slow-attractor points only)
when the dots get busy — bright = many seeds land there; elongated smears reveal seed spread (e.g. Test
C/D). A **small white dot** marks each **KDE density peak/mode** (`_kde_modes`: local maxima ≥50% of peak,
deduped within 0.5) so the dots match the *visible clouds* (e.g. 2 in the Autonomous panel, 2 per bright
smear-end in Test C/D) rather than over-splitting a smear. A second colorbar ("attractor density") is
added (the Greys colorbar stays for agreement); in KDE mode the meaningless init_style patch is dropped
from the legend. Auto-falls back to the scatter per panel when a condition has <4 or degenerate points.

Both use `_reduce_marginals` on the Autonomous panel, so slow-manifold memory states appear as slow
attractors, not marginal clutter.

## Publishing to the gallery — `scratchpad/publish_gallery.sh` (2026-09-10)

```bash
./scratchpad/publish_gallery.sh sweep_r2sclnl      # -> ~/dual/rnn/<TITLE>/{traj,flow,accuracy,misc}
```

Two things it encodes, both of which were got wrong by hand first:

- **The gallery folder name comes from `results/dual/<sweep>/TITLE`** (falling back to the raw dir
  name). Leon's convention is short and lowercase — `<phi>_<init>_<what the arm adds>_<dpa if
  DPA-only>`, e.g. `lif_sub_cue_nolick_nogo`, `relu_sub_dpa`, `lif_termwin_dpa`. Write the TITLE
  file when launching the sweep. Raw `results/dual/sweep_*` dirs are NOT renamed — `dpa_ckpt`/
  `gng_ckpt` paths in `make_configs` point at them.
- **Classification uses the SOURCE SUBDIRECTORY, not the filename.** Flow-field panels live in
  `individual/<seed>/flow/` but are named `traj_naive_dpa.png`; a filename-only classifier files
  every one of them under `traj/` and the sweep appears to have a single flow figure.

Cost of the figures themselves (measured 2026-09-09, 4-run 3-stage sweep ≈ 71 PNGs, ≈ 28 min): no
single step is slow — flow field 151² = 1.31 s, `find_all_fixed_points` 0.88 s @21 seeds / 3.34 s
@41, trajectory sim 3.08 s (CPU), streamplot 0.33 s, savefig 0.07 s. It is the PRODUCT of panels
(conditions × stages), each needing its own field evaluation and fixed-point search — ≈ 23 s per
output PNG. DPA-only sweeps are ~3× cheaper. Levers: `--plots`, `--no_summary`, `--n_grid 101`,
`--n_fp_seeds 21`, `--device cuda:0` (defaults to CPU; only the sims move, the field is numpy).
