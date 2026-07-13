---
name: bifurcation-probe
description: Probe a low-rank RNN sweep for the decision-mode bifurcation — compute the mode self-gains g·λ₀/g·λ₁, find and classify the autonomous fixed points (wells), tabulate against accuracy, and render real reduced-field κ-plane flow figures (attractors/saddles/repeller + F₁=0 nullcline). Use when asked to check g·λ / self-gain / the pitchfork, find the wells / isolated low wells / ring / U, ask "did it find the sweet spot", plot bifurcation/flow fields for specific runs, or make the generic Gaussian bifurcation illustration.
---

# Bifurcation probe

The recurring analysis for the ring→lower-plane thread: given a trained low-rank sweep,
read off where each run sits on the **decision-mode pitchfork** and what the κ-plane wells
look like. Three tools at the repo root, all pure-analytic (fast, exact for `LowRankModel`).

**Always prefix with `LD_PRELOAD`** (torch + matplotlib):
`LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python …`

## The physics (how to read the output)

Reduced field `F_r(κ) = (1/N) Σ_i n_{ir} φ(g(m_i·κ + I_i)) − κ_r`. Origin Jacobian overlaps:
`g·λ_r = gain·n_rᵀm_r/N`. The **decision self-gain g·λ₁** is the pitchfork parameter:

| g·λ₁ | decision mode | κ-plane |
|---|---|---|
| ≫1 (≈3.5) | strongly bistable | ring / **270° U** (3 wells + saddles) |
| ≈1 | critical (soft) | **two isolated wells** — the sweet spot |
| <1 | monostable | wells collapse toward one |

Sweet spot = **exactly 2 attractors, both at κ₁<0, g·λ₁≈1**, with DPA/go/match-nonmatch intact
(nogo is the fragile one). The `kappa1_reg_weight` penalty is what drives g·λ₁→1; gain scaling
only partially lowers it (see `docs/ring_lowerplane_log.md` §13, `docs/theory_landscape.md`).

## 1. Probe table — `bifurcation_probe.py`

Per run: `gain`, `g·λ₀`, `g·λ₁`, the off-diagonal overlaps (mode coupling), attractor/saddle
counts, the attractor ("well") positions, and after-Dual dpa/go/nogo from `results.jsonl`.

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python bifurcation_probe.py \
    --sweep_dir results/dual/sweep_gainscan [--run_ids s0_g05 s1_g05] [--xlim 2.5] [--csv t.csv]
```
Start here — it answers "did it find the sweet spot" in one table.

## 2. Flow figures — `bifurcation_flows.py`

Per-run κ-plane flow: magma streamplot of the autonomous field + classified fixed points +
F₁=0 nullcline + shaded no-lick (κ₁<0) half-plane + `g·λ₁` label. → `results/figures/<sweep>/bifurcation/<rid>.pdf`.

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python bifurcation_flows.py \
    --sweep_dir results/dual/sweep_gainscan --out_root results/figures \
    [--run_ids s0_g05 s0_g10] [--xlim 2.0] [--field_noise]
```
`--field_noise` renders the noise-averaged field `E_x[Ψ]` (K=16) to check a result isn't a
clean-input artifact. This complements `plot_sweep.py` (same field) but labels g·λ₁ + draws the
nullcline; for full standard figure sets still use `plot_sweep.py`.

## 3. Generic Gaussian illustration — `bifurcation_gaussian.py`

Theory figure (no checkpoints): Gaussian-drawn populations, sweep the decision self-gain across
its pitchfork with the symmetry-break bias off (odd) vs on (lowers). Shows the mechanism is
generic, not weight-specific.

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python bifurcation_gaussian.py \
    [--N 512] [--gain 2.0] [--gl0 4.0] [--gl1 0.6 1.0 2.0 3.5] [--beta 0.5] \
    [--out results/figures/theory/bifurcation_gaussian]
```
Set `--N` to the sim `hidden_size` (512) to check finite-N robustness.

## Gotchas

- **`--xlim`**: low-gain / deep-memory runs push memory wells past ±1.5 (to ~±1.5–2). Widen the
  fixed-point search (`bifurcation_probe --xlim 2.5`) and the figure window (`bifurcation_flows --xlim 2.0`)
  or wells get missed / clipped.
- **Attention**: the tools auto-detect `attention_input` and turn the tonic last channel ON in the
  autonomous field (matches training; makes the origin a fixed point). Don't override.
- **`LowRankModel` only**. eistp/ei models → use `ei_flow.py` (simulated field); these analytic
  tools will refuse them.
- `find_all_fixed_points` returns `(fps, residuals)` — a tuple (the helpers already unpack it).
- Probe/flows import each other; run them from the repo root (they add it to `sys.path`).
