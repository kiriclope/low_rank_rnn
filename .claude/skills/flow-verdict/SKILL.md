---
name: flow-verdict
description: Score a rank-2 sweep's flow geometry against the project goal (sample-memory wells below the lick line) with flow_verdict.py, and read flow figures without the known misinterpretation traps. Use whenever asked to analyse/interpret flows or portraits, judge whether wells were pushed down, score an arm's geometry, or compare arms — BEFORE writing any interpretation of a flow figure.
---

# Flow verdict — how to score flows without fooling yourself

The project has ONE geometric success criterion (Leon, ring_lowerplane_log §23):
**the two SAMPLE-MEMORY wells — the ± κ₀ attractor pair that carries the A/B bit — sit at
κ₁ < 0 in the expert AUTONOMOUS field.** Nothing else counts as "pushdown".

## Step 1 — run the tool, never eyeball first

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python flow_verdict.py \
    --sweep_dir results/dual/<sweep> [--stage expert] [--run_ids ...] [--xlim 4.5] [--mem_k0 0.5]
```

Per run it prints: `ALL-DOWN ✓ / not down / NO PAIR ✗`, the memory wells (κ₀, κ₁) with an
`S` marker on spirals, the EXTRAS (non-memory attractors), dual_dpa, and behaviour at the
fixed boundary 0. The last line (`SUMMARY: k/N seeds all-down`) is the arm's score. Lead
every report with it. Figures come AFTER the table, as illustration — the table is the claim.

## The traps this skill exists to prevent (each one was committed in Aug 2026)

1. **A deep attractor at κ₀ ≈ 0 is NOT pushdown.** It carries no sample bit — it's a
   memory-less parked "basement" state (spw arms: (0, −3.2)). Only the ± κ₀ pair matters.
   The tool separates these as `extras`; never promote an extra into a well in prose.
2. **Behaviour is not geometry.** Deep nogo response means (even −2.9) can be entirely
   input-driven transient or basement-visiting; the wells may sit ABOVE the line
   (sign2w5/spw lesson: response-window pressure is absorbed; wells answer only to
   delay-time forces). Never infer well movement from response accuracy or means.
3. **One fixed decision boundary: 0.** Each arm's own eval midpoint depends on its
   thresholds ((th_go+nogo_ref)/2 — 0.5 for th=1 arms, 0.125 for ε=0.25) and flatters
   large-threshold arms. Cross-arm behaviour comparisons only at boundary 0 (the tool does).
4. **Verify fixed points; don't trust a finder.** The brainpy/SlowPointFinder path (incl.
   `bifurcation_probe.py`) returned spurious FPs (claimed wells at ±2.8 with |F|≈1.4).
   The tool uses the scipy finder AND re-checks |F(κ*)| directly. If you compute FPs any
   other way, spot-verify with `low_rank_field_np` before claiming anything.
5. **Match the box to the run's hinge shape — it is NOT one number.** The loss `hinge_shape`
   sets the κ scale as much as φ does: `softplus` keeps rewarding overshoot so wells inflate
   to |κ₀| ≈ 2.6–3.3 (extras to ±4) → use **±4.5**; `relu2` releases at threshold and wells
   sit at |κ₀| ≈ 1.1–1.5 → use **±2.0**. Too narrow clips real structure; too wide makes a
   relu² portrait unreadable (everything in the middle quarter). Widen if any attractor sits
   within 15% of the box edge, narrow if the outermost sits inside half the box. Same flag
   for figures: `plot_sweep --xlim -4.5 4.5` / `--xlim -2 2`. (Table: `docs/analysis.md`.)
6. **fp_stages figure gotchas:** the cyan markers at ≈(0.9, −1.4) in some panels are the
   LEGEND, not fixed points; `--field_input_noise` OVERWRITES `fp_stages.png` (rename to
   `fp_stages_noise.*` before publishing both).
7. **Bit-identical same-seed runs across arms = an inert loss term** (the rwd_window bug):
   check `dual_loss_components` in results.jsonl — the term you're dosing must be > 0.
8. Report per-seed (never average well positions across seeds), count attractors exactly,
   and name spiral wells (`S`) — complex Jacobian eigenvalues at a well matter.

## Reading `mem_k0` edge cases

`--mem_k0 0.5` classifies |κ₀| ≥ 0.5 attractors as memory wells. Borderline attractors
(|κ₀| 0.5–0.8, e.g. a (−0.66, +1.46) satellite) err on the side of counting as wells —
that makes ALL-DOWN harder, which is the conservative direction. If a run shows >2 memory
wells or a missing sign (`NO PAIR ✗`), the substrate itself is broken (parking/proliferation)
— say that, not "the wells are at ...".

## Companion tools

`bifurcation_probe.py` (g·λ table — but see trap 4), `plot_sweep --plots flow` (portraits),
`traj_flow.py` (real integrated trajectories), `bifurcation_flows.py` (nullcline figures).
History and calibration context: `docs/ring_lowerplane_log.md` §23.
