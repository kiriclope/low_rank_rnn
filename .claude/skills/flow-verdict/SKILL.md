---
name: flow-verdict
description: Score a rank-2 sweep's flow geometry against the project goal (sample-memory wells below the lick line) with flow_verdict.py, and read flow figures without the known misinterpretation traps. Use whenever asked to analyse/interpret flows or portraits, judge whether wells were pushed down, score an arm's geometry, or compare arms — BEFORE writing any interpretation of a flow figure.
---

# Flow verdict — how to score flows without fooling yourself

The project has ONE geometric success criterion (Leon, ring_lowerplane_log §23):
**the two SAMPLE-MEMORY wells — the ± κ₀ attractor pair that carries the A/B bit — sit at
κ₁ < 0 in the expert AUTONOMOUS field, input-noise-averaged (the field the trials follow — trap 9).**
Nothing else counts as "pushdown".

## Step 1 — run the tool, never eyeball first

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python flow_verdict.py \
    --sweep_dir results/dual/<sweep> [--stage expert] [--run_ids ...] [--xlim 4.5] [--mem_k0 0.5] [--field both]
```
`--field both` (default) scores the input-noise-averaged field and prints the deterministic field's wells on a
second indented line; `--field noise` / `--field clean` print one field only. Each memory well carries a tag of the
Dual trial types that sit nearest to it at the end of the delay (per sample, ≥ 50 %): `N` no Go/NoGo, `G` Go,
`X` NoGo. An upper well tagged `[G]` is where Go trials park after the response (licking after the Go cue is not
priced), not a lost sample memory — say so, and still count it: ALL-DOWN means every memory well below.

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
   `fp_stages_noise.*` before publishing both). Render gallery flows WITH `--field_input_noise` (trap 9):
   the default clean render draws wells the trials do not use.
7. **Bit-identical same-seed runs across arms = an inert loss term** (the rwd_window bug):
   check `dual_loss_components` in results.jsonl — the term you're dosing must be > 0.
8. Report per-seed (never average well positions across seeds), count attractors exactly,
   and name spiral wells (`S`) — complex Jacobian eigenvalues at a well matter.
9. **Score the INPUT-NOISE-AVERAGED field, not the deterministic one** (2026-09-25). The nets are trained with
   input noise, which lowers each unit's effective gain by 1/sqrt(1 + g²σ²‖w_i‖²) — median 0.55–0.72 in trained
   nets, 10 % of units ≤ 0.4. The trials follow the averaged field: s1_log's end-of-delay state (+0.94, −0.56)
   sits on its averaged well (+0.93, −0.58), not the deterministic one (+1.23, −0.34); noise-free, several nets
   even lose the sample (A and B land in one well). The two fields share their topology at λ = 7 but the
   deterministic wells sit further out and higher, and "all down" flips: recipe7_bfix 5/8 deterministic vs 8/8
   averaged (the published 8/8 is the averaged count); logsub (λ = 2) has a different topology. Before this date
   the tool scored the deterministic field — rescore old sweeps before comparing (`verdict_noiseavg.log`).

## Reading `mem_k0` edge cases

`--mem_k0 0.5` classifies |κ₀| ≥ 0.5 attractors as memory wells. Borderline attractors
(|κ₀| 0.5–0.8, e.g. a (−0.66, +1.46) satellite) err on the side of counting as wells —
that makes ALL-DOWN harder, which is the conservative direction. If a run shows >2 memory
wells or a missing sign (`NO PAIR ✗`), the substrate itself is broken (parking/proliferation)
— say that, not "the wells are at ...".

## Companion tools

**`traj_verdict.py` (+ the `traj-verdict` skill) — the behavioural half of the verdict**: this file
scores where the wells are, that one scores what κ(t) does on the task per stage. Neither substitutes
for the other (trap 2); when asked "is this run good?", cite both tables.
`bifurcation_probe.py` (g·λ table — but see trap 4), `plot_sweep --plots flow` (portraits),
`traj_flow.py` (real integrated trajectories), `bifurcation_flows.py` (nullcline figures).
History and calibration context: `docs/ring_lowerplane_log.md` §23.
