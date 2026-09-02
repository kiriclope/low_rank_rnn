---
name: traj-verdict
description: Score a run's TRAJECTORIES (κ(t) on the task, per stage) against the expected behaviour with traj_verdict.py, instead of eyeballing traj figures. Use whenever asked whether a network "behaves right", why a trajectory looks weird, what GNG did to the DPA memory, whether the nolick/decay/response terms actually took, or before interpreting any trajectory figure or per-stage accuracy.
---

# Trajectory verdict — score the behaviour, don't eyeball the traces

`flow_verdict.py` scores the autonomous LANDSCAPE (where the wells sit). This scores what the
network actually DOES on the task. The expected κ(t) (Leon, 2026-09-02) lives in one place —
`EXPECT` at the top of `traj_verdict.py`:

| stage (ckpt) | probe | expected |
|---|---|---|
| DPA (`dpa_`) | DPA | A/B memory on κ₀ maintained or decaying, sample-off → test; choice on κ₁ at test offset |
| GNG (`naive_`) | GNG | go/nogo held on κ₁ at ±θ, maintained or transient; the cue pushes BOTH classes toward +κ₁; a response is expressed (sometimes wrong) |
| GNG (`naive_`) | DPA | A/B memory maintained or slightly disrupted; choice still on κ₁, sometimes wrong |
| Dual (`expert_`) | Dual | A/B memory maintained or decaying; go/nogo as in GNG; choice on κ₁, mostly right |

## Step 1 — run the tool, never eyeball first

```bash
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python traj_verdict.py \
    --sweep_dir results/dual/<sweep> [--run_ids ...] [--stages dpa gng dual] \
    [--all_probes] [--verbose] [--json out.json]
```
~10 s per run (4 probes, n=768, CPU). Per run it prints a `variants:` line, one line per
(stage, probe) with every check and its number, and a per-arm `SUMMARY` + `TRAJ OK: k/N`.
Lead the report with the SUMMARY line; figures come after, as illustration.

`--all_probes` adds the standalone DPA and GNG tasks at the expert checkpoint — the trajectory
counterpart of `after_dual/dpa` and `after_dual/gng`.

## ★ Expectations follow the RUN'S OWN loss — check the `variants:` line first

The "±1" in the spec is really **±θ**, and θ is per-arm. `_variant`/`_adapt` read it off the config
exactly as `sweep.py` builds `UnifiedLoss`: θ⁺ = `go_hinge_thresh` (else 1), θ⁻ = −`nogo_hinge_thresh`,
θ_pair = `dpa_hinge_thresh` (else 1). Also adapted: `nogo_target=None` / `rwd_nogo_onesided` (that
side is FREE → reported, not scored), `gng_response=False` (no response window), `nolick_late_delay`
(adds the late-delay κ₁≤0 check), `dpa_prelick_free`, `decay_to_zero` / `gng_decay_to_zero` /
`kappa1_reg_weight` (demand a transient → `relax` becomes scored), `dual_gng_memory=False` (the Dual
rule level is unsupervised → sign scored, level reported), `hinge_shape="softplus"` (raised ceiling).
2026-09-02 flags: `nolick_full_delay` (the κ₁≤0 check covers the WHOLE delay on the DPA trials —
the rows with no go/nogo stimulus and no cue), `nolick_thresh` (ε echoed in the txt; scoring stays
at boundary 0 — depth is flow_verdict's job), `dpa_nolick_weight` (`prelick` becomes a ONE-SIDED
scored check at the DPA stage — κ₁<0 is free by design), `gng_decouple_decision` (tagged; expect
`leak` ≈ 0).

**Never quote a level/threshold verdict without reading that line.** A ±1 band copied from a th=1
arm fails every sign-based arm for behaving exactly as designed — that is the mistake this
adaptation exists to prevent.

## Reading the output

- `✓ / ✗` = **scored**; `·` = **reported only** — nothing in this arm's loss demands it, so it never
  decides the verdict. Don't call a `·` a pass or a failure; quote its number.
- **Memory categories** come from `hold = amp(delay end)/amp(delay start)` plus `sep`:
  `HELD` ≥0.8 · `DECAY` 0.3–0.8 · `FADE` <0.3 · `GROW` >1.25 · `LOST` (sep < sep_min) ·
  **`FLIP` (sep < 0.45 — the sample bit is INVERTED, not lost)**. FLIP explains sub-chance DPA
  (0.28 is not noise around 0.5, it's the readout running backwards) — always say which of
  LOST/FLIP/FADE it is; "the memory degraded" throws away the finding.
- `GROW` on the Dual probe means the sample kick lands small and the attractor pulls it out during
  the delay — real, and NOT "maintained"; report the hold ratio.
- **`nolick` reports steps>0 and trials>0 as well as the mean**: the loss is a MEAN, so a violating
  minority hides behind a negative average (the 2026-08-13 trap). A run with mean −0.01 and
  trials>0 = 0.35 is failing the constraint on a third of its trials.
- The nolick check scores exactly the steps whose target the loss left **free** (`isnan(y)`), not
  the whole window — the go response and decay pins inside the span have their own terms.
- **`leak`** (GNG/dpa probe) = |κ₁(A)−κ₁(B)| over the early-mid delay: the sample memory leaking
  onto the decision axis. It is the sharpest predictor of the GNG memory INVERSION (§26/§27:
  ≥0.85 → flip 6/6, ≤0.27 → intact 5/5; threshold 0.50 calibrated on 12 runs — indicator, not
  constant). The raw overlap g·n₁ᵀm₀ does NOT predict it; measure the realised leak.
- `cue`: nogo must be driven strictly up; go only has to not be pushed DOWN (it's usually already
  at its hinge ceiling before the cue, where Δ≈0 is correct behaviour).

## Traps

1. **Behaviour ≠ geometry** (and vice versa). This tool cannot tell you where the wells are;
   `flow_verdict.py` cannot tell you whether trials reach them. Cite both, separately.
2. **Boundary 0 always.** Every sign test here is at 0, never an arm's own `(th_go+nogo_ref)/2`
   midpoint — the same rule as flow-verdict trap 3, so the two tables compose.
3. **The GNG-stage DPA probe is the `after_gng/dpa` story.** That's the project's key metric; a run
   that passes DUAL/dual but fails GNG/dpa has had its memory rebuilt by Dual, not retained.
4. **A free side is not a passing side.** With `nogo_target=None` and `nolick_weight=0` NOTHING
   imposes don't-lick; the tool tags that `★nogo-UNCONSTRAINED` and declines to score it. Read the
   printed κ₁ value — the network may be licking on every nogo trial.
5. Report per-seed. Seeds in one arm diverge qualitatively here (FLIP in some, DECAY in others).

## Companion tools

`flow_verdict.py` (autonomous wells — the geometric verdict), `bifurcation_probe.py` (g·λ table),
`plot_sweep.py --plots traj` / `traj_flow.py` (the figures this table explains),
`analyze.py` (the accuracies). Spec and history: `docs/analysis.md`, `docs/ring_lowerplane_log.md`.
