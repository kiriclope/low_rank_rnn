# Ring → lower-plane attractors: research log

Working log for the thread on shaping the κ-plane geometry across DPA → GNG → Dual.
Pick up here next session.

## 1. Goal

Get a vanilla rank-2 low-rank RNN to reproduce, structurally, what an advanced EI
model already does:

1. **After DPA** — a ring / bistable memory attractor in **κ₀** (memory axis).
2. **After GNG** — that manifold is *deformed* (deformation is allowed; destruction is not).
3. **After Dual** — input-driven **attractors tuned to the lower κ-plane** (κ₁ < 0),
   on a deformed-or-still-circular manifold, while **DPA memory is preserved**.

Key principle (clarified this session): the lower-plane attractors should **emerge
structurally** for the relevant input conditions — *not* be imposed by an explicit
κ₁ target / regularizer / bias. The autonomous (no-input) dynamics must keep the
bistable DPA memory; only the input-driven conditions should collapse to single
lower-plane attractors.

## 2. Current task setup

- `nonlinearity` ∈ {relu, tanh}, `gain=1.0`, `init_style="random"`, `hidden_size=512`.
- `nogo_target=-1.0`, `cue_on_go_input=True` (cue rides on go channel 4 at `cue_scale=2.0`),
  `cue_on_go_input` ⇒ `input_size=6`.
- **`rwd=False`** (no teacher-forced reward; default flipped this session).
- **Hinge dropped**: `go_hinge_thresh=None` ⇒ go response uses **MSE toward `go_target`**
  (was a squared hinge `relu(thresh-pred)²`). nogo uses MSE toward `nogo_target=-1`.
- New `go_target` RunConfig field (default 1.0); swept {1.5, 2.0} in one test.
- Tasks (`src/tasks.py`): GNG/Dual go target on the **cue window only**
  (`n_on[k]:n_off[k]`); nogo target spans **cue → pre-test** (`n_on:n_on[next]`),
  i.e. a longer window. Memory-period targets are commented out.
- Per-stage epochs 100/100/100, `optimizer="adam"`, `stop_loss=0.1`, `dual_loss="separated"`.

Sweeps live in:
- `results/dual/sweep_relu_new/` — relu, go MSE, 5 seeds.
- `results/dual/sweep_tanh_new/` — tanh, go MSE, 5 seeds.
- `results/dual/sweep_tanh_gotarget/` — tanh, 3 seeds × `go_target` ∈ {1.5, 2.0}.

Accuracy is solved in all of these (after_dual dpa≈gng≈0.99; tanh retains DPA through
GNG much better than relu: after_gng/dpa ≈ 0.96 vs 0.61).

## 3. Key finding — why nogo collapses the memory but go doesn't

**Observation.** Under a clamped input, the tanh expert's fixed points are:
- **Autonomous**: 3 asymmetric attractors (deformed ring), bistable κ₀.
- **Go (ch4 clamped)**: **2 attractors** at κ₀ = ±1.1, κ₁ ≈ +0.85 — *bimodal*
  (the two DPA memory poles, lifted to the go-decision level; DPA memory preserved).
- **NoGo (ch5 clamped)**: **1 attractor** at κ₀ ≈ 0, κ₁ ≈ −2.1 — *unimodal*,
  in the lower plane (memory collapsed).

**Mechanism (reverse-engineered).** The κ₀ memory bistability needs the memory
self-coupling `G₀₀ = (gain/N) Σ_i n₀[i]·φ′_i·m₀[i] > 1` near κ₀=0. Measured on tanh s0:

| condition  | G₀₀ at κ₀=0 | mean φ′ | % units saturated (φ′<0.1) | result   |
|------------|-------------|---------|----------------------------|----------|
| autonomous | 2.58        | 1.00    | 0%                         | ring     |
| go (ch4)   | 1.80        | 0.48    | 22%                        | bimodal  |
| nogo (ch5) | 0.94        | 0.25    | 49%                        | unimodal |

nogo drives κ₁ out to ≈ −2, which (through `m₁·κ₁` in each unit's pre-activation)
pushes ~half the population into tanh saturation (φ′→0). That drops G₀₀ below 1 and
the κ₀ double-well flattens into a single well → single lower-plane attractor.
go only reaches κ₁≈+0.85, saturates 22%, G₀₀ stays 1.8 > 1 → two memory poles survive.

**Structural root.** The nogo input column is both larger and far more aligned with
the decision readout n₁:
- go:   `‖gain·Ai·w₄‖ = 17.1`, `⟨·,n₁⟩ = +3.9`, `⟨·,n₀⟩ = −0.3`
- nogo: `‖gain·Ai·w₅‖ = 21.8`, `⟨·,n₁⟩ = −12.8`, `⟨·,n₀⟩ = +3.9`

nogo's ~3× stronger n₁ projection is what drives κ₁ to −2 and triggers the saturation
collapse. (Diagnostic script is inline in the session transcript — recompute from
`src/models.py` dynamics: φ(gain·(Ai·W·x + M·κ)); J_ac = (gain/N) Σ n_a φ′ m_c.)

## 4. What didn't work

- **Dropping the hinge** (go → MSE toward +1): no change to go geometry —
  still bimodal, go fixed point still κ₁≈+0.85.
- **Raising `go_target` to 1.5 / 2.0**: lifted the go fixed point along κ₁
  (+0.85 → +1.15 → +1.42) but **did not collapse κ₀** — still 2 poles at κ₀=±1 at
  every target. Confirms the bimodality is **structural (input-projection geometry),
  not a drive-strength / target-magnitude issue**. Raising the target shifts the
  operating point but leaves the go *input column* magnitude (~17) ≈ unchanged, so
  saturation never reaches the nogo level.
- Imposing κ₁<0 via a delay-period target / signed κ₁ regularizer / tonic bias was
  considered and **rejected** by the user — that paints the result on instead of
  letting it emerge.

## 5. Next steps

Port nogo's structural collapse to go via the **input-column strength/alignment**,
not the target:
- Increase the effective go input column magnitude and its projection onto n₁ so the
  go condition saturates ~half the units (G₀₀ < 1) like nogo, collapsing κ₀ for that
  condition while the autonomous field keeps the bistable memory.
- Existing scaffolding in `RunConfig`: `project_go_on_n1` (forces go ⟂ memory → keeps
  it bimodal — the *wrong* direction) and `project_gng_orth_n0` (decouples from κ₀ —
  also protective). We want the *opposite*: a go column with a deliberate
  memory-destabilizing component.
- **Open tension**: collapsing go by driving |κ₁| large lands it in the *upper* plane
  (κ₁ > 0, since go is the positive response). For a *lower*-plane go attractor the go
  drive would need a sign-flip or a different coupling. Resolve this before sweeping —
  decide whether "go unimodal" and "go in lower plane" are both required or mutually
  exclusive for the go condition specifically.

## 6. Files touched this session

- `src/tasks.py` — added `go_target` param to `generate_gng_trials` /
  `generate_dual_trials` (replaces hard-coded 1.0 on the go response window).
- `sweep.py` — `RunConfig.go_target` (default 1.0); threaded into both task calls;
  flipped defaults: `gain=1.0`, `init_style="random"`, `go_hinge_thresh=None`,
  `rwd=False`; `make_configs(out_dir, nonlinearity)` + `--nonlinearity` CLI arg.
- `src/dynamics.py` — `flow_specs_for_task` adds a **Cue** panel on channel 4
  (`value=cue_scale`) for the `cue_on_go_input=True` case (GNG + Dual), with go & nogo
  mean trajectories; specs can carry a per-panel clamp `value`; `cue_scale` threaded
  through `plot_task_flow_fields`.
- `plot_sweep.py` — passes `meta.cue_scale` into the flow plotter.

Figures regenerated with the cue panel for tanh s0 only; rerun
`plot_sweep.py --plots flow` to refresh the rest.

---

## 7. Session 2026-06-15 — input strength collapses go; cue strength and τ do not

### 7a. Symmetric go/nogo task
Tasks made **symmetric**: both go and nogo targets applied **after the cue** in the
same window (GNG `n_off[1]:`; Dual a 500 ms post-cue window
`n_off[2] : n_off[2]+(n_off[-1]-n_on[-1])/2`), go=+1 / nogo=−1. Cue windows shortened
(GNG cue 4.0–4.5, Dual cue 6.0–6.5).

### 7b. **WINNER — input strength is the lever (3× cue-off).**
New `input_scale` RunConfig field multiplies **all** stimulus + cue amplitudes
(stimuli `±= input_scale`, cue `±= cue_scale·input_scale`). Swept tanh at 1×/3×/10×,
cue-on (ch4, isz 6) and cue-off (own ch6, isz 7).

| sweep | after_gng/dpa | dual_dpa | go attractors | nogo | autonomous |
|---|---|---|---|---|---|
| tanh 1× cue-on/off | 0.94 | 0.91–0.94 | 2–3 (bimodal) | 1 | 1–4, messy |
| **tanh 3× cue-off** | **0.98** | 0.90 | **1 (unimodal)** | 1 | **2 (bistable)** |
| tanh 3× cue-on | 0.67 | 0.93 | 1 | 1 | 2 |
| tanh 10× cue-on/off | 0.82–0.85 | 0.83–0.87 | 1 | 1 | 2 |

- **3× inputs collapse go to a single attractor (unimodal) in every seed**, while the
  **autonomous field stays bistable** (DPA memory preserved). This is the structural
  goal — achieved by the predicted saturation mechanism (stronger drive → φ′→0 → G₀₀<1
  under the go input → κ₀ collapses *for that condition*; autonomous has no drive, keeps
  G₀₀>1, stays bistable). Attractors: go κ₁≈+1.4 (upper), nogo κ₁≈−1.5 (lower) —
  symmetric across κ₁=0 (natural from symmetric ±1 targets; nogo lower, go upper).
- **Cue routing matters at 3×**: cue-**on**-go badly hurts DPA retention (0.67) — the
  6.0 cue on the *shared* go channel disrupts memory during GNG. Cue on its **own
  channel** → retention 0.98 (best of all). **→ tanh + 3× inputs + cue-off is the
  current best config.**
- **10× over-saturates**: geometry still right (cleaner 2-pole autonomous) but
  dual_dpa drops to 0.83–0.87, cue-off advantage vanishes, seed variance ↑. 3× is the
  sweet spot.

### 7c. Time-scale (τ) does NOT change unimodal/bimodal — proven.
Fixed points solve the **steady-state** equation `r=φ(gain·(W_in x + W_rec r))`, which
has **no τ** (the `exp(-α)` persistence terms cancel at equilibrium). Demonstrated:
go stays at 3 attractors in *identical* locations at τ, τ/3, τ/10. τ only sets
convergence speed and oscillatory stability (the lever for the old *spiraling* issue),
not attractor count. Caveat: for a finite trial, faster τ helps trajectories *reach*
an attractor — relevant to trajectory appearance, not fixed-point count.

### 7d. **NEGATIVE RESULT — cue strength does NOT collapse the bimodality.**
Hypothesis (user): raising cue strength only (cue on go), keeping sample inputs at 1×,
might break the bimodality while sparing DPA encoding. Tested `cue_scale ∈ {6, 12}`,
`input_scale=1`, cue-on, tanh, 5 seeds.
- Accuracy excellent (after_gng/dpa 0.96–0.97, dual_dpa 0.93) — better than 3× cue-on,
  because only the cue (not the sample) is scaled, so memory encoding stays clean.
- **But response-window κ₀ stays bimodal**: A_go/B_go = +1.0/−1.0, A_nogo/B_nogo =
  +0.85/−1.0, at *both* cue=6 and cue=12. Trajectories sit at κ₀≈±1, not κ₀≈0.
- **Why:** the clamped cue field's single attractor (κ₀≈0) is what a *frozen* strong
  cue pulls toward, but when you **train** with a strong cue the network learns weights
  that **resist** that pull and keep κ₀=±1 — it must, because Dual needs the A/B memory
  to survive to the post-test DPA decision. The bimodality is a **task requirement**,
  not a removable artifact; a cue can't break the memory the task forces it to keep.
- **Conclusion:** the operative variable for go-collapse is the **sample/memory input
  drive** (scaling the sample, as in 3×/10×), *not* the cue. The cue is downstream of a
  protected memory.

### 7e. Files touched (2026-06-15)
- `src/tasks.py` — symmetric go/nogo targets (after-cue window); new `input_scale`
  param on all three generators (scales every stimulus pulse + cue); new shared
  `make_timings(dt)` (single source of timings for sweep.py + plot_sweep.py).
- `sweep.py` — `RunConfig.input_scale`; threaded `input_scale` through all task-gen
  and accuracy-helper call sites; `make_configs(out_dir, nonlinearity, cue_on_go_input)`
  + `--cue_on_go_input` CLI arg; timings now from `make_timings`.
- `plot_sweep.py` — `TIMINGS = make_timings(DT)` (was a stale hardcoded copy — fixed the
  trajectory stimulus-window/target misalignment).
- Input/target figure plots true on-amplitude via per-timestep conditional mean
  (`results/figures/task_inputs_targets_symmetric.pdf`).

Sweeps: `sweep_tanh_3x`, `sweep_tanh_3x_cueoff`, `sweep_tanh_10x`,
`sweep_tanh_10x_cueoff`, `sweep_tanh_cue6_12`, plus 1× symmetric `sweep_tanh_cueon` /
`sweep_tanh_cueoff` (all under `results/dual/`).

### 7f. Next steps
- 3× cue-off is the working recipe. To push further: sweep `input_scale ∈ {2,3,4}`
  cue-off to map exactly where go collapses while retention stays ≥0.95.
- The go-collapse lever is **sample-input drive**, not cue or τ. If a *lower-plane go*
  (not just unimodal) is required, that needs an asymmetric manipulation — open.

---

## 8. Session 2026-06-16 — nogo_target=0 rerun; flow fixed-point review

### 8a. nogo_target=0 Dual rerun (from GNG checkpoints)
Retrained **only the Dual stage** from `sweep_tanh_cue6_12` GNG (`naive`) checkpoints
with `nogo_target=0` (was −1) → `sweep_tanh_cue6_12_ng0`.
- **Retention improved**: dual_dpa 1.00 (cs6) / 0.98 (cs12) vs 0.93 at nogo=−1;
  dual_gng=1.0. A milder nogo response (target 0) perturbs the κ₀ memory less.
- **nogo decision moved up to κ₁≈0**: response-window A_nogo κ₁ ≈ −0.15 (was ≈ −0.95).
  So nogo no longer sits in the lower plane — it parks near 0.
- **Bimodality unchanged** (go still 2 attractors; autonomous bistable). Trade-off:
  `nogo_target=0` ⇒ better retention but *no* lower-plane nogo; `nogo_target=−1` ⇒
  lower-plane nogo but slightly worse retention.
- Tooling: `rerun_dual.py` now (i) uses shared `make_timings` (was stale hardcoded
  timings — would have mismatched the symmetric task!), (ii) has a `--nogo_target`
  override, (iii) threads `go_target`/`input_scale` into dual generation + accuracy.

### 8b. Flow fixed-point review — "upper-ring attractors" are a SOFT SLOW RING, not artifacts
User suspected the autonomous upper-ring fixed points were a code artifact. Reviewed
`src/dynamics.py`:
- `make_input(None)` → all-zeros, so the autonomous field is genuinely input-free (no bug).
- Fixed-point **locations** solve the exact steady state `κ = nᵀφ(gain·Mκ)/N` — verified:
  full-simulation autonomous attractors land exactly on the analytic roots (basin counts
  confirm, e.g. cueon/s1 all 4 analytic attractors reproduced by simulation).
- **The real story**: the ring attractors have eigenvalues |λ| ≈ 0.94–0.97 (genuinely
  stable but *slow*), and the saddles ≈ 1.05–1.07 — a "soft" weakly-broken ring. A truly
  marginal ridge point (|λ|=1.0000) exists between attractors but fails `residual_tol=1e-8`
  so is never plotted. So the points are **real shallow attractors**, not artifacts — but
  showing them as sharp attractor dots over-states how sharp the ring is.
- **Two-timescale caveat**: `classify_fixed_points` uses a single-timescale map Jacobian
  `I + β·J_flow` (β=1−e^{−α}); the true dynamics has a separate rec-input timescale α_rec.
  Locations are exact regardless, but the analytic attractor *set* can differ from the
  full sim (in cueon/s0 the sim has a 3rd slow attractor the analytic finder misses).
  For a faithful set use `--use_sim_field --sim_n_warmup N`.

### 8c. Code added (2026-06-16)
- `classify_fixed_points` / `classify_sim_fixed_points`:
  - **`marginal`** class (default `marginal_tol=1e-2`): near-unit eigenvalue with no
    unstable direction → line/ring degeneracy (rendered as a gold square). Catches true
    `|λ|≈1` degeneracies (does NOT fire on the 0.95 soft-ring — those are genuine).
  - **`slow_attractor`** class (optional `slow_tol`, off by default): a *stable* attractor
    whose slowest eigenvalue is within `slow_tol` of the unit circle (`1−max|λ| ≤ slow_tol`)
    → rendered as an **orange ring**. This is the "real fix" for the soft ring — flags the
    shallow ring attractors as ring points without mislabeling saddles.
- `plot_sweep.py`: `--mark_slow` (enables slow_attractor, `--slow_tol` default 0.06),
  and `--n_batch` (overlaid-trajectory trials, default 256). `--n_grid` / `--n_fp_seeds`
  already existed. Default behaviour unchanged (annotations off unless `--mark_slow`).
- `sweep_tanh_cue6_12_ng0` flows regenerated with `--mark_slow --n_grid 261
  --n_fp_seeds 41 --n_batch 512`: the two soft-ring autonomous attractors now show as
  orange slow_attractor rings, confirming the fix.

### 8d. The slow attractors sit on a SLOW RING (manifold), not isolated wells
Follow-up to 8b, prompted by the observation that the field shows a near-zero-velocity
*arc*. Quantified on `sweep_tanh_cue6_12_ng0/s0_cs6` (autonomous):
- **Seed-robust**: FP-finder returns identical locations across `n_seeds=21/41/61` and
  random grid jitter → the discrete attractors are *not* seed artifacts.
- **But there is a real slow ring**: tracing the min-|F| ridge by angle, the **radial
  (across-ring) velocity is ~10× smaller than the tangential (along-ring)** everywhere
  (across ≈ 0.01 vs along ≈ 0.07–0.33). So the arc is a *radially attracting manifold* —
  points on it are trapped against leaving, exactly the near-zero-velocity arc seen in
  the field.
- **It's a slow manifold with directed drift, not a flat continuum**: tangential |F|
  median ≈ 0.12 (barriers ≈ 0.35 near saddles), so a directed slow flow toward the 2
  discrete wells. Per-step drift ≈ β·|F| ≈ 0.07·0.12 ≈ **0.008 κ/step**; over a ~444-step
  trial that integrates to a large arc, so trajectories do reach the wells — but slowly,
  and on shorter timescales any ring point behaves as a **quasi-attractor**.
- **Reconciliation**: asymptotically → 2 discrete robust attractors; on trial timescales
  → a radially-trapped slow ring where points drift only ~0.008/step. "Any point could be
  a slow attractor" is correct in the finite-time sense; it is *not* a true line attractor
  (tangential flow is directed, with barriers). The discrete dots are real fixed points,
  not artifacts — but they **under-represent** the functional object (the slow ring).

**What sets the tangential drift:** the **gain × anisotropy of the trained rank-2
structure**. An isotropic memory mode (equal effective gains along the ring, no κ₀–κ₁
coupling) → true line attractor, zero drift. GNG/Dual training makes it anisotropic
(λ₀≠λ₁ + off-diagonal m–n coupling), tilting the ring into a shallow landscape; drift ≈
the angular gradient of that tilt. More isotropic ⇒ flatter/slower ring; more anisotropic
⇒ faster drift to discrete wells.

### 8e. Slow-manifold overlay (code, 2026-06-16)
- `src/dynamics.py`: `trace_slow_manifold(model, ff_input, xlim, ylim, vel_thresh=0.12)`
  — per-angle min-|F| ridge tracer; returns ridge pts + |F| + tangential speed.
  `plot_task_flow_fields` gained `show_slow_manifold` / `slow_manifold_thresh`: draws the
  ridge as dots colored by |F| (spring colormap, magenta=slowest→yellow≈threshold).
- `plot_sweep.py`: `--show_slow_manifold` + `--slow_manifold_thresh` (default 0.12).
  Opt-in; defaults unchanged.
- Reference figure: `/tmp/fp_expert_slowmanifold.pdf` (regenerate via `plot_sweep.py
  --plots flow --run_ids s0_cs6 --auto_xlim --mark_slow --show_slow_manifold`). The slow
  ridge runs through both orange slow_attractor rings; magenta (slowest) points cluster at
  the wells.

---

## 9. Session 2026-06-16 (cont.) — symmetry-breaking attempts and the EI model

### 9a. Why the memory ring is symmetry-locked to the origin (tanh)
The autonomous κ-field `F(κ)=(1/N)nᵀφ(gain·Mκ)−κ` is **odd** for tanh (zero-bias-effective:
the trained input bias is large in unit-space, ‖d‖≈14, but projects ~1% onto the readout
n, net κ-pull `Ψ(0)≈0`). Odd ⇒ fixed points come in **origin-symmetric ± pairs** ⇒ the
memory ring is centered at κ=0 and its attractors **straddle κ₁=0** (one upper, one lower).
So you can't get them *all* lower without breaking the symmetry. **Correction to an earlier
framing:** the memory attractors are NOT at κ₁≈0 ("no decision"); they sit out on the ring
at κ₀≈±1, κ₁≈±0.5 — it's the ring *center* that's pinned, not the attractors.

### 9b. Generic symmetry-breaking ≠ directional lowering (three null results)
To get lower-plane attractors *without imposing* a −n₁ drive, tried breaking the symmetry
generically and letting training place the ring:
- **`unit_bias`** (new): trainable per-unit bias inside φ, free in all stages, random init
  σ=0.2. Confirmed active (‖trained‖≈13, zeroing it changes the attractor set: e.g. s2 3→1),
  but training keeps it **⊥ to the readout** (⟨ub,n₁⟩≈−0.3 vs ‖13‖). Breaks ± symmetry
  (3–4 attractors) but mean κ₁ ≈ 0 — **not lowered**. dual_dpa even ↑ to 0.97.
- **`unit_bias` + `nogo_target=0`** (Dual rerun): dual_dpa **1.00** (best retention seen),
  but autonomous still centered; nogo input-driven attractor stays lower (κ₁≈−1.4).
- **`tanh_asym`** φ=tanh+γ·tanh² (new nonlinearity, γ-knob): the γ·tanh² **even** component
  is non-removable (tied to m,n, not a bias the net can rotate away). Sweep γ∈{0.3,0.6}:
  γ=0.3 keeps retention 0.96 (γ=0.6 → 0.80, over-deforms); **spiraling near zero**
  (frac complex eig 0–4% vs relu 15%). Breaks ± symmetry (odd #attractors=3) but autonomous
  **mean κ₁ ≈ 0** — again broken, not lowered.

**Lesson (robust across all three):** breaking the symmetry is necessary but **not
sufficient** — without something that *prefers* "lower", training de-centers the ring in a
random/zero-mean direction. Lower-plane attractors need a **directional** ingredient
(imposed −n₁ drive, or a structural one), not just broken symmetry.

### 9c. Why relu spirals (and tanh never does)
Measured fraction of the κ-plane with **complex** (spiraling) Jacobian eigenvalues:
**relu 15%, tanh 0%**. Mechanism: spiral ⇔ antisymmetric cross-coupling `J₀₁·J₁₀<0`
dominates. tanh's `φ′=1−tanh²` shrinks as units saturate → radial damping kills rotation
everywhere (that's why the old `gain=2` fix worked). relu's `φ′∈{0,1}` never decays for
active units → no radial damping; the hard on/off switching creates an undamped rotational
component. `tanh_asym` saturates too → keeps ~0% spiraling while being asymmetric.

### 9d. EI model — `EILowRankModel` (v1)
New class (`src/models.py`): 512 E / 128 I, **Dale-signed frozen static backbone** (balanced,
spectral radius `static_radius`=1.5), **weak trained low-rank on E→E** with the **total** E→E
weight rectified ≥0 (`relu(W_static_EE + m@nᵀ/n_exc)`, Option-2 clipping — free signed m,n so
signed κ survives), **relu rates**, **inputs to E only**, κ readout = `rates_E·n/n_exc`.
Wired into `sweep.py` (`model_type="ei"`, `n_inh`, `static_radius`, `low_rank_scale`) and
`plot_sweep.py` (`_build_model` branch). Interface matches LowRankModel; W_static is a
persistent buffer (saved/reloaded). `rerun_dual.py` also EI-aware.

**`sweep_ei_v1`** (5 seeds, relu, radius 1.5, low_rank_scale 0.3): trains end-to-end
(DPA 0.998, GNG 1.0, dual_dpa 0.92, dual_gng 0.99) but **after_gng/dpa = 0.63** (poor, relu-like).
Geometry: input-driven attractors split (nogo lower, go/cue upper) but the **autonomous field
collapses to a single origin attractor — no persistent memory ring**.
**Ground-truthed**: in a real DPA delay κ₀ *drifts and degrades* (A: 0.38→1.64, B: −0.4→+0.5;
A/B separation 0.78→0.42 over 5 s). So the memory is a **slow drifting transient**, not an
attractor — survives only because the delay is finite. This is the smoking gun for the missing
ingredient: **short-term plasticity** (advanced model has it; v1 doesn't).

### 9e. EI flow tooling (NeuroFlame binned method)
The analytic/sim κ-tools assume the low-rank `h≈Mκ` embedding, which the 640-dim EI state
lacks. Implemented the NeuroFlame approach (ref `~/models/NeuroFlame/org/train/dual/
flow_dual_alt.org`): **on-manifold grid init** by injecting a current `X·n₀+Y·n₁` along the
readout vectors for `set_w` steps, then release → run full EI sim → **binned drift field**
(`histogram2d` mean displacement per κ-bin, mask low-count, Gaussian-smooth) → KMeans fixed
points. Reusable script **`ei_flow.py`**:
`python ei_flow.py --sweep_dir results/dual/<sweep> --out_root results/figures --device cuda:0`.
Produces per-seed, per-stage figures matching the vanilla layout:
`results/figures/<sweep>/individual/<run_id>/flow/fp_{dpa,naive,expert}.pdf` (panels mirror
`flow_specs_for_task`). Generated for all of sweep_ei_v1 (15 figures).

### 9f. Code added (2026-06-16 cont.)
- `src/models.py`: `unit_bias` (LowRankModel), `tanh_asym` + `nl_gamma`, `EILowRankModel`.
- `src/dynamics.py`: `unit_bias` threaded into `low_rank_field_np`/jacobians; `tanh_asym`
  φ/φ′ branch.
- `sweep.py` / `plot_sweep.py` / `rerun_dual.py`: `use_unit_bias`/`unit_bias_*`, `nl_gamma`,
  `model_type`/`n_inh`/`static_radius`/`low_rank_scale` config + model-build branches.
- `ei_flow.py`: standalone EI binned-flow figure generator.

### 9g. Next steps
- **STP (v2)** on E→E is the indicated ingredient — should turn the EI drifting transient
  into a persistent memory attractor and fix retention. Then re-check the lower-plane geometry.
- Best vanilla retention recipes remain: tanh 3× cue-off (dual_dpa 0.90, after_gng 0.98) and
  unit_bias+nogo0 (dual_dpa 1.00). `tanh_asym` γ=0.3 = clean spiral-free asymmetric tanh if
  a directional lever is added later.

---

## 10. Session 2026-06-17 — theory note + static backbone (tanh + W_fixed)

### 10a. Theory write-up
`docs/theory_landscape.md` (+ PDF via `./make_pdf.sh`): derives the κ-plane potential
`V(κ)=½‖κ‖²−(1/gN)Σ log cosh(g(b+Mκ)_i)` (exact for n=M), why **tanh odd ⇒ F odd ⇒ ± pairs
⇒ even V ⇒ symmetric wells centred at origin**, the supercritical-pitchfork bistability
`gλ₀>1`, the spiraling criterion (relu 15% vs tanh 0% complex eig), the slow manifold
(near-critical `gλ₀≈1`), and per-stage landscape evolution. The analytic backbone for §9.

### 10b. tanh + static connectivity backbone — `sweep_tanh_static`
`use_fixed_weights=True, fixed_weight_scale=1.0, fixed_weight_orthogonalize=False` (backbone
*shapes* the κ-plane), tanh, cue-on-go, 5 seeds.
- **Best retention so far:** after_dpa 0.999, **after_gng/dpa 0.89**, **dual_dpa 1.00**,
  dual_gng 1.00. The static backbone *stabilises* the memory.
- **Geometry reshaped but symmetry intact:** 4 autonomous attractors pulled inward
  (κ₀≈±0.8 vs ±1.2 plain tanh), but **exact ± pairs** straddling κ₁=0 (mean κ₁=0, ±-asymmetry
  =0). **Not lowered.**
- **Why (theory-consistent):** a *linear* backbone composed with an *odd* nonlinearity keeps
  the autonomous field odd (`W_fixed(−r)=−W_fixed r`, `tanh` odd ⇒ `F(−κ)=−F(κ)`). It reshapes
  V (well positions/count) and helps retention, but **cannot break the origin symmetry**.
  ⇒ static backbone = a *retention* ingredient, not a *lowering* one. Lowering still needs an
  even-in-φ term, non-negative (EI) rates, or an imposed directional drive.

### 10c. Tooling fixes (2026-06-17)
- `LowRankModel.w_fixed` is now a **persistent** buffer (was `persistent=False` → not saved →
  non-orthogonalised backbones were irreproducible for analysis). Old ckpts still load (strict=False).
- `ei_flow.py` **generalised** to any model (EI *and* LowRankModel incl. fixed-weight backbone):
  generic recurrent (`make_hidden_fn`), nonlinearity, per-unit-bias; readout/grid widths from the
  model. Validated it reproduces the plain-tanh symmetric ring. **Use it for any sweep whose
  backbone breaks the analytic κ-reduction** (plot_sweep's analytic flow ignores `W_fixed`).
- `plot_sweep._build_model` / RunMeta now rebuild the fixed-weight backbone (so traj/scatter are
  correct); flows for such sweeps must come from `ei_flow.py`, not plot_sweep `--plots flow`.
- `make_pdf.sh` (repo root): math-md → PDF via pandoc MathML + headless Chrome (no LaTeX engine here).

## 11. EISTP model — persistent memory + lower-plane wells (2026-06-22) ★ BREAKTHROUGH

After the vanilla / static-backbone family confirmed that an *odd* φ + *linear* recurrence is
symmetry-locked (no lowering), we ported the **NeuroFlame dual-EI network** (the model that
consistently pushes the wells) to a minimal self-contained class **`EISTPModel`** (`src/models.py`).
See `docs/architecture.md` for the full mechanism.

**Ingredients (all essential, faithful to NeuroFlame `conf/train_dual_EI.yml`):**
- 2-pop EI, N=2000 (1500E/500I) — runs use **N=1000/K=125** for speed (K scaled with N to hold
  connection prob `K/N=0.125`); sparse binary connectivity `C` (in-degree K); Dale block strengths
  `Jab` balanced 1/√K; **relu** rates.
- **Two timescales**: synaptic filter (`τ_syn`) then rate filter (`τ`).
- **Markram STP on E→E** (USE=0.05, τ_fac, τ_rec): output `u·x·r`; gate sweeps USE(rest)→~1(facilitated).
- **Trained rank-2 low-rank `m,n` on E that MULTIPLICATIVELY modulates the STP E→E weight**:
  `W_EE = gain·j_stp·(C/√K)·(1 + n@mᵀ/lr_scale)`, clamped ≥0 (Dale). The memory mode *rides on* the
  facilitating synapses — not an additive perturbation to a backbone (that was the v1/v2 EI failure).
  `n` = output/readout direction, `m` = presynaptic selection; readout κ = rates_E·n/N_E.
- Driven by the **vanilla `src/tasks` generators** (DPA/GNG/Dual, input_size=6) with **hinge** targets
  (`dpa_hinge_thresh=1`, `go_hinge_thresh=1`); j_stp fixed at 1.0 (matches the notebook, where J_STP is
  an nn.Parameter but frozen).

### 11a. The decisive scaling — `lr_scale` ("C is K/N on average")
The low-rank `n@mᵀ` is multiplied **element-wise by C**, whose mean entry is `⟨C⟩=K/N`, so the
memory-mode gain is
```
g_mem = gain·j_stp · (1/√K)[balance] · (K/N)[⟨C⟩] · (N⟨mn⟩)[mᵀn] / lr_scale  =  √K·⟨mn⟩ / lr_scale
```
The K/N density of C cancels the N of the overlap, leaving **√K**:
- `lr_scale='N'` (NeuroFlame `train_scale='all'`): g_mem = √K/N ≈ **0.015 → DEAD**. DPA stuck at chance
  (0.50) regardless of epochs (60 & 100). κ1 encodes the sample then **decays to 0** over the delay.
- `lr_scale='sqrtK'` (NeuroFlame `train_scale='sparse'`): g_mem = ⟨mn⟩ = lr_ini² ≈ **1.0 → CRITICAL**;
  STP gate then sweeps it ×USE(0.05)→×1. **★ κ1 HOLDS FLAT at ±10 through the whole delay — a genuine
  persistent working-memory attractor (first time in the project).** κ2 fires the correct ±decision.

### 11b. Result (`sweep_eistp_sqrtK`, N=1000, 5 seeds, 100 epochs, lr=0.1)
- **3/5 seeds perfect**: after_dpa/dpa **1.0**, after_gng/dpa **0.92**, dual_dpa **1.0**.
  2/5 diverged to NaN (strong /√K + STP → supercritical runaway). **STABILIZED** in
  `sweep_eistp_sqrtK_stab` → clean **5/5** (DPA 1.0, dual_dpa 1.0, after_gng/dpa 0.81) via:
  rate cap `eistp_r_max=200` (relu has no upper bound → runaway; cap is 5× the ~43 operating peak,
  science untouched), NaN-skip in `Optimization`, and gentler lr 0.05 / grad_clip 0.5.
- **κ-plane flows:** autonomous = **bistable memory** (κ0≈±8); **Go → upper plane** (κ1≈+6, often a
  line attractor); **NoGo → lower plane** (κ1≈−3.5). Reproduces BOTH the persistent memory and the
  **lower-plane decision wells** the project was chasing.

### 11c. Tooling (2026-06-22)
- `EISTPModel` wired into `sweep.py` (`model_type="eistp"`; fields `n_neuron`, `eistp_K`, `j_stp`,
  `eistp_lr_scale`∈{"N","sqrtK"}), `plot_sweep.py` (build + RunMeta), `ei_flow.py`.
- **`ei_flow.py` eistp flow path** (`_run_grid_eistp`): drives `model.update_dynamics` (exact
  two-timescale + Markram STP); grid is **auto-calibrated** — inject along *unit* orthogonalised
  readout vectors, probe each axis χ=κ_held/S_probe, set S=R/χ (capped) so the held state spans κ≈±R
  in 2-D without overdriving the near-critical net (`bscale·n` with trained ‖n‖≈100 blows up to 1e30).
  make_stage_figure auto-uses R=15, T=600, fp tolerances ×10 for eistp.
- **`--style` flag** (default **`magma`** = vanilla look: magma speed + white streams; `binned` =
  original coolwarm z-score). Generic flow path (vanilla/static/EI) untouched & verified unchanged.
- NOTE: an earlier "input noise scale is too large" flag was **wrong** — noise scale is O(√K), SNR≈1
  matches the working vanilla model; only structure differs (rank-input_size vs per-neuron).

### 11d. Stabilisation + ablations (2026-06-22/23)
- **Why it diverges:** relu (no upper bound) + supercritical-facilitated E→E loop → forward rates
  → ∞. Root-cause fix = **rate cap** `eistp_r_max` (~6× the ~80 operating peak; only catches
  runaway). Plus **NaN-skip** in `Optimization` and **graceful epoch divergence** (`_run_epoch`
  returns nan on an all-non-finite epoch → `fit()` keeps best state, no "zero examples" crash).
- **Random init works** (`eistp_lr_ueqv=False`, m,n independent → init overlap≈0.01, g_mem≈0):
  training builds the correlation up; DPA still 1.0, same persistent memory + wells. The m=n init
  was convenient, not necessary.
- **Grad clipping matters**: without it ~1–2 seeds destabilise in the **Dual** stage; with
  `grad_clip=1.0` all runs complete (instability now isolated to Dual, handled gracefully).
- **NoGo-well knob** (`--nogo_target`): nogo=−1 → NoGo well firmly lower-plane (κ1≈−3.5);
  nogo=0 → NoGo well at the κ1≈0 midline (and more stable, dual_dpa 0.987). The well depth is set
  by the target value — a controllable feature, not an accident.
- Reference runs: `sweep_eistp_sqrtK_stab` (matched init, clip 0.5) and `sweep_eistp_rand_clip_ng0`
  (random init, nogo=0) — both clean 5/5.

**STATUS: the project goal (persistent memory + lower-plane wells) is achieved and robust.** Open
follow-ups: confirm at faithful N=2000/K=250; reduce the residual Dual-stage instability further if
desired (lower lr / lr_ini<1).

### 11e. Notebook re-read + scaling/regime reconciliation (2026-06-30)
Went back to the original NeuroFlame notebook (`org/train/dual/train_dual.org`) and compared the
optimization + inputs against our port.

- **`j_stp=5 + lr=0.01` is the best DPA-through-GNG retention yet** (`sweep_eistp_jstp5_lr01`):
  clean 5/5, after_dpa/dpa 0.997, **after_gng/dpa 0.93** (vs 0.81/0.84 refs), dual_dpa 1.0. The 5×
  recurrent gain doesn't destabilise (rate cap + clip + gentler lr hold it).
- **★ The "÷N is dead" finding was a *regime* artifact, not a scaling barrier.** The notebook uses
  `TRAIN_SCALE='all'` (÷N_E, = our `lr_scale="N"`) — the setting we'd called dead. Reproduced it
  *working* in our port by matching the notebook regime — `lr=0.1`, **no grad clip**, `j_stp=1` —
  in `sweep_eistp_ablate_all`: DPA **1.0**, after_gng/dpa 0.91, dual_dpa 0.999, 5/5. Mechanism:
  g_mem = √K·⟨mn⟩/lr_scale; with ÷N_E g_mem≈0.015 at init, but with lr=0.1 + no clip the optimizer
  grows ‖m,n‖ ~10× to compensate. Our earlier lr≤0.05 + grad-clip throttled that growth → stayed
  dead. **`'all'` (÷N_E) and `'sqrtK'` (÷√K) are two routes to the same fixed point.**
- **Notebook optimization (for the record):** `Adam(lr=0.1)` rebuilt per stage, `ExponentialLR
  gamma=0.9`, **no grad clip**, **only ~10/10/5 epochs** (DPA/GNG/Dual), `zero_grad` freezing
  (GNG freezes rank-0 of U&V), early stop at loss<0.15, NaN→stop. Loss = `DualLoss`: with the
  shipped `bce_alpha=1.0` the BCE/sigmoid branch of `SignBCELoss` is fully zeroed, leaving a **pure
  hinge** `relu(thresh − sign(2t−1)·readout)` (thresh=1.0, same family as ours) + a `0.1·|overlap|`
  suppression on class-0 + a `SmoothL1` term pinning the memory readout to 0 before the sample.
  (Earlier notes said "sigmoid+BCE" — that branch is dead code at α=1.) The one loss ingredient we
  lack is that explicit pre-stim zero-memory `SmoothL1`.
- **Notebook inputs are a *frozen* dataset:** fixed random `odors=randn(10,N_E)` patterns, `ff_input`
  built **once** over 4 signal conditions (sample±×test±) with VAR_FF noise baked in, reused every
  epoch; only the init recurrent kick is resampled. Our feedforward (generator noise in `X`) is
  already frozen per stage; only our init kick was resampled.
- **Frozen-input ablation** (`eistp_init_noise=0` → fully deterministic forward; `sweep_eistp_frozen`):
  ~identical accuracy (DPA 1.0, dual_dpa 0.999, after_gng/dpa 0.87), **no convergence speedup**
  (~150 DPA epochs, not 10), no generalization loss (eval on fresh trials still 1.0). **Conclusion:
  a frozen dataset is NOT the lever behind the notebook's 10-epoch convergence** — the remaining
  suspects are the loss's pre-stim zero-memory `SmoothL1` term, the optimizer recipe (Adam wd=0,
  lr=0.1, ExpLR), and the ring/cosine task — not the data-freezing.
- **Tooling:** `plot_sweep.py` now auto-routes eistp away from the analytic FP scatter/flow (which
  crash on `EISTPModel` — no `.alpha`) to the simulation path, so a plain run yields the full figure
  set. New `eistp_init_noise` RunConfig field.

---

## 12. Session 2026-07-03 — vanilla push-down: the U/half-ring, and the two-ingredient isolation fix

Back to the **vanilla rank-2** thread with a sharpened goal: get the two **A/B sample-memory wells**
(on the memory axis) into the **no-lick region** (κ₁ < 0), which even the EISTP model doesn't do
(it lowers only the *decision* wells, not the memory). Axis convention in the flow figures:
**κ₀ = horizontal = memory, κ₁ = vertical = decision/lick** (top = go/lick, bottom = nogo/no-lick).

### 12a. The "no-lick" mechanism (directional pressure × realizing symmetry-breaker)
theory_landscape.md §8 proves the deadlock: with an *odd* φ and `unit_bias=0` the autonomous field
`F(κ)=(1/N)nᵀφ(gain·Mκ+ub)−κ` is **odd regardless of the loss** ⇒ wells are ± pairs centred at the
origin; both-into-κ₁<0 is symmetry-forbidden. Past failures split into two halves that each fail
alone: symmetry-breakers (unit_bias, tanh_asym) tried with **no directional pressure** → de-centre
randomly (⊥n₁); a κ₁ target tried with **no symmetry-breaker** → odd φ has no DOF to satisfy it.
**The untried combination:** a *behavioural, one-sided* **no-lick hinge** `relu(κ₁)²` on the decision
channel over the currently-**free** delay/memory windows (penalise lick only, leave κ₁<0 free — not a
painted κ₁ value) **×** an enabled symmetry-breaker (`tanh_asym`/`unit_bias`) to realise it downward.
- **Code (`nolick_weight`, default 0.0 ⇒ byte-identical):** one-sided term added to `MaskedGNGLoss`
  and `MaskedMultiTargetDualLoss` (`src/train.py`). The free windows are **exactly**
  `~torch.isfinite(target_dec)` (the loss zeroes preds on NaN targets), so `nolick_mask = ~finite`
  — no timing math, no task-generator edits. Uses **raw** `pred` (not `safe_pred`, which is 0 there).
  Threaded through both loss call sites + `RunConfig.nolick_weight` (`sweep.py`).
- **Weight-decay caveat:** AdamW decays `unit_bias`→0 (guts the asymmetry) → use `optimizer="adam"`.

### 12b. Sweep 1 `sweep_nolick_lower` — and why it was a mis-read
Vanilla tanh, structured init, gain 2, nogo=0, adam, 100/100/100; arms control / pressure(nw=0.5) /
unitbias / tanhasym, 3 seeds. **Retention unharmed** (every arm after_dual dpa 0.997–1.000; transient
after_gng dips 0.75–0.82 all heal). Measured DPA-delay held-κ₁ appeared to lower modestly
(control −0.15 → tanhasym −0.31). **This differential was NOISE on a shared transformation** —
see 12c.

### 12c. ★ THE ACTUAL STRUCTURE (Leon's correction): 4-well ring → 270° U/half-ring
All arms produce the **same** solution. **DPA:** four attractors on the cardinals — memory poles at
(±1.15, ≈0) and decision poles at (0, ±1.3) — joined by saddles/slow lines that mimic a **ring**.
**Dual:** the **top (go) well opens up**, leaving a **U / half-ring** (~270°) running
left → bottom → right — i.e. the memory poles are now connected *underneath*, through the deep nogo
pole. This is **invariant** to `nolick_weight` and the symmetry-breaker. Mechanistic reason: the go
response must *drive* κ₁ up out of the top, so Dual destabilises the top into an input-driven state
while nogo/rest stays at the bottom → the autonomous manifold keeps the lower ¾ and opens the top.
The plain go/nogo dual task already does this; the no-lick pressure was redundant for producing it.
**Lesson: don't read point-well κ₁ differences; read the manifold topology.**

### 12d. Refined target (Leon): TWO ISOLATED wells at κ₁<0 — *not* a U
The 270° U is undesirable because it is "full no-go": one continuous slow manifold fusing both
memory poles with the deep nogo pole. Target = **two disconnected A/B memory wells, both at κ₁<0,
with no connecting arc** (the bottom must not be part of the memory manifold).

### 12e. ★ Validated fix: the U survives because the DECISION mode is autonomously bistable
The ring exists because **both** axes are autonomously bistable — memory (g·λ₀) *and* decision
(g·λ₁) — and near-criticality links all four wells. Trained nets grow **g·λ₁ ≈ 3.3–3.8** (way
supercritical) → strong autonomous go/nogo wells → ring. **Validated analytically**
(`scratchpad/test_subcritical.py`: scale the decision columns m₁,n₁ down on a trained model and
recompute autonomous fixed points):

| dec_scale (g·λ₁) | autonomous attractors |
|---|---|
| 1.0 (3.3) | 2–3 wells incl. off-axis/lower — the ring/U |
| 0.4 (1.3) | **2 clean memory wells at (±1.15, ≈0)** |
| 0.0 (0.0) | 2 wells on the κ₀ axis |

So the fix is **two orthogonal ingredients**: **(1) ISOLATE** — hold g·λ₁ ≈ 1 (no autonomous
decision bistability) → ring collapses to two discrete memory wells; **(2) LOWER** — the directional
break (`tanh_asym` γ=0.3 + `nolick`) pushes those two wells to κ₁<0. Sweep 1 had only (2), which is
why the ring always survived (decision left supercritical in every arm). **Isolation alone lands the
wells at κ₁≈0 (boundary) — both ingredients are needed for κ₁<0.**
- **Levers:** `decision_lambda` (structured-init decision self-gain; ↓0.5→0.25 = subcritical start)
  + `kappa1_reg_weight` (Dual penalty `w·relu(gain·n₁ᵀm₁/N − 1)²`, `sweep.py:686`). `memory_lambda=0.8`
  stays supercritical (deep A/B). The decision mode regrows in DPA/GNG (no reg there) → reduced init
  and Dual reg may both be needed; the k0 arm tests whether reduced-init alone holds.

### 12f. Sweep 2 `sweep_isolate_low` — RUNNING (results pending)
tanh_asym γ=0.3, `decision_lambda=0.25`, `memory_lambda=0.8`, gain 2, nolick=0.5, nogo=0, adam,
100/100/100; arms **`kappa1_reg_weight ∈ {0(k0), 0.3(k03), 1(k1), 3(k3)}`**, 3 seeds.
**Read-out when done:** per-run **g·λ₁** (=gain·n₁ᵀm₁/N, want ≈1), autonomous **fixed-point count**
(want exactly 2, no bottom arc) + their **κ₁** (want <0), and retention. Tools: `sim_ab_wells.py`,
`test_subcritical.py`, `kappa1_extract.py` (all in the session scratchpad).

### 12g. Process gotchas (cost real time this session)
- **The `plotting` subagents STALL:** they background `plot_sweep.py` then idle forever waiting for a
  notification that never fires (one even relaunched the process as it was killed). **Run
  `plot_sweep.py` yourself** as a single background job. Also: multi-line `run_in_background` bash
  gets its newlines flattened → keep the command on one logical line (`&&`-joined, inline env vars).
- **`find_all_fixed_points` returns a `(fps, residuals)` TUPLE** — unpack it. And its finder set is
  incomplete/asymmetric → don't average its κ₁; measure the held state by simulation
  (`sim_ab_wells.py`) instead.
- All Session-12 code + docs are **UNCOMMITTED**.

## 13. Session 2026-07-05 — ★ ISOLATION ACHIEVED: the two-ingredient fix, confirmed

> **⚠ SUPERSEDED by §15 (2026-07-21):** the isolation attributed here to `kappa1_reg` (ingredient 1) and
> the lowering to `nolick`/`tanh_asym` (ingredient 2) is **not** what does it. Ablation shows the
> **attention input** isolates the wells (kills the ring by destabilising the lick attractor); `kappa1_reg`
> and `nolick` can both be 0 and the geometry holds. Read §15 for the corrected mechanism.

§12e's prediction was **validated in a live network**: the ring opens into two isolated low
wells *iff* the decision self-gain g·λ₁ is driven to ≈1 (ISOLATE) on top of the directional
lowering (LOWER). Everything else this session only did (2) and left the ring intact.

### 13a. What only *lowered* (ingredient 2 alone — ring always survived)
Each pushed both A/B memory wells to κ₁<0 but left g·λ₁≈3.3–3.9 supercritical ⇒ ring/U persists:
- **`sweep_curriculum`** (DPA→GNG→Dual-paired→Dual): held-κ₁ = **−0.44** (best lowering of any
  no-clamp lever), 3/3 both wells <0 — but ~3.3 wells (ring).
- **`sweep_recscale`** (trainable per-mode recurrent scale `rec_scale`, decouples recurrence from
  readout): given a free knob the net **grew s₁ to ≈1.4** (wants a *stronger* decision) → g·λ₁≈3.7,
  ring. Reveals the decision's supercriticality is the net's *preferred* solution, not an artifact.
- **`sweep_slowtau`** (τ×{1,2,4}, "hold the decision by slow transient"): **falsified** — g·λ₁ *grew*
  with τ (3.27→3.79) and nogo degraded; slowing τ doesn't subcriticalize.
- **`sweep_fasttau`** (τ×{1,½,⅓}, plain tanh + attention): both wells κ₁<0 (attention breaks the odd
  symmetry, replacing `tanh_asym`), but g·λ₁≈3.5, ~3 wells (ring). τ<0.3 also breaks *optimization*
  (τ=0.10 stalls DPA, Dual never converges — ~4 steps/τ). **τ=0.3 (fast1) is the sweet spot base.**

### 13b. ★ `sweep_kappa1reg` — the winner (fast1 base + Dual `kappa1_reg_weight`)
Base = fast1 (τ=0.3, plain **tanh** + `attention_input`, `nolick=0.5`, `hinge_gng=True`,
`decision_lambda=0.25`, `memory_lambda=0.8`, gain 2, hinges all 3 stages). Sweep `w =
kappa1_reg_weight` (Dual penalty `w·relu(gain·n₁ᵀm₁/N − 1)²`), 3 seeds:

| w | g·λ₁ | #wells | mem-well κ₁ | match/nonmatch | go | nogo |
|---|---|---|---|---|---|---|
| 0 | 3.47 | 3 (ring) | −0.1 | 0.97 | 1.0 | 0.81 |
| **1** | **1.01** | **2 (isolated)** | **−0.9** | **1.0** | **1.0** | **0.79** |
| 3 | 1.01 | 2 | −0.7 | 1.0 | 1.0 | 0.43 |
| 6 | 1.01 | 2 | −0.5 | 1.0 | 1.0 | 0.15 |

**w=1 is the operating point:** g·λ₁ pinned to 1.0 → decision's autonomous wells vanish → the ring
collapses to **two isolated wells at (±1, −0.9), deep in the no-lick plane**, with DPA=1.0,
match/nonmatch=1.0, go=1.0, nogo=0.79. It's also the **best-converging** Dual arm (0.11 ≈ the nolick
floor): at criticality the decision is a line attractor, so the symmetric ±1 match/nonmatch is
trivial to hold. **Why it works where §12 clamps didn't:** the penalty targets g·λ₁ directly (the
bifurcation parameter), and the asymmetric hinge keeps go/nogo functional (input-driven) as the
decision recurrence weakens.

### 13c. The nogo tradeoff (the only cost)
g·λ₁ is already pinned at 1.0 by w=1, so more penalty buys **no more isolation** — it only
over-suppresses: nogo falls 0.81→0.79→0.43→0.15 and the wells shallow (−0.9→−0.5) as w grows. So the
knee is at the low end; the unmapped region is **w∈{0.5,1,1.5,2}** (the "finer nogo tradeoff" sweep)
to find the min penalty with max-retained nogo. nogo softens because at g·λ₁=1 the decision is a line
attractor (holds the symmetric match/nonmatch, but the one-sided nogo drifts).

### 13d. Robust to input noise
The isolation is a property of the *deterministic* field, but it also survives the **noise-averaged**
field `E_x[Ψ(κ)]` (`plot_sweep --field_input_noise`): reg1's wells shift ~0.1 toward the origin
((−1.0,−0.84),(0.83,−0.78)) but stay two, isolated, κ₁<0. A *single* noise draw is NOT enough — one
draw is a correlated input bias through Wi that tilts the field and drops a well ~half the time; K≥8
draws is stable (default 16).

### 13e. Loss cleanups this session (see `docs/running.md` "Isolation recipe")
`hinge_gng` is now the single switch: **True → hinges all 3 stages, False → pure MSE**. Under True:
go/nogo asymmetric one-sided (go≥`go_hinge_thresh`; nogo≤−1 in the GNG memory delay / ≤0 after cue),
match/nonmatch **symmetric ±`dpa_hinge_thresh`** (same shape in DPA `ThresholdLoss` and Dual). `nolick`
is a separate one-sided `relu(κ₁)²` over free windows, **excluding the sample** — its ~0.13 floor
(the go lick-ramp just before the response window) is why Dual never hits `stop_loss=0.1`. `_dual_accuracy`
now scores per-side (go/nogo) against the target in the after-cue window (`dual_go`/`dual_nogo`).

## 14. Session 2026-07-13 — self-gains are task-locked; strong-memory wins; analysis tooling

Systematic follow-up to §13. Question: reach the **two isolated low wells** robustly, *without* the
fragile isolation of §13. Ran ~10 sweeps (all `docs/experiment_log.md`, 2026-07-13). Findings:

### 14a. The self-gains are TASK-LOCKED (the unifying negative result)
`memory_lambda`, `decision_lambda`, `gain`, `noise` are all **initialization / scale** knobs; the
**trained** self-gains `g·λ₀`, `g·λ₁` are set by the *task*, not the init — training regrows them.
- `sweep_gainscan` (gain {0.5,1,1.5}, init self-gains held fixed): g·λ₁ tracks gain but stays
  supercritical (1.8–2.8) — never reaches ≈1.
- `sweep_noise_g10` (noise {0.1..1}): high noise → higher g·λ₁ (robustness needs a deeper attractor);
  low noise plateaus ~2.2 and can *collapse* to a single deep nogo well. Doesn't reach the sweet spot.
- `sweep_relu_ml` (memory_lambda {0.6..1.6}, relu): trained g·λ₀ regrows to ~1.6–2.0 regardless of init
  → relu stays a near-marginal, unstable **integrator** (Re₀>0, no point attractors); DPA still solves.
- Setting `decision_lambda` small does NOT help isolation — the decision must autonomously *hold*
  go/nogo across the no-input delay, so training forces g·λ₁>1. Confirmed conceptually + by every sweep.

### 14b. Isolation is fragile (confirms §13's caveat)
`sweep_nolick1` (reg + strong nolick): driving g·λ₁→1 gives the two low wells in only ~1/3 seeds; in
the rest the marginal decision + coupling **destabilizes the memory** (0 stable attractors even in ±4.5).
"Get rid of the nogo pole" ⟺ remove the decision's down-well ⟺ isolate ⟺ fragile (`sweep_nogopole`:
raising `nogo_hinge_thresh` toward 0 does NOT remove the pole — it's the decision's autonomous down-well,
a consequence of supercriticality, and worsens at thresh 0).

### 14c. ★ WINNER — strong memory (no isolation): `sweep_mem` mem50
`memory_lambda=5` (init g·λ₀=5, trains to ~3), `kappa1_reg_weight=0` (decision left SUPERCRITICAL/robust),
gain 1, noise 0.25, nolick 0.5. A deep-enough memory forces the deep no-lick state to **retain κ₀=±1**,
splitting the memory-erasing nogo pole at (≈0,−1.5) into **two memory-preserving low wells at (±1,−0.8)**
— the two-low-wells target with a *robust supercritical* decision (g·λ₁≈2.3–2.6), DPA/go=1.0, **nogo up to
1.0**, in **2/3 seeds**. The failing seed had the *highest* g·λ₁ (2.88) → a mild g·λ₁ trim (not full
isolation) is the obvious follow-up to make it 3/3. Even mem16 (g·λ₀≈2.1) hits it in a good seed.

### 14d. Task-side attempts (both negative)
- **Ramping-GNG** (`ramping_gng` flag, `sweep_ramping`): removed the delay memory-hold so the decision is
  cue-driven (go expresses the cue ramp, nogo cancels it). Prediction: subcritical decision. **Failed** —
  g·λ₁ still ~2.2 (init 0.1 or 0.5), ring/pole persists, nogo often *worse*. The go/nogo identity still
  has to survive the delay so the net holds it in κ₁ supercritically anyway. Removing the *explicit*
  target isn't enough; you'd have to make go/nogo genuinely reactive (present at response).
- **cue_scale** (`sweep_cuescale`, cue {2,4,6,8}): hypothesis = stronger cue → deeper wells to keep nogo.
  **Refuted** — stronger cue *amplifies the decision poles* (more ring-like) and monotonically *hurts*
  nogo (0.91→0.71). And there's a **ceiling/peak**: the cue-driven κ₁ peaks ~cue 2 (~0.73·ceiling) then
  *decreases* — a very strong cue overrides the recurrence and forces the raw input-sign pattern (less
  n₁-aligned). So cue_scale's useful range is ~1–3; beyond that it only degrades.

### 14e. Why the decision/cue poles sit at κ₁≈1.5 (not π/2)
The go/nogo/cue attractors are **input-clamped** → decision units saturate → κ₁ → the readout ceiling
`(1/N)Σ|n₁|` (≈1.9), landing at ~**0.8·‖n₁‖₁/N** (the input aligns ~80% with the readout sign pattern).
Same law on the memory axis with `|n₀|` (smaller → memory poles at ±1 are shorter than decision poles).
So the ~1.5 is the trained decision-readout scale; **π/2 is a coincidence**.

### 14f. Analysis tooling built this session (see `docs/running.md`, `docs/analysis.md`)
- **`/bifurcation-probe` skill** — `bifurcation_probe.py` (g·λ₀/g·λ₁, off-diagonals, Re₀, wells, accuracy
  table), `bifurcation_flows.py` (labeled κ-plane flows), `bifurcation_gaussian.py` (generic-Gaussian
  bifurcation illustration).
- **brainpy `SlowPointFinder`** as the default fixed-point finder (`--finder brainpy`, scipy fallback),
  with `--slow_tol`/`--marg` exposed. Verified it matches scipy on tanh; adds slow-manifold detection.
  Marginal points that are transversely-attracting slow segments are relabeled **slow attractor**.
- **New `plot_sweep` summaries**: `fp_scatter_{stage}.pdf` (per stage, panel per input condition,
  across-seed attractor/slow-attractor scatter) and `fp_meanflow_{stage}.pdf` (mean vector field +
  across-seed agreement background + attractor overlay). Replaced the old `fp_scatter_by_*`.
- **Figure 1 draft**: `results/figures/paper/fig1_model.pdf` (model + κ-framework + tasks + curriculum).

## 15. Session 2026-07-21 — ★★ THE MECHANISM: attention isolates the wells (supersedes §13's attribution)

**§13 said isolation = `kappa1_reg` (hold g·λ₁≈1). That attribution is wrong.** A factorial ablation
this session shows the isolation of the two no-lick memory wells — killing the 270° ring — is done by the
**attention input**, with `kappa1_reg=0` throughout. The reg was masking the real lever.

### 15a. The ablation
Base = subcritical rank-2 recipe: N=1024, tanh, gain=1, `memory_lambda=0.8`, `decision_lambda=0.5`,
`cue_scale=2`, `nolick=0.5`, **fixed lr** (`use_scheduler=False`), `kappa1_reg=0`. All flows computed with
attention ON (`ff[-1]=1`, incl. Autonomous), ±2 window.

| sweep | attention | nolick | DPA ep | dual_dpa | dual_nogo | autonomous flow |
|---|---|---|---|---|---|---|
| `sweep_subcrit` | trained | 0.5 | 100 | 0.95 | 0.40 | 2 isolated wells, κ₁≈−0.8 |
| `sweep_nonolick` | trained | 0 | 100 | 0.95 | 0.21 | 2 isolated wells, κ₁≈−0.8 |
| `sweep_subcrit_dpa300` | trained | 0.5 | 300 | ~1.0 | 0.53 | 2 isolated wells |
| `sweep_frzatt` | **frozen** | 0.5 | 100 | 1.0 | 0.29 | 2 isolated wells |
| `sweep_noatt` | **OFF** | 0 | 100 | 1.0 | **0.00** | **3-attractor RING** (lick well returns) |

### 15b. The mechanism (not "lowering" — *isolation by killing the lick node*)
The A/B memory wells sit at **κ₁≈−0.8 in every regime** — attention does NOT push them down. What it does:
- **Attention OFF:** the go/lick node at κ₁≈+1.7 is a **stable attractor**. Autonomous flow = the old
  270° ring/U (top lick well + two bottom memory wells). On nogo trials the state falls into the lick
  well ⇒ **nogo = 0**.
- **Attention ON:** that top node becomes a **repeller**; only the two no-lick memory wells survive,
  isolated, no ring. This is exactly the geometry §13 chased with `kappa1_reg`.

So the effective self-gain story from §12/§13 (g·λ₁≈1 to kill the ring) is achieved here **by the
attention drive shifting the operating point of the decision mode**, not by a reg penalty. Removing the
reg does nothing; removing attention brings the ring straight back.

### 15c. What each knob actually does (by elimination)
- **attention = necessary and sufficient for isolation.** OFF ⇒ ring, nogo→0.
- **attention need NOT be learned.** `frzatt` freezes the attention `wi` column at random init (new
  `freeze_attention_input` flag, freezes it in DPA+GNG; Dual freezes all inputs anyway) and still
  isolates (nogo 0.29). Any fixed tonic break suffices — the specific projection barely matters.
- **`nolick` ≠ isolation/lowering.** `nnl` gives identical well positions; nolick only buys a modest
  nogo margin (0.40 vs 0.21). It is NOT the directional ingredient we thought (contra §13's ingredient-2).
- **init criticality washes out.** subcritical (λ₀=0.8) vs supercritical (λ₀=2.0) → identical geometry
  (self-gains task-locked, confirming §14a).
- **longer DPA helps nogo:** 100→300 epochs lifts nogo 0.40→0.53. (DPA never reaches stop_loss 0.1 —
  floors ~0.34 — because the pre-sample baseline is attention-OFF where κ=0 is a supercritical saddle;
  κ₁ drifts there, penalised but structurally un-removable. The Dual stage pins its baseline and does
  converge <0.1.)

### 15d. Reinterpretation of the whole thread
The years-long fragility (§12/§13/§14) was partly a **task-target problem** masked by the reg/nolick/
tanh_asym scaffolding. This session cleaned the targets (pinned baselines, clean κ₀/κ₁ role windows,
fixed the response-window `dt` bug) and went to fixed lr; with a well-posed objective the network finds
the clean two-well-in-no-lick geometry on its own, and **attention alone supplies the ring-killing
isolation**. Historic "isolation recipes" (kappa1_reg, isolate_clamp) were treating a symptom.

Figures: `rnn/{sweep_subcrit,sweep_nonolick,sweep_subcrit_dpa300,sweep_frzatt,sweep_noatt}/` in the
localhost gallery (±2, KDE mean-flows). Full inventory: `docs/experiment_log.md` (2026-07-21).

## 16. Session 2026-07-24 — transient/windowed decisions; the decay-to-0 target hurts pairing convergence

Goal of this thread: decisions that **express then relax to 0** (a lick is transient), with a short pre-cue
memory hold so nogo learns κ₁=−1 *before* the go-push cue. Implemented as **windowed targets**: 0.5 s
pre-cue hold → 0.5 s response after cue-off → optional 0.5 s decay to 0; DPA pairing = 1 s expression after
test-off → optional decay. **GNG nogo no longer reset on cue onset** (holds −1 pre-cue, free through the
cue, then decays).

### 16a. Flag split
`decay_decision` → **`windowed_targets`** (the windowing) + new **`decay_to_zero`** (default True) gating
*only* the decay-to-0 lines. Lets us ablate windowing-with-decay vs windowing-without on an otherwise
identical recipe.

### 16b. `sweep_win_decay` — decay vs no-decay (8 seeds, everything else identical)
- **No-decay = clean 4/4** — pairing 1.0, go/nogo 1.0 every seed.
- **Decay = 2/4** — s1/s3 clean; s0 stuck (val 0.82, pairing 0.59), s2 stuck (val 0.96, pairing collapses
  on go trials).
- Mechanism: the decay-to-0 target repeatedly pulls the decision back to 0, which **fights the match
  decision's need to hold at +1 against the no-lick-biased field** (§15). Without the decay the match can
  express and stay up long enough to learn the boundary; with it, 2/4 seeds fall into bad minima.
- go/nogo unaffected either way — nogo (−1/0) and go (+1 briefly) are both compatible with the no-lick
  bias; only the sustained match/+1 pairing is at odds with it. So **the decay tax lands entirely on the
  pairing.**

### 16c. Metric artifact fixed (again — same class as prior sessions)
The pairing was scored by averaging κ-last from test-off to trial **end**; under windowing that spans the
decay/free tail and dilutes the ±1 signal toward 0, reading ~chance even when the pairing is perfect in its
window. Fixed to the **expression window** in both `sweep._dual_accuracy` and
`plot_sweep._eval_dual_by_trialtype` (+ pairing label from condition names, not the NaN/0 last timestep).
Standing lesson: **always score a windowed target inside its window.**

### 16d. Takeaway
Transient decisions are fine for these nets **as long as the tail isn't over-constrained.** The decay-to-0
target is the expensive part (it fights the no-lick-biased match); leaving the post-window free gives the
same transient behaviour with far better convergence. The residual go-trial-pairing ceiling is the
shared-κ₁-axis tension (match=+1 competing with the no-lick memory on one axis) → the **rank-3 split**
(separate κ₂ lick axis) remains the structural fix.

Figures: `rnn/{arm_decay,arm_nodecay}/` (localhost gallery). Inventory: `docs/experiment_log.md` (2026-07-24).

## 17. Session 2026-07-25/31 — decay = the lower-ring lever; all-stage reg; the unified loss

### 17a. Decay favours the lower ring (measured)
`scratchpad/wells.py` (drive-and-release A/B on 'none' trials, κ₁ in the deep delay): the **decay arm
holds both memory wells at κ₁≈−0.73 (4/4 seeds); the no-decay arm lifts them to +0.30 (0/4)**. The
decay-to-0 target is what pushes the memory onto the no-lick arc — it forbids κ₁ resting anywhere but 0
after the window, so the only stable structure left (attention already killed the lick node, §15) is the
two no-lick memory wells. The convergence tax (§16) and this geometry benefit are the SAME constraint.

### 17b. All-stage decision reg — constrains the mechanism, not the incentive
`sweep_win_reg`: `kappa1_reg_weight=1` at all stages (`w·relu(g·λ_dec−1)²`). It reliably pins the
decision self-gain g·λ_dec≈1.1 — but **the network re-routes**: with a free tail, parking a decision is
loss-free, so the job that used to sit in the decision mode moves into the MEMORY mode (its g·λ grows to
2.1–2.7) → nodecay+reg wells lift even higher (+0.70). decay+reg fixes pairing convergence (marginal
decision can't build the lick attractor that fought the decay → match 1.0 on all 4 seeds, incl. the
reg-0 stuck seeds) but flattens the wells to ≈0 and destabilises go/nogo per-seed (spiral eigenvalues).
**Lesson (a new form of §14 task-locking): a reg constrains a mechanism; the task pressure re-emerges
through whatever mode is still free.** No arm dominates — decay=geometry lever, reg=pairing lever.

### 17c. The ≤−1 pairing-tail bug (fixed)
The Dual pairing decay-to-0 zeros fell into the match/nonmatch hinge's else branch → trained as
**κ₁≤−1**, a hard basement shove every trial-end, not "return to 0". Explains why decay tails parked at
−1 and part of "decay favours the lower plane" (a painted push, philosophically like `nogo_push_memory`
which we keep OFF). Fixed via `pin_decay_zeros` — all decay zeros pinned to 0 (MSE), all stages, baseline
kept separate. **Every decay-arm geometry number above predates this fix and is contaminated.**

### 17d. ★ One loss for three stages — `UnifiedLoss`
The three stage losses existed to carry task semantics in TIME MASKS. Windowed+pinned targets moved all
semantics into the target VALUES, so one value-based loss now covers DPA/GNG/Dual: **+1→one-sided hinge
(overshoot free) · −1→hinge · 0→pin (MSE-to-0) · NaN→free**. This is ThresholdLoss generalised, with
per-class/per-group separate means (no short-window dilution) and independent weights:
`bl` (own timing split, SEPARATE from decay) · `gng_pos/neg` · `gng_decay` · `rwd_go/rwd_nogo` ·
`pair_pos/neg` · `pair_decay` · `mem` · `nolick`; decision channel splits gng-vs-pair at test onset.
The zero-branch bug family is now impossible by construction (±1/0/NaN exhaustive). Intended changes:
Dual nogo pre-cue hold now ≤−1 (was gentle ≤0); pairing class-balanced. **`gng_response`** flag re-adds
the response window as a separate **rwd group** (`rwd_go` +1 hinge / `rwd_nogo` 0-pin, independent
weights = the go/nogo imbalance knob for suppressing false licks). Targets visualised per trial type in
`rnn/task_targets/`. Verified against the old losses; not yet run — it's the next objective.

### 17e. UnifiedLoss sweep (RUNNING) + two optional decision knobs
First UnifiedLoss run: three sweeps (`sweep_uni_{base,rwd,decay}`, 4 seeds, all weights 1, no reg),
isolating the response window and decay against a minimal base (base = pre-cue hold + pairing only).
rwd effect = rwd vs base, decay effect = decay vs base; all free of the ≤−1 tail bug. Two new
default-off knobs added while building it:
- **`rwd_nogo_onesided`** — response window scores ONLY nogo lick (`relu(κ₁)²`); go response and the
  nogo no-lick value both free. The one-sided no-lick philosophy applied to the response window (vs
  pinning nogo to exactly 0). "Only penalise nogo licking, nothing else."
- **`dual_gng_memory`** (Dual only) — optionally drop the go/nogo pre-cue hold, so the go/nogo working
  memory is not re-supervised in Dual and must ride the GNG-learned structure. Combined with
  `gng_response=F` this is the fully-emergent go/nogo case (only the pairing is supervised on κ₁).
Four independent Dual decision knobs now exist: `dual_gng_memory` · `gng_response`(+`rwd_nogo_onesided`) ·
`decay_to_zero` · pairing. Score the running sweeps with `scratchpad/wells.py` + expression-window acc.

### 17f. ★ Unified-loss results — "decay lowers the wells" was the ≤−1 tail bug (SUPERSEDES 17a)
`sweep_uni_{base,rwd,decay}` (unified loss, honest pinned decays, no reg). ALL solve the task
(pairing 1.0). But the geometry OVERTURNS §17a: with the ≤−1 pairing-tail bug fixed, **decay no longer
lowers the memory wells.**
| arm | pairing | go/nogo | held κ₁ (mean, deep-delay 'none' trials) |
|---|---|---|---|
| base (hold+pairing) | 1.0 4/4 | 1.0 | +0.14 (straddle 0) |
| rwd (+response window) | 1.0 4/4 | 1.0 (2/4 spiral, complex eig) | +0.19 (straddle 0) |
| decay (+decay-to-0) | 1.0 4/4 | **2/4 degraded** | **+0.02** (straddle 0) |
The OLD decay arm (separated loss, buggy tail) gave κ₁≈**−0.73**; the SAME recipe under the unified loss
(tail honestly pinned to 0, not shoved to −1) gives **+0.02**. That −0.75 is ENTIRELY the tail semantics
— i.e. the "decay favours the lower ring" result (§17a, and Leon's observation) was the accidental
basement-shove doing the lowering, not "return to 0". Under the honest loss NONE of base/rwd/decay lowers
the wells (all straddle κ₁≈0), and decay additionally COSTS go/nogo (2/4 degrade) for no geometry gain.
**⇒ no-lick well-lowering needs a real mechanism, not the decay artifact.** Honest candidates left:
`nolick_weight` on the free windows (kept at 0 so far) or the rank-3 κ₂ split. The reg re-routes (§17b).

### 17g. plot_sweep target-overlay bug (fixed)
`RunMeta` didn't carry the target-scheme flags (`windowed_targets`/`decay_to_zero`/`gng_response`/
`dual_gng_memory`/`ramping_gng`), so every trajectory plot built its dashed TARGET overlay with the
DEFAULT non-windowed scheme — wrong for every windowed run (all `sweep_win_*`/`arm_*`/`sweep_uni_*`
traj figures until now). Model trajectories, inputs, accuracy, and flows were unaffected (they don't use
the overlay). Fixed: flags added to `RunMeta` + `_load_sweep_meta`, threaded into all four trajectory
generators. `sweep_uni_*` traj figures regenerated + republished.

### 17h. ★★ First HONEST well-lowering: nolick during retention; freeze×pressure 2×2
Chasing an *intrinsic* lowering (rwd imbalance had NO effect — it's a behavioural knob on the post-cue
decision, not the memory well; decay was the ≤−1 bug §17f; reg re-routes §17b). The well is the κ₀
sample-memory attractor — its κ₁ is a FIELD property, so only a pressure on the HELD state during the
delay can move it. That pressure = **`nolick_weight·relu(κ₁)²` over the free decision windows** (Dual
stages, sample+baseline excluded — the sample exclusion added to UnifiedLoss to avoid the go-lick-ramp
floor). One-sided (κ₁<0 free) ⇒ lowering must EMERGE, no painted value.

**2×2 (nmrwd base: no pre-cue memory, pin response; unified loss; held κ₁ on deep-delay 'none' trials):**
| | freeze κ₀ | unfreeze κ₀ |
|---|---|---|
| **no nolick** | +0.06 (0/4 both<0) | +0.09 (1/4) |
| **nolick 0.5** | **−0.15 (3/4)** | **−0.16 (4/4)** |

- **nolick lowers the wells honestly**: +0.06 → −0.15, task PERFECT (pairing/go/nogo 1.0, 4/4). First
  non-artifact lowering. Confirms the mechanism: *sustained "don't lick while remembering" pressure
  tilts the held memory into the no-lick plane* — even with κ₀ frozen (it reshapes how κ₁ reads κ₀).
- **Unfreezing κ₀ is SAFE** (retention survives on the pairing alone: κ₀ A–B sep ~2.4, pairing 1.0 — the
  freeze was NOT load-bearing under the clean loss; the pairing-must-read-the-sample IS the memory
  pressure). But it did NOT amplify lowering (−0.16≈−0.15) — only made it CONSISTENT (4/4 vs 3/4).
- **⇒ the lowering magnitude is capped by the nolick PRESSURE (w=0.5), not the frozen DOF.** The nolick
  residual never→0 (~0.15) = genuine tension with the field's tendency to keep κ₁ up. To lower more,
  raise `nolick_weight` (dose-response) — the next lever. Modest so far (−0.15, not the old −0.8).
Figures `rnn/sweep_uni_{nolick,unfrozen_nonolick,unfrozen_nolick}/`.

### 18. ★★ An FP-classifier bug hid a real result — relu lowers the wells (2026-08-03)

Hunting a **structural** (not painted) well-lowering lever, we tried rectifying nonlinearities on the
clean base (unified loss, unfrozen κ₀, `attention_gated`, nmrwd = `dual_gng_memory=False`, **no
nolick**, `memory_lambda=0.8`, structured init, 100/100/300): `sweep_relu_cap` (relu ×4),
`sweep_softplus_sp` (softplus ×4), plus the tanh L1-pin sweep `sweep_nogo_pin`
(`rwd_nogo_l1=True`, w∈{0.1,1.0}, ×4). First reads said relu/softplus form a *continuous/marginal*
memory manifold at κ₁≈0 (0/4 bistable) — a dead end. **That was a measurement bug**, and the truth is
the opposite.

**The bug (two parts).** (1) `classify_fixed_points` tags any FP with a discrete-map eigenvalue within
`marginal_tol=1e-2` of the unit circle as "marginal", and `wells.py`/the plots drop those. Shallow
subcritical & non-saturating wells are genuine but **SLOW** attractors (map |λ|≈0.99) → mislabeled
marginal. (2) The analytic root-finder searches the global `XLIM=±2` box, but non-saturating φ put the
wells at κ₀≈±5–25 → it misses them or lands on a nearby saddle. `--auto_xlim` widens only the flow
axes, not the FP-search box.

**Ground truth = grid-sim** (forward-integrate a κ-grid, cluster settled endpoints — no Newton, no
derivative, no eigenvalue knife-edge, so relu's Heaviside kink is a non-issue): the grid collapses to
**two tight discrete points** per seed (σ≈0.05–0.1, same order as tanh) = genuine bistable wells.

**Result (autonomous, attention-on FPs; grid-sim for relu/softplus, analytic+fixed-tol for tanh):**
| φ / arm | task | bistable A&B | well κ₁ | frac<0 |
|---|---|---|---|---|
| **relu** | 4/4 | 2/4 clean (2 asym-basin) | **−0.29** | **6/6** |
| softplus | 3/4 | 3/4 | −0.63 | 4/6 |
| tanh, L1 pin w=0.1 | 4/4 | 4/4 | +0.29 | 0/8 |
| tanh, L1 pin **w=1.0** | 4/4 | 4/4 | **−0.18** | 5/8 |

- **relu is the first STRUCTURAL lowering**: discrete bistable wells, **every one below the no-lick
  line** (κ₁≈−0.29), task perfect, **no nolick / no painted κ₁** — the non-saturating rectifier's
  intrinsic even (|x|) curvature tilts the field down emergently. softplus goes the same way (messier,
  huge κ).
- The bug also **corrupted the pin-sweep conclusion**: `pinng10L1` (strong L1 pin) is actually **4/4
  bistable, wells mostly <0 (−0.18)**, not "1/4, memory degraded". "Strong pin degrades memory" and
  "rectifiers give continuous wells" were the *same* artifact.

**Fixes.** `wells.py`: non-saturating φ → grid-sim ground truth (adaptive κ-box; tanh/erf keep analytic
with `marginal_tol=2e-3`), plus a marginal-well κ₁ stat. The classification default itself
(`classify_fixed_points` / `classify_sim_fixed_points`, dynamics.py) is lowered **1e-2 → 2e-3** — that's
what fixes the FLOW figures, whose FP overlay is a THIRD call site (`plot_task_flow_fields`) using the
default; before it, the flow plots showed the slow wells as faint "marginal" squares (or dropped them
via `_reduce_marginals`) instead of filled attractor dots. `plot_sweep` also passes `MARGINAL_TOL=2e-3`
at its two scatter call sites; re-render relu/softplus with wide `--xlim` (search box = plot box). The
analytic path is exact and recoverable — at 25 seeds + the tol fix, `s0_relu`'s wells come back correct
(matches grid-sim); the residual fragility is sparse seeding over wide κ + relu's non-smooth φ′ (tuning,
not fundamental), and the analytic finder is still needed for the plots (it finds saddles; the sim only
finds attractors). Figures `rnn/sweep_{relu_cap,softplus_sp,nogo_pin}/`.

### 19. Attention amplitude fails; relu's landscape is bad (2026-08-04)

**Attention is the symmetry-breaker, but amplitude is not a lever.** The odd-φ "wells forbidden below 0"
argument (§ theory §3) assumes `b⊥n`; but attention is clamped ON in the autonomous field, so `b_attn`
sits in the readout plane and breaks the odd symmetry (theory §8 route 1). Decomposed on the trained
nets, the well κ₁ = **attention-direct** term `⟨n₁,φ(g·b_attn)⟩/N` (tanh: −0.13, DOWN) vs the
**memory-modulated even coupling** `⟨n₁·φ''(g·b_attn)·m₀²⟩` (tanh: +0.17, UP). A fixed-weight scan
suggested scaling attention flips the net-even negative at ~2.5–3×. **But retraining refutes it**
(`sweep_wellpush` wp_attn1/2/3, `attention_scale` knob added to `_attn_window`): wells at +0.07/+0.02/
**+0.17** — *higher*, not lower. The trainable attention weight `wᵢ[:,−1]` re-optimizes to neutralize
the bias (theory §8 route-1 caveat, confirmed); the fixed-weight de-risk froze exactly that weight.

**relu's low wells sit on a BAD landscape** (Leon): κ₀≈±3–6 (softplus ±10–25), only 2/4 clean-bistable
(the rest asymmetric-basin), and theory §5 spiraling (~15% of the plane, relu never saturates). So the
§18 relu −0.29 is a low well on a degenerate/blown-up/spiraling landscape — not a real solution.
`wp_lifsc` (bounded rectifier) → +0.30 (UP): **non-negativity alone isn't enough**; relu's
*unboundedness* is what shifts down, and that is exactly what wrecks the landscape. `wp_reludeep`
(relu λ₀=3) diverged (κ→10¹², 3/4 task-fail). ⇒ clean landscape (needs saturation) and wells-below-0
(needs unbounded rectification) were mutually exclusive across everything. Figures `rnn/sweep_wellpush/`.

### 20. ★★ lif (Gaussian CDF) + decision-readout DC — clean landscape, and the well BIFURCATES down (2026-08-04)

**The transfer function φ = ½[1+erf(x/√2)] (Gaussian CDF = `lif`)** is non-negative **and** saturating —
the "saturating directional break" the tension needed. Being non-negative, baseline firing φ(0)=½ makes
the RESTING decision κ₁ = φ(0)·⟨n₁⟩ = ½⟨n₁⟩, so `½[1+erf] = ½ + ½erf` ⇒ `Ψ₁ = ½⟨n₁⟩ + ½·(erf field)` —
a clean DC shift by the **mean of the decision readout** `⟨n₁⟩`, on a compact saturating landscape.
New init knob **`decision_readout_mean`** (`init.py`: add DC to `n₁` after install; sets ⟨n₁⟩ exactly,
leaves λ₁ untouched since m₁ is zero-mean). Toy (n=M, no attention/baseline): wells shift to ½⟨n₁⟩,
amplified by feedback.

`sweep_lifdc` (lif, gain 2, λ₀=3 supercritical since φ'(0)≈0.4, `decision_readout_mean ∈ {0,−0.3,−0.6,
−1.0}`, clean base, no nolick, 4 seeds):
- **Landscape SOLVED** — compact bistable wells at κ₀≈±1, no relu blowup, no spiral; **task-perfect**
  (dual_dpa≈1, go/nogo=1) at every DC. lif is a viable clean φ and the memory holds.
- **The DC works, but by BIFURCATION not translation.** As ⟨n₁⟩→negative each sample memory splits
  into an UP (κ₁≈+0.5) and a DOWN (κ₁≈−0.6) attractor. At `lifdc10` (⟨n₁⟩ init −1.0 → trained −0.75)
  **all 4 seeds show 4 wells: 2 up + 2 down**, down wells at **κ₁≈−0.58** (≈½⟨n₁⟩ amplified, both
  samples). Monotone: down wells appear/multiply as ⟨n₁⟩ drops. So the mechanism **creates memory wells
  in the no-lick plane** — closer than anything prior. Remaining: kill the UP copies (GOAL "all wells
  down" still 0/4). Also: the baseline-pin (pre-sample κ₁=0) **erodes** ⟨n₁⟩ (init −0.3/−0.6/−1.0 →
  trained −0.06/−0.31/−0.75), fighting the DC — a lever to protect.

**`wells.py` had a TWO-WELL bug that hid all of the above.** `_side_well` picked one attractor per κ₀
sign (outermost |κ₀|) and averaged A with B — so a 4-well (2up/2down) structure collapsed to 2, and an
up-well averaged with a down-well read as a meaningless ≈+0.08. This mis-reported the well locations
repeatedly. **Fixed:** `wells.py` now lists EVERY memory-well attractor (|κ₀| large) with (κ₀,κ₁), tallies
up/down, and reports the GOAL metric "ALL memory wells down / seed"; the grid-sim path clusters all
attractors too (no per-side split). Earlier sweeps (relu, attn, softplus) should be re-checked with it.
Figures `rnn/sweep_lifdc/`.

### 21. ★★ A DPA-metric bug hid convergence; the DC×attention 2×2; and why the nogo-hinge never pushes the wells (2026-08-05)

**★ The `dpa≈0.50` in every windowed run was a METRIC BUG, not under-training.** `_dpa_accuracy`/
`_dpa_accuracy_by_type` (`sweep.py`) regenerated DPA trials **without** `windowed_targets`/`decay_to_zero`
and read the target at the LAST timestep `y[:,-1,-1]` — which is 0/NaN once the windowed decision decays.
So `pair`/`unpair` masks were empty (**pair=nan**) and `overall` collapsed to ~0.50 for **every**
`windowed_targets` run. DPA was actually always solving: probe on the DPA-stage ckpt gives match κ₁=**+0.43**
vs nonmatch **−0.46** (Δ=0.89), memory κ₀ held ≈0.73. **Fixed** via a shared `_dpa_score()` that reads the
SUPERVISED decision window (post-test steps where the ±1 target is set); re-scored runs show **dpa=1.0**.
This retroactively corrects every "task-perfect" claim above — they reported the buggy 0.50, not true DPA.

**The DC × attention 2×2** (all lif, gain 2, λ₀=3, one-sided nogo, windowed, DPA 250 ep). Autonomous wells:

| cell | sweep | wells | down-κ₁ | all-down | g·λ₁ |
|---|---|---|---|---|---|
| **DC + attn** | `sweep_os_ep` | 4 (2 deep-dn + 2 ≈0) | **−0.67** | 3/4 | 9.4 |
| DC only (no attn) | `sweep_noattn` | erratic, collapses to 1 well | −1.03 | 1/4 | 11.9 |
| attn only (no DC) | `sweep_nodc` | clean 2-well | −0.09 | 3/4 | 11.0 |
| neither | `sweep_nodc_noattn` | straddles 0 | −0.08 | 1/4 | 10.8 |

**Verdict: the DC does the DEEP lowering** (structural coordinate shift ½⟨n₁⟩, unopposed); **attention
STABILIZES the two-sided A/B memory** (without it the deep-DC push collapses the memory to one side). Both
needed. Decision is supercritical (g·λ₁≈9–12) in every cell — the DC works by shifting the whole bistable
structure down, not by changing criticality. More DPA epochs deepened DC+attn vs the original run (−0.45→−0.67).

**freeze-attention-in-GNG is now DEFAULT** (`sweep.py` `gng_freeze_input` adds the attention channel when
`attention_input`): attention is a DPA-learned tonic **context** input → trained in DPA, frozen GNG+Dual, like
the DPA dims. A/B test `sweep_nodc_afrz` vs `sweep_nodc`: small deepening **−0.09→−0.14**, within seed noise —
confirms GNG was diluting the push, but the attention push is inherently small.

**★ Why the `gng_response` nogo-hinge does NOT push the wells** (corrects three of my wrong assumptions —
the response cue IS shared across go/nogo `tasks.py:290`, noise IS always on `:271`, and the response window
IS cue-off `co:co+half`, co=cue-offset). The real reason: **the loss is indifferent to well depth below 0** —
κ₁=0 already means "no lick", so nothing rewards a nogo well at −0.5 vs 0. The **only** below-0 pressure is the
**noise-robustness margin** (avoid noise-driven false licks); the attn-only wells sit at −0.09 ≈ the input-noise
scale (0.093), i.e. exactly one noise-width. The one-sided **never-lick collapse** (go=0) removes even that.
The DC supplies depth for free. Emergent levers under test: **(a)** raise input noise (bigger margin → deeper
wells) — `sweep_noise` (`nzA`, noise 0.5/0.75); **(b)** **go-preserving one-sided** (new `rwd_keep_go_hinge`:
keep the go +1 hinge so go MUST lick → under the shared cue the nogo memory must sit below the lick line) —
`nzB`, queued after (a).

**Also this session:** `gng_criterion` is always TWO-sided (`rwd_nogo_onesided` only relaxes Dual, via
`_uw_gng`) so one-sided can't collapse the go-memory in GNG; `stop_loss` 0.02→**0.05** (0.02 overtrains Dual
to ~300 ep for no gain; DPA converges well above it); `plot_sweep`/`dynamics` now emit **STACKED 3-row
(dpa/naive/expert) flow portraits** (`plot_stage_stacked_flow`) + stacked summary mean-flow
(`_render_meanflow_stacked`) — same panel columns, shared κ-limits + speed scale, so a column reads top→bottom
as DPA→GNG→Dual. Figures `rnn/sweep_{os_ep,noattn,nodc,nodc_noattn,nodc_afrz}/`.

## 22. Session 2026-08-06/07 — cue-driven response; the NOISE mean field; genuine sim-trajectory flow

Arms this session: rank-2 baseline `r2go` (go-preserving, no decay) + rank-2/3 cue-driven `r2cue`/`r3cue`,
rank-3 baseline `r3o` (subcritical κ₂, no decay). Sweeps `sweep_r2go`, `sweep_r3o`, `sweep_cue`.

### 22a. Cue-driven response (`response_in_cue`) — necessary framing, NOT sufficient
New flag (sweep.py + all 3 generators + shifted eval windows): score the go/pairing response in the **last
0.5 s of its triggering stimulus** (cue/test ON) instead of after it turns off, so the lick can be
input-driven rather than held from memory. Diagnosis it was meant to fix: the old window `co:co+half` (co =
cue-OFFset) *demands a lick when the cue is already gone* → the net is forced to wire the held rule κ₁ into
κ₂ → that coupling IS the up-copy. **RESULT: cue-locking ALONE made the up-copies WORSE** — r3cue go-rule
wells κ₂≈+0.7 (vs r3o +0.3), 0/4 all-down. Two reasons: (1) the lick the net learns is **additive** (κ₂ ≈
rule-drive + cue-drive), not a **conjunction** (rule AND cue); holding the go-rule is mandatory, so an additive
readout leaks a standing lick regardless of when it's scored. (2) We ran `decay_to_zero=False` (purest
emergent) → removed the only post-cue downward pressure r3o still had (its pin-to-0). **Conclusion:
"make go cue-driven" ⟺ force the conjunction, and that needs an ACTIVE downward force** (one-sided κ₂ decay,
or attention-baseline depression) — timing alone doesn't do it. r3o already has subcritical κ₂ (g·λ=1) so the
lick *can't self-hold*, but the bistable rule still drives it up: subcriticality is necessary, not sufficient.

### 22b. ★ The NOISE mean field — input-only EXACT Gaussian resummation (production)
The deterministic reduced field ignores the training input noise (σ_eff = noise·√(1−e^{−2α}) ≈ 0.37 for
noise=1). Correct object: E_ξ[Ψ]. For a Gaussian-CDF φ (**lif** c=1, erf c=2, lif_sc c=2π) the input-noise
average is **exact for all σ**:
  **Ψ_σ(κ) = (1/N) Σⱼ nⱼ φ(āⱼ / √(1 + c·sⱼ²)),  sⱼ² = g²Aⱼ²σ²‖wⱼ‖²**  — noise DIVIDES each neuron's drive
by √(1+c sⱼ²), a **per-neuron effective-gain compression**. Why it is exact and not just a good
approximation: ⟨Ψ⟩ = (1/N)Σⱼ nⱼ⟨φ(aⱼ)⟩ is LINEAR in the expectation, so only each aⱼ's **marginal** is
needed — the cross-neuron correlations induced by the shared ξ never enter the mean field.

**Magnitude — do NOT read the per-neuron factor as the mode-level one** (measured on s0_r3o10, σ=0.37):
per-neuron √(1+c s²) averages **2.12** (min 1.27, max 5.50), but the effect on the REDUCED modes is much
milder — g·λ at the origin goes 2.11/1.16/1.16 → 1.33/1.05/1.05, i.e. only **1.58× / 1.11× / 1.11×**. The
compression sits inside φ' evaluated at a compressed argument and is reweighted by the n·m overlap, so it
is *not* a uniform rescale of g. (An earlier note claimed "≈2.3×, would drop g·λ=3→1.3, subcritical" — wrong
twice over: the mode-level reduction is 1.58×, and g·λ≈2.11 is the *trained* value, not the init 3.0.)

Wired into `low_rank_field_np(noise_sigma=σ)` / `low_rank_jacobian_flow_np(noise_sigma=σ)`.
**Validated to ~2e-3 vs Monte-Carlo E_ξ[Ψ]** (`scratchpad/validate_noise_field.py`). The naive 2-term φ''
Taylor **fails** here (½⟨s²⟩≈2.15 is not small — origin −0.21 vs MC −0.07) and is no longer used anywhere.
Tools: `rank3_flow.py --noise` (lif/lif_sc added to its jax PHI), `plot_sweep --field_input_noise`
(REWIRED 2026-08-10 to the analytic term, f6f8d72: 8.2× faster and exact — legacy estimator still
available via `--field_noise_mc`), `scratchpad/wells3.py`.

**Non-Gaussian-CDF φ (fixed 2026-08-10, commit dcc0cec).** Two bugs found in the math review:
(i) **relu was treated as noise-transparent** ("φ''=0 a.e.") — false, relu's φ'' is a **delta at 0**, and
⟨relu(ā+η)⟩ = ā·Φ(ā/√Δ) + √Δ·N(ā/√Δ); at ā=0,Δ=1 the code returned 0.000 vs the true 0.399, so the noise
field silently returned the DETERMINISTIC field for every relu net (relevant to §18's relu arms).
(ii) the production field still used the bad φ'' **Taylor** for non-Gaussian φ while the SC path used
quadrature (tanh ā=0.5,Δ=1: Taylor 0.0987 vs true 0.2952). Both fixed: everything now routes through the
single `_phi_avgs` — exact closed form (Gaussian-CDF φ, relu), Gauss-Hermite otherwise. **lif is
bit-identical**, so every lif result above stands.

### 22c. Noise result — destabilizes MARGINAL wells, not a directional fix
Clean-vs-σ wells (flows published for all four sweeps):

| sweep | rank | noise effect on go-rule up-copies | all-down @σ |
|---|---|---|---|
| r2go10 (σ0.37) | 2 | pushes up-wells to/below the line | **1/4** (s2 → 4w, 0 up) |
| r2cue | 2 | shrink, survive | 0/4 |
| r3o10 | 3 | shrink; sometimes kills the *down* (good) wells (s0) | 0/4 |
| r3cue | 3 | **s1: both up-copies ANNIHILATED** (saddle-node) | 1/4 |

Noise compresses the gain and tips whichever wells are **shallowest** over the saddle-node — clears up-copies
in only **2/16 seeds** across configs, and can equally kill the desired down-wells. **Not a directional fix**;
it nudges marginal wells, consistent with the "well depth = noise-robustness margin" story (§21).

### 22d. Self-consistent DMFT — ⚠ EXPERIMENTAL (over-predicts stiff modes)
Derived + implemented the recurrent-variance closure (`solve_sc_variance`, `low_rank_field_sc_np`,
`low_rank_jacobian_sc_np`): Δᵢ = input-direct + cross + **Mᵢᵀ C Mᵢ**, C = (I−σ̃)⁻¹ g²σ²ŨŨᵀ(I−σ̃)⁻ᵀ,
σ̃ = (g/N)nᵀdiag(φ̄')M (renormalized overlap = reduced Jacobian), Ũ = (1/N)nᵀdiag(φ̄'A)W; iterate to
self-consistency ((I−σ̃)⁻¹ amplifies near criticality). **Validation vs noisy sim (`scratchpad/validate_sc.py`):
right STRUCTURE (rule-mode variance matches exactly at σ0.37) but OVER-predicts the stiff/slow modes ~10–20×** —
our **two-timescale discrete dynamics temporally FILTER** the injected noise (fluctuation–dissipation), a factor
the instantaneous-variance closure omits; it bites the stiff κ₀ memory hardest. **Decision (Leon): keep the
input-only exact term as production, SC as qualitative** (finish it by folding the α/α_rec FDT factor into s²).

### 22e. Dubreuil (`~/models/dubreuil`) comparison — their flow field is finite-N DETERMINISTIC
`low_rank_rnns/ranktwo.py:plot_field` computes `F = −x + m(nᵀtanh x)/N + I` in the full N-space then projects —
the true finite-N field, **no noise, no Gaussian integrals, no self-consistent equations anywhere in the repo**
(grepped). Same family as our reduced field; algebraically **equivalent for a pure single-timescale low-rank net**
(they add I_orth to the φ-argument + affine in-plane; we put the full input inside φ — same result). The
self-consistent-noise theory is the Mastrogiuseppe–Ostojic *framework*, not coded here. What Dubreuil does that
generalizes: full-N-space FP finding (robust when there's off-plane W_fixed).

### 22f. ★ Genuine simulated-trajectory tooling (`--use_sim_field` was a one-step map)
Leon flagged that `plot_sweep --use_sim_field` "looks analytical" — correct: with `n_warmup=0` it seeds on the
slow manifold and takes ONE step, collapsing to ≈β·(analytic field). New honest tooling:
`src/dynamics.integrate_kappa_trajectories(model, ff, κ₀, n_steps, noise_sigma)` (rank-general primitive — seed
full state on the κ-manifold, integrate the true two-timescale dynamics with input clamped; σ>0 = noisy
trajectories) + **`traj_flow.py`** CLI (rank-2: stages×conditions; rank-3: conditions×3-planes `--stage`;
`--noise`). Confirms the analytic reduced field IS the true flow for these pure low-rank nets (the state can't
leave the m-manifold), while exposing transients / empirical basins / stochastic wander. Figures
`rnn/sweep_{cue,r2go,r3o}/flow` (clean + `_noise` + `trajflow`). Also `scratchpad/plot_cue_{targets,inputs}.py`
(target + clean/noisy input time-courses). Committed on `main` (f6f8d72).

### 22g. ★ WHERE THE GOAL STANDS — the reframing, and the open routes

**The sharpest statement of the problem this session produced.** The lick the nets learn is **ADDITIVE**:
κ₂ ≈ (rule-drive) + (cue-drive). What the goal requires is a **CONJUNCTION**: κ₂ high ⟺ (rule = go
**AND** cue present). With an additive readout the up-copies are not a bug to be tuned away — they are
*forced*: holding the go-rule is mandatory (you need it to know whether to lick later), and any additive
κ₁→κ₂ coupling therefore leaks a standing lick at the held-go state. **"Make go cue-driven" ⟺ "force the
conjunction."** Every negative result below is an instance of attacking the symptom instead of the AND.

**Three routes now closed (each necessary, none sufficient):**
| route | tested by | why it fails alone |
|---|---|---|
| make the lick **subcritical** so it can't self-hold | r3o (g·λ₂=1) | a subcritical unit still faithfully *tracks its input* — the held rule drives it up |
| make the response **cue-locked in time** | r3cue (§22a) | removes the *demand* for a memory-held lick, not the *ability*; and dropping the post-cue pin removed the only downward pressure ⇒ WORSE (κ₂ +0.7 vs +0.3) |
| let **noise** flatten the wells | §22c, all 4 sweeps | non-directional — kills whichever wells are shallowest (2/16 seeds; in r3o s0 it killed the *good* down-wells) |

**Open routes, in the order I'd try them:**
1. **r3cue + one-sided κ₂ decay** (the direct next experiment; base config exists, just add
   `decay_onesided=True`, `gng_decay_weight` 0.5 and 1.0). Adds an active downward force on top of the
   cleaner cue-locked response. Rank-3 is what makes this affordable: the decay's spiral lives in the
   κ₁–κ₂ plane, so the κ₀ sample memory is spared (the rank-2 tightrope of §17/§21). *Imposed, not emergent.*
2. **Attention-baseline depression** (`attention_scale` 1/2/3 on the r3o or r3cue base) — the *emergent*
   version of the same downward force: a tonic readout-plane bias that lowers the κ₂ rest point until the
   retained rule alone is sub-threshold and only rule+cue crosses. This is the §15/§19 mechanism applied
   to κ₂ and **has never actually been run in rank-3**. Threshold-with-a-depressed-baseline *is* an AND
   gate, so this is the closest thing to implementing the conjunction structurally.
3. **A genuine multiplicative gate** — the only route that makes the readout a conjunction *by
   construction* rather than by tuning a threshold. Nothing in the current architecture does this;
   it would need a gating nonlinearity or a cue-modulated κ₂ gain. Biggest change, most principled.

**Diagnostic worth running first (cheap, no training):** clamp the response cue ON vs OFF on an existing
r3o/r3cue net and measure κ₂'s marginal move at a go-rule state. If it is additive (fixed cue increment on
top of a positive rule floor) that confirms the framing and quantifies how far the floor must drop (~0.4).

**Tooling debt / smaller open items:** (a) the SC-DMFT needs the α/α_rec fluctuation–dissipation factor to
become quantitative (§22d); (b) a flow-fields *skill* was scoped but not built — the doc map now lives in
`docs/analysis.md`; (c) the across-seed `fp_meanflow` published for sweep_r2go was rendered from ONE seed
(verification run) so its agreement background is meaningless — re-render on the full 8-run sweep if wanted.

## 23. Session 2026-08-10 — ★★ SIGN-BASED supervision: the rule wells DISSOLVE (not pushed — gone)

### 23a. The reframing (Leon): only 2 wells, and where the 4 came from
Target portrait restated: the autonomous landscape should have **exactly TWO wells — the A/B sample
memories — both κ₁<0**; go/nogo/lick must be input-driven/transient, never autonomous. The 4-well
portraits of §22 (2 up / 2 down) are *supervised into existence* by amplitude demands sitting on the
shared rank-2 lick axis: (i) the windowed ±1 **pre-cue holds** (dim 1 IS the readout in rank-2 — the
up/down copies are literally the targets); (ii) the go→+1 response and nogo≤−1 delay hinge (park at
amplitude for the 1 s rule→cue gap ⇒ autonomous supercritical κ₁); (iii) the DPA **two-sided 0-pin**
on the readout from t=0 to test-on (`tasks.py`, clamps the sample wells ON the line). The historical
**spiraling** of subcritical attempts is the same conflict: reach ±1 (amplify) AND be transient
(contract) is only jointly satisfiable by a complex eigenpair — rotation. Remove the amplitude
demands and pure contraction satisfies the loss: no spiral, no parking. XOR/counter-cue task
redesigns were considered and REJECTED (they dissolve the go/nogo asymmetry under study).

### 23b. First launch (`sweep_r2sign`, thresholds 0/0) — two calibration traps
go_hinge_thresh=nogo_hinge_thresh=0 (pure sign) + response_in_cue on the r2go10 base. Two traps:
(1) hinges at exactly 0 have a **degenerate optimum κ₁≡0** (both sides zero-loss at 0) broken only
by noise tails — and the realized κ fluctuation is FDT-filtered well below σ_eff, so the drive is
weak; (2) sign hinges shrink the loss scale ~25× ⇒ **stop_loss=0.05 was already met at GNG start**
— GNG stopped at ~5 epochs (16 s), Dual ~20 (95 s). go learned instantly (0.96+, the additive cue
push), **nogo never got its margin** (0.38–0.64), dual_gng≈0.56. Portraits: sample wells clamped AT
κ₁=0 (the DPA pin, (iii) above) + a (0,±2) autonomous **pairing** attractor pair — built by DPA's
±1 pairing hinge on the same κ₁ axis. Lesson: in rank-2 every ±amplitude anywhere on the readout
recruits autonomous κ₁ structure; and loss↔accuracy decouple near 0 (margin² pricing).

### 23c. ★ The sign2 calibration (Leon): margin on the HOLD, sign on the RESPONSE, delay FREE
`sweep_r2sign2` arm `sign2` = r2go10 base + response_in_cue +
- **holds** (pre-cue rule memory): go ≥ **+0.25** / nogo ≤ **−0.25** (`go_hinge_thresh=0.25`,
  `nogo_hinge_thresh=−0.25`) — the 1 s rule trace must carry a real ±ε separation (ε≈⅔σ_eff);
  kills the κ₁≡0 optimum without ±1 parking (a ±0.25 decaying trace is subcritically feasible).
- **response** (cue ON): go ≥ +0.25 strict (a lick must clear threshold); nogo ≤ **0** one-sided,
  FREE below (`rwd_nogo_onesided` + NEW `gng_rwd_onesided` so it applies in the GNG stage too —
  the legacy GNG-stage two-sided nogo pin is gone; response-side depth left fully emergent).
- **DPA delay**: NEW `dpa_prelick_free=True` — readout supervision only pre-sample; sample→test
  fully FREE (Leon: no delay penalty at all; wells placed by pairing training alone).
- `stop_loss=0.005` (≈1.5–2σ margins at the sign scale). Pairing & κ₀ memory keep ±1
  (`dpa_hinge_thresh=1`).
Infra (all legacy-bit-identical, smoke-tested `scratchpad smoke_sign{,2}.py`): `UnifiedLoss` gained
**`gng_thresh`/`gng_neg_thresh`** — gng-side hinge thresholds split from `thresh` (pair+mem), wired
from go_hinge_thresh/−nogo_hinge_thresh (without the split, sign-gng would silently relax the
pairing/κ₀ amplitude supervision — caught in review); eval decision boundary now follows the loss:
`(th_go + max(nogo_target, nogo_hinge_thresh))/2` (legacy 0.5/0.0 preserved; sign2 → 0.125).

### 23d. ★★ RESULT — structure achieved; wells straddle the line (the measured cost of the free delay)
Task (4/4 seeds, full budgets): dpa ≥0.995, **after_gng/dpa ≥0.994**, gng nogo ≥0.996 after GNG,
dual_gng 0.87–0.97 (s1 weakest). Geometry (expert autonomous field; scipy FPs + direct-field
verification + Jacobian classification):

| seed | #att | sample wells (κ₀,κ₁) | extras | spirals | dual_gng |
|---|---|---|---|---|---|
| s0 | **2** | (−1.39,−0.17) (+1.34,+0.19) | — | none | 0.955 |
| s1 | 4 | (−1.40,−0.13) (+1.38,+0.20) | 2 weak central | none | 0.872 |
| s2 | **2** | (−1.35,−0.09) (+1.27,+0.28) | — | none | 0.944 |
| s3 | 3 | (−1.25,+0.34) (+1.28,+0.30) | 1 weak central (spiral) | wells clean | 0.966 |

Three structural firsts, all 4/4: (1) **NO autonomous rule attractors** — the up-copy problem is
not pushed down, it's *dissolved*; the rule is a genuine transient. (2) The DPA-stage autonomous
pairing pair (0,±2) **dissolves during Dual** (response_in_cue reads pairing test-ON — no parking
needed). (3) **Zero spirals at wells** (real negative flow eigs — the amplify-vs-contract conflict
resolved). Noise field (σ_eff≈0.37): everything survives, wells pulled in to ≈±1.1, no
annihilation. **The gap:** wells sit at κ₁ ∈ [−0.17,+0.34] — STRADDLING the line, not below.
With the delay unsupervised nothing selects the side; free-delay placement is line-centered ±0.25
(s3 both above). **Open decision:** the below-line pressure — one-sided delay hinge (the §17h
family, now with a clean 2-well substrate) vs noise↑ vs `attention_scale` — is the single
remaining step to the goal portrait.

### 23e. Tool caveats found this session
(a) `bifurcation_probe`'s **brainpy pass returned spurious FPs** on these runs (claimed wells at
κ₀≈±2.8 where direct evaluation gives |F|≈1.4) — use the scipy finder (`find_all_fixed_points`)
and verify |F| directly; probe needs a fix. (b) `plot_sweep --field_input_noise` **overwrites**
`fp_stages.png` (same filename as the clean render) — rename to `fp_stages_noise.*` (done for
sweep_r2sign2). Figures: `rnn/sweep_r2sign2/{flow(+_noise),traj,accuracy,targets}` +
`rnn/sweep_r2sign/*` (the failed-launch reference).

## 24. Session 2026-08-11 — the rwd_window BUG; threshold/weight/softplus dose arcs; ★ the FACTORIZATION

### 24a. ★★ BUG: the response window never followed `response_in_cue` (fixed; invalidates a class of readings)
The unified loss's rwd carve-out was hardcoded to the LEGACY post-cue slice `(co, co+half)` while
`response_in_cue` writes the targets IN-cue `(co−half, co)` — disjoint intervals. Consequences for every
response_in_cue unified run (sign2, th1, first th1w launch): (i) `rwd_go`/`rwd_nogo` computed over an
EMPTY target set (= 0 forever; `rwd_nogo_weight` inert — caught because w5/w10 same-seed runs trained
BIT-IDENTICAL for 300 epochs); (ii) the real in-cue nogo 0-targets fell through to the generic class
rules = **two-sided pin to 0** — the "one-sided free-below" nogo response NEVER RAN before the fix, and
explains sign2's nogo hovering at +0.05 (pulled toward 0 from below too). FIX (sweep.py, all three
UnifiedLoss sites): window follows the flag. Diagnosis receipts: `dual_loss_components` rwd_*=0.0;
post-fix w5-vs-w10 divergence at GNG epoch 10. Sanity rule going forward: a dosed term must show a
nonzero component, and bit-identical same-seed arms = an inert term.

### 24b. Threshold control `sweep_r2th1` (= sign2 with all hinge thresholds at 1, single-variable):
parking is CAUSAL in the threshold. Rule wells + spirals RETURN at th=1 (4/4/2/5 attractors, s3 textbook
up/down copies both sides, 3/4 spiral wells) on the identical free-delay/one-sided base. Behaviourally
"best" at its own boundary (dual_gng .99) — but that boundary is (1+0)/2=0.5 and at the honest boundary 0
nogo(≤0)=0.40–0.52: the lenient eval masked near-licks at +0.3. (Pre-fix loss, so nogo was pin-scored.)

### 24c. `sweep_r2th1w` (th1 + rwd_nogo_weight 5/10, POST-fix, from th1 DPA ckpts): behaviour fixed on
the parking substrate — nogo(≤0)=0.93–0.99 (means −0.18…−0.30), go≥.99, dual_dpa=1.000, spirals mostly
gone — but geometry = the 270° U again: one side splits into a deep up/down parked pair (down copy at
κ₁≈−0.6), 3 attractors/run. Depth on an amplitude substrate buys a U (twice now: old decay arms, th1w).

### 24d. `sweep_r2sign2w5` (sign2 + fix + w5, from sign2 DPA ckpts): substrate SURVIVES the pressure —
exactly 2 attractors, no U, no spirals (3/4 seeds; s3 grows 2 weak extras) and the nogo RESPONSE goes
below (means −0.10…−0.14, from +0.05). **But the memory wells DO NOT MOVE: still straddling
(−0.23/+0.27), identical to sign2.** Prediction that cue-coupled response pressure would pull the wells
was WRONG — the network sinks the state only where scored, via input coupling, leaving the autonomous
wells untouched.

### 24e. Softplus arc `sweep_r2sp` (hinge_shape="softplus" on ALL hinges, th ±1, gng response moved
POST-cue via new `gng_rwd_after_cue`, w 1/5, full runs): the BCE-with-logits shape (gradient σ(x) never
dies on the correct side — the §21 "nothing rewards depth" fix, from the NeuroFlame `BCEOneClassLoss`;
NOTE the shipped notebook used bce_alpha=1.0 = pure hinge, so the branch was never validated there).
RESULT: **indiscriminate depth-reward inflates EVERY supervised amplitude into parked structure** —
memory wells at κ₀≈±2.3–2.9, 4–6 attractors, deep MEMORY-LESS basement states at (κ₀≈0, κ₁≈−3), and at
w5 DPA breaks (dual_dpa .64–.73). Loss scale note: softplus floor ≈ ln2 per satisfied ±1 class ⇒ "val
1.8 with dpa=1.0" is CONVERGED; stop_loss is dead in softplus runs. Lesson: the depth incentive must be
SELECTIVE (no-lick terms only), never on holds/pairing/memory.

### 24f. ★★ THE FACTORIZATION (the arc's one theorem-shaped result) + honest scoreboard
Across sign2 / th1 / th1w / sign2w5 / spw: (1) **hinge thresholds set the substrate** — ε≈0.25 ⇒
transient rule, 2 wells; th=1 ⇒ parking/U; softplus-everywhere ⇒ inflation+parking. (2) **response-window
pressure sets only the in-cue/post-cue state depth** — absorbed by input coupling at any weight/shape/
placement. (3) **the MEMORY WELLS answer only to DELAY-time forces — and the delay is the one window
still unsupervised (Leon's "don't penalize", §23).** Honest scoreboard (memory-well κ₁ only; a deep
κ₀≈0 attractor is NOT pushdown — it carries no sample bit): sign2 −0.17/+0.19 · sign2w5 −0.23/+0.27 ·
th1/th1w parked +0.2…+0.5 · spw +0.2…+1.3 ⇒ **0 seeds all-down in every arm of the arc.** The only
both-below result in the project record remains §17h delay-nolick (−0.16, 4/4, shallow). Open decision
unchanged but now fully cornered: supervise the delay (nolick relu² at 0 / at −ε / selective-softplus)
or accept straddling.

### 24g. Tooling: `flow_verdict.py` + the `flow-verdict` skill (misreading-proof scoring)
Repo-root tool: scipy FPs on a ±4.5 box, every root re-verified via |F(κ*)| (the brainpy path returned
SPURIOUS wells at ±2.8 with |F|≈1.4 — do not trust unverified finders), attractors split into MEMORY
WELLS (|κ₀|≥0.5, ± pair) vs EXTRAS, per-run ALL-DOWN verdict + spiral count + behaviour at the FIXED
boundary 0, summary k/N. Skill `.claude/skills/flow-verdict/` encodes the 8 traps of this session
(basement≠pushdown, behaviour≠geometry, fixed boundary, verify FPs, wide box, figure legend/overwrite
gotchas, inert-term check, per-seed reporting). Housekeeping: pre-Aug checkpoints deleted (~1.7 GB,
incl. EISTP refs — Leon's call; §11 no longer re-analysable without retraining), cuda:0-ONLY policy
(cuda:1 = another user's), gallery server wedged once (restart procedure works).

### 25. ★★ Delay supervision: the NeuroFlame port, three hinge shapes, and the metric I was reading wrong (2026-08-12/13)

The §24f cornered decision — *supervise the delay, or accept straddling* — got taken. Six sweeps, and the
first well-pushdown that actually responds to a knob. Also the correction that reframes all of it: on the
project's PRIMARY metric (`after_gng/dpa`) the winning-looking arms are the worst (Leon, 2026-08-30).

**25a. Why NeuroFlame's dual training makes 2 lower wells** (`~/models/NeuroFlame/org/train/dual/train_dual.org`).
Its Dual loss hinges the CHOICE axis DOWN during the memory DELAY on every no-lick trial. `gng_idx` =
distractor-onset→cue-onset (l.1590) read on the choice axis, `class_bal` hardcoded 1 (l.434) ⇒ full
two-sided sign hinge; with `bce_alpha=1.0` the BCE part is OFF and `SignBCELoss` reduces to
`relu(thresh − sign(2t−1)·κ)`, so **class 0 means "≤ −1"**. Labels (l.1639–51): `labels[2]=1` for Go ONLY —
**NoGo *and* the cue-less DPA trials both get the down-hinge**, and on DPA trials the net is sitting ON the
sample attractor (no distractor, no cue), so the hinge grades the WELL's κ₁ directly. Its response/cue
window uses `class_bal=0` → only `0.1·|κ|`, a weak symmetric pull. I.e. all down-force in the delay, none
in the response — the exact mirror of every arm we had run.

**25b. The port.** `nolick_late_delay` (RunConfig) restricts the existing nolick term to a WINDOW —
Dual `(cue-off 6.5, test-on 8.0)`, GNG `(cue-off 4.5, end)`. Free-(NaN)-steps only, so finite response
targets inside the span keep their own rwd terms. Covers nogo + 'none' (pure-DPA) delay and the go
post-response tail. `UnifiedLoss(nolick_window=…)`; no new target class (Leon: "we don't need a new target").
Two companions: `nogo_target=None` (nogo response target dropped as REDUNDANT with the nolick hinge —
same one-sided form) and `gng_decay_to_zero` (GNG stage pins BOTH types to 0 from response-end to trial end,
`decay_to_end` in `generate_gng_trials`).

**25c. Sweeps** (all lif rank-2, N=1024, gain 2, noise 1.0, in-test pairing, `dpa_prelick_free`; ALL-DOWN =
every memory well κ₁<0, `flow_verdict`, fixed boundary 0). `sweep_r2spnl` w1/w5 → **2/8** — first ALL-DOWN
of the project, dose-dependent (w1 wells hug the line, nolick at its ln2 floor; w5 wells −1.4…−2.2 but 2/4
NO PAIR + dpa 0.65–0.79). `sweep_r2spnlc` (go response back IN-cue) → 3/8; up wells die in some seeds only.
`sweep_r2spnld` (+`gng_decay_to_zero`) → **3/4, up wells gone in ALL seeds**, s1/s3 exactly 2 wells + origin.
`sweep_r2renld` (= spnld, relu²) → 2/4, wells pinned at the line, up-copies+spirals return, |κ₀| 1.1–1.5
(vs softplus 2.6–3.3). `sweep_r2renldw` (w0/w5 ladder) → 1/8; **w0 = 0.01 nogo-correct**, w1 = 0.90 ⇒ the
delay term is the ONLY source of no-lick behaviour. `sweep_r2linld` (L1) → 3/4, exactly 2 wells in the good
seeds, `in_well` 0.96–1.00. `sweep_r2linldh` (+ hinged nogo response) → 3/4 and nogo@0 0.65→0.98 mean.

**25d. Three diagnostics that matter more than the sweeps.**
(i) **The effective decision gain is a strong function of κ₀** — the reason GNG is "so sensitive to what
happened on κ₀" (Leon). `g·λ₁ᵉᶠᶠ(κ₀) = g·(1/N)Σ nᵢ₁mᵢ₁ φ'(g(mᵢ₀κ₀+Iᵢ))`, measured at the naive ckpt with
attention on: **0.91 at κ₀=0 → 0.19 at 1.0 → 0.03 at 1.1 → −0.09 at 1.25**; mean φ' 0.117→0.027, units with
φ'<0.05 **37%→85%**. A κ₀ well is stable BECAUSE it saturates the population, and saturation is not
mode-selective, so it removes the slope κ₁ needs. Consequence: identical weights score **1.00 nogo in the
GNG task and 0.00 in the Dual task** — the rule trace is −1.11 at cue-on in GNG but has already crossed to
+0.08 in Dual before the cue, which then pushes it to +0.74 (cue rides input ch.4 = the GO channel at
`cue_scale`=2×). NOTE this also corrects the raw-overlap reading: `gain·n₁ᵀm₁/N` = 10–25 assumes φ'=1
(tanh convention); for lif the effective gain is ≈0.9 at rest — near critical, NOT 10× supercritical.
(ii) **The nogo response window was unsupervised** under `nogo_target=None`: 0/5632 finite target steps in
5.98–6.48 s while the nolick window starts at 6.48 — nogo crossed into the lick region exactly at decision
time (+0.17 expert, +0.75 naive). "Windows disjoint" was reported as a feature; it was the hole. Fixed by
`linldh1` (`nogo_target=0.0` + `rwd_nogo_onesided` = one-sided hinge, not a pin).
(iii) **No naive-dual evaluation exists.** After GNG we score the DPA and GNG tasks separately (both
healthy) and never the DUAL task at the naive ckpt — the actual starting condition for Dual. The 0.00 nogo
above appears in no `results.jsonl` field. Two-line fix in `_eval`, not yet made.

**25e. Hinge shapes = how force scales with violation size** (`hinge_shape`; `relu` added 2026-08-13,
and the 0-target PINS follow the norm — |p| under relu, p² otherwise). relu² force 2x **vanishes near the
target** (0.10 at x=0.05, 10× weaker than relu; parity only at 0.5) ⇒ near-threshold straddling is nearly
free. relu force 1 up to threshold then 0 ⇒ crisp satisfaction. softplus σ(x) never dies ⇒ depth keeps being
rewarded — the ONLY shape that asks for depth, which is why it was the only one that moved wells; the price
was inflation of EVERY hinge. **relu and relu² share an equilibrium: zero gradient once satisfied, so
neither can place a well strictly below 0 — depth needs a DISPLACED threshold (`relu(κ₁+ε)²`), not a
different shape.** Measured confirmation: 'none' trials rest flat at κ₁ = −0.11 (relu², residual 0.001 = no
gradient left) vs −1.24 (softplus, still pushing).

**25f. ★ The corrected scoreboard (Leon, 2026-08-30: "you are wrong about your conclusions").** I ranked arms
on geometry + `dual_dpa`, and `dual_dpa` is measured AFTER Dual has repaired the memory (rank-0 trainable in
Dual). On the project's stated key metric `after_gng/dpa`: **relu² 0.84 mean (0.57/0.91/1.00/0.88) ≫
softplus 0.44 ≈ L1 0.42 ≈ L1+hinge 0.45 (three of four seeds at/near chance)**. Also never reported:
`after_dual/gng` (GNG task after Dual) is **at chance in every arm except linldh1** — spnld 0.50–0.60,
renld 0.49–0.76, linld 0.48–0.54 vs linldh1 0.69–1.00 — so the earlier "3/4 ALL-DOWN" portraits belong to
nets that had lost go/nogo standalone. And ALL-DOWN flatters: wells sit at κ₁ −0.09…−0.38 while σ_eff≈0.37,
i.e. inside one noise SD of the line. **No arm is good on both axes.** relu² retains DPA through GNG but has
up-copies and loses GNG in Dual; L1+hinged-nogo gives clean 2-well portraits and keeps both tasks within
Dual but sacrifices DPA retention. Wider box (±4.5) re-run confirms no structure was missed at ±2.

---

## §26 — Scoring the TRAJECTORIES (2026-09-02): the memory doesn't decay, it FLIPS

**26a. `traj_verdict.py` + the `traj-verdict` skill — the behavioural half of the verdict.**
`flow_verdict.py` answers "where are the wells"; nothing answered "does κ(t) do the right thing on the
task", so every trajectory read was an eyeball of one panel. The expected κ(t) (Leon, 2026-09-02) now
lives in ONE place — `EXPECT` at the top of the file — and is scored per stage:

| stage (ckpt) | probe | expected |
|---|---|---|
| DPA (`dpa_`) | DPA | A/B memory on κ₀ maintained or decaying, sample-off → test; choice on κ₁ at test offset |
| GNG (`naive_`) | GNG | go/nogo held on κ₁ at ±θ, maintained or transient; the cue pushes BOTH classes toward +κ₁; a response is expressed (sometimes wrong) |
| GNG (`naive_`) | DPA | memory maintained or slightly disrupted; choice still on κ₁, sometimes wrong |
| Dual (`expert_`) | Dual | memory maintained or decaying; go/nogo as in GNG; choice on κ₁, mostly right |

Checks: `mem` (κ₀ hold across the delay → HELD/DECAY/FADE/GROW/LOST/**FLIP**), `rule`, `cue`, `resp`,
`relax`, `nolick`, `choice`, `prelick`. Sign tests at the FIXED boundary 0 (flow-verdict trap 3), so
the two tables compose. ~40 s per 4-seed sweep on CPU.

**★ Expectations follow the run's own loss** (`_variant`/`_adapt`, mirroring sweep.py's `UnifiedLoss`
construction) — Leon's requirement, and the thing that makes the table valid across arms. The spec's
"±1" is really ±θ: θ⁺=`go_hinge_thresh`, θ⁻=−`nogo_hinge_thresh`, θ_pair=`dpa_hinge_thresh`. Those are
ONE-SIDED hinges, so a level test is a FLOOR (beyond θ is free), never a band, and at θ=0 the floor is
σ_eff — otherwise every sign-based arm fails for behaving exactly as designed. Free targets
(`nogo_target=None`, `rwd_nogo_onesided`, `gng_response=False`) are reported and NOT scored (marked
`·`); `nogo_target=None` with `nolick_weight=0` is tagged `★nogo-UNCONSTRAINED` because then nothing
imposes don't-lick at all. `nolick_late_delay` adds its check, scored on exactly the steps the loss
left free (`isnan(y)`); `decay_to_zero`/`kappa1_reg_weight` make `relax` scored — `gng_decay_to_zero`
only for the GNG probe, since it is a GNG-STAGE-ONLY flag (getting that wrong imported a Dual
expectation the loss never imposed). Each run prints a `variants:` line: never quote a level verdict
without it.

**26b. ★ The finding: GNG INVERTS the sample bit in ~44% of runs.** First fleet run, all eight
delay-supervision arms (48 run-stages, n=512, trained σ): on the GNG-stage DPA probe the memory splits
**21 FLIP · 7 LOST · 2 FADE · 11 DECAY · 7 HELD**. FLIP (sep at delay end < 0.45) is not forgetting —
the A/B code is REVERSED. Trace evidence, s3_linldh1 at `naive_`, κ₀(A)−κ₀(B) through the DPA delay:
**+1.70 (sample-off) → +0.44 (5 s) → −1.50 (test-on)** — it crosses zero mid-delay and settles in the
opposite well. Not a start-window artefact: s0_renld1 on the identical measurement merely decays
(+1.92 → +0.29). Independent corroboration from a different quantity (κ₁ at test): seven runs score
DPA **below chance** after GNG — s3/s2_linldh1 0.28/0.29, s3/s2_linld1 0.29/0.30, s2_spnld1 0.31,
s3_spnlc5 0.37, s1_linld1 0.44. Noise-only degradation sits at 0.5; a reversed code does not.

This is what `after_gng/dpa ≈ 0.42` (§25f) was averaging over, and the two failure modes are different
problems: a decayed memory needs amplitude, an inverted one needs the κ₀ dynamics not to reverse.
**Mechanism NOT established.** rank-0's own overlap is bit-identical `dpa_`→`naive_` (g·λ₀ = 10.66 both
— the freezing works), while the rank-1→rank-0 coupling g·n₀ᵀm₁/N grows ~3× during GNG (−1.36 → −4.22
in linldh) — but it grows the same amount in renld, which only decays (−1.56 → −4.24), so coupling
alone does not separate flip from decay. These are RAW overlaps (φ′=1, the tanh convention); the
φ′-weighted effective version at the operating point is the next thing to compute before claiming a
route. Candidate: the flip needs a sustained κ₁ offset during the delay to drive κ₀ through zero, so
the arms that hold κ₁ away from 0 in the DPA delay should be the ones that flip — testable against the
`prelick` column, which the tool already reports.

**26c. Two more things the table surfaced.**
- **Dual-stage memory GROWS more often than it decays**: 21 GROW · 17 HELD · 7 LOST · 3 DECAY. Most
  GROW sit at hold 1.28–1.6 (κ₀ ≈ 0.85 after the sample → ≈ 1.2 at test); linldh is extreme at 2.6–3.9
  (0.19–0.49 → 0.74–1.38). The sample kick lands SHORT of the well and the attractor pulls it out
  during the delay — neither "maintained" nor "decaying". The DPA stage does not do this (hold
  1.04–1.10), so it is not the 0.25 s start-window settling bias.
- **The nolick dose ladder, per seed and per quantity**: nogo response at boundary 0 in Dual goes
  w0 → **0.00/0.01/0.01/0.03** (κ₁ +0.30…+0.46 — it licks on every nogo trial) vs w5 →
  0.92/0.49/0.99/0.98 (κ₁ −0.35…+0.03). Same `dpa_`/`naive_` checkpoints, so this is purely the Dual
  delay term — an independent confirmation of §25c's "the nolick term is the only source of no-lick",
  now with the κ₁ values rather than a single accuracy.

---

## §27 — Why the designed task couldn't give the target solution, the reset, and ★ THE FOUNDATION (2026-09-02)

**27a. The three-gap analysis (why §25's task+loss never produced "wells down, memory retained").**
Answering Leon's "I want to understand why the task and loss we designed does not provide this
solution". The target is specified in GEOMETRIC, CROSS-STAGE terms (the A/B bit persists through
GNG with bounded scrambling; wells sit below the line by a cue-sized margin; frozen pieces stay
calibrated) — but every optimized term is behavioural and stage-local:

- **Gap 1 (GNG, a SELECTION problem):** the GNG batch contains no A/B trial, so no gradient prices
  the memory; freezing m₀,n₀ protects parameters, not the attractor. Demonstration: s0 vs s3 of
  linldh are indistinguishable to every loss term (GNG 1.00 both, dual_dpa 1.00 both) yet s0 kept
  the bit and s3 inverted it. New measurement (GNG task at `naive_`): the flipper's GNG trials sit
  IN the sample wells (median |κ₀|=0.43, 50% beyond 0.5) with sign(κ₀) RULE-LOCKED (+κ₀↔nogo −1.09,
  −κ₀↔go +1.16) — the rule was stored partly along the memory axis; the survivor's trials hover at
  the origin (median |κ₀|=0.11, 1% in wells). Two degenerate basins; φ′ saturation at visited wells
  (gain 2) tilts selection toward the entangled one. Correction to the §26 fix: TWO couplings are
  involved — n_dec⟂m₀ kills only the κ₀→κ₁ readout leak; the κ₁→κ₀ drive is n₀ᵀφ′m_dec, so the
  projection must also do m_dec⟂n₀ (not yet implemented as a pair).
- **Gap 2 (Dual, an OBJECTIVE problem):** the nolick hinge is a boundary demand with a 1.5 s
  recovery window — "recovered by test" is satisfiable by a well at 0⁻; depth ∝ cue-push is never
  required and is actively expensive (match climb + φ′ gain steal). Observed depths −0.09…−0.38 vs
  σ_eff≈0.37 = exactly the one-sided-hinge noise equilibrium. The deep-well solution has strictly
  higher loss.
- **Gap 3 (the pipeline SEAMS):** each stage relocates geometry other frozen pieces were calibrated
  to (cue sized under naive geometry then frozen; Dual re-deepens wells → sample kick lands short =
  the GROW finding; GNG-task routing dies after Dual). Retention (`after_gng/dpa`) is a metric, not
  a term; rank-0 trainable in Dual lets Dual REBUILD, hiding the damage from every loss.

**27b. ★★ THE FOUNDATION — `sweep_r2nocue` (Leon's reset: simplest two-memory task, NO cue).**
"Let's first run a sweep without cue in gng and the simplest flags. AB memory, go nogo memory."
Config (the baseline we ALWAYS build on): lif, N=1024, rank 2, **gain 1.0**, noise 1.0, τ=0.3,
lr 0.01 fixed, 250/100/300 epochs, stop 0.005; **cue_scale=0.0** (window kept in the timing —
adding the cue later is a one-scalar delta); supervision = A/B ±1 on κ₀ across the DPA delay +
pairing (0.25 s at test-off, free tail), go/nogo ±1 on κ₁ over stim-off→phantom-cue (**new flag
`gng_hold_full_delay`**, symmetric with the A/B supervision; re-supervised in Dual via
`dual_gng_memory=True`); EVERYTHING else off — no response window (`gng_response=False`,
`nogo_target=None`), no nolick, no decay pins, no decoupling, no reg; `dpa_prelick_free`; standard
freezing (GNG: rank-0+DPA inputs; Dual: all inputs; rank-0 trainable in Dual). Verified from the
artifacts: cue-window input identically 0; Dual loss line `nolick_w=0.0`.

**Result: the stripped task produces the desired solution class, 4/4 seeds.**
- Behaviour: `after_gng/dpa` = **0.996–1.000** (vs 0.42–0.84 in every cue-era arm); both tasks
  ≥ 0.999 after Dual.
- After GNG = exactly "scrambled but not lost": mem HELD 4/4 (hold 0.97–1.00, sep 1.00→1.00),
  wells at (±1.2–1.3, ·) with small κ₁ offsets — leak 0.03/0.24/0.25/0.55. Even 0.55 flips nothing
  at gain 1 (the 0.50 flag threshold was calibrated at gain 2 — tolerance is wider here).
- Geometry: DPA stage ends with 2 memory wells PLUS parked pairing attractors at (0,±1.5) (free
  tail → ±1 decisions persist); GNG reuses (0,±1.5) as the go/nogo memory (rule@0 = 1.00, relax
  MAINT 0.93–0.97 — parked, nothing demands transience; caveat: at naive the finder shows only the
  +κ₁ attractor in s1/s2, nogo may ride a slow transient); at expert the free-standing κ₁
  attractors DISSOLVE — only the two memory wells remain (s0–s2), rule held semi-transiently
  (relax 0.70–0.82) on top of them.
- NO pushdown claim: expert well κ₁ offsets (−0.09…−0.47; s3 diagonal +0.54/−1.21 S) are ~1
  σ_eff and there is no down-pressure in this loss — incidental, and that's the honest baseline.
- Couplings never entangle: GNG moves n₀ᵀm₁ by ≤0.4, n₁ᵀm₀ by ≤0.1 (vs growth to −4.2 in the
  gain-2+cue lineage). Raw g·λ ≈ 6.6–8.6 both modes.
- Clean control for the cue: pairing and κ₀-at-test IDENTICAL on none vs go/nogo trials
  (1.00/1.00/1.00; κ₀ ±0.94–0.99 both) — any future by-condition gap is attributable to the cue.

Three things were stripped at once (cue, response window, gain 2), so attribution between them is
exactly what the next sweep isolates.

**27c. ★ FOUNDATION ADDON — `sweep_r2cue1` (DONE): the cue at scale 1.0 is ABSORBED, and the
missing safeguard becomes visible.** = nocue + **cue_scale=1.0**, cue INPUT only — no targets
during or after the cue (Leon: "Don't add any target during or after cue, just the input itself").
DPA stage REUSED from sweep_r2nocue via `dpa_ckpt` per seed, so GNG/Dual start from the identical
memory solution — every delta below is attributable to the cue alone.

*What held (per seed vs nocue):*
- **Retention untouched**: `after_gng/dpa` 0.987–1.000; mem HELD 4/4 (hold 0.97–0.99, sep
  1.00→1.00); leak at naive 0.35/0.32/0.62/0.00 vs nocue 0.24/0.25/0.55/0.03 — the cue adds only
  a whisper of scrambling, no flips. Couplings stay clean at naive (|n₀ᵀm₁| ≤ 0.7): no
  entanglement at gain 1 even with the cue present.
- **Pairing untouched**: by-condition split 1.00/1.00/1.00 (none/go/nogo), κ₀ at test-on identical
  cued vs uncued (±0.94–1.04). At scale 1.0 the κ₁ displacement does NOT misroute the test
  decision — the predicted DPA-pairing damage does not appear at this dose.

*What the cue does (the asymmetry, measured):*
- Dual cue Δκ₁: **nogo +0.73…+0.76**, go +0.21…+0.29 (already near ceiling), 'none' +0.01…+0.04
  (clean control). Up-only and rule-indiscriminate, as designed.
- **After cue-off, nogo lingers AT/ABOVE the line for the whole late delay**: mean κ₁ over
  cue-off→test-on = +0.07/−0.01/+0.16/+0.17 with 49–76% of steps >0 (nocue: −0.15…−0.37,
  18–41%). By the task's own (untrained) response criterion this reads as licking — the
  `dual_gng` eval drops to 0.71–0.77 for exactly this reason, while every TRAINED quantity (rule
  hold 1.00, choice 1.00) is perfect. Nothing in the loss penalises the lingering, so the net
  never comes down: **the missing-safeguard premise, now measured on a controlled before/after.**

*Geometry (expert):* the wells go ASYMMETRIC — the +κ₀ well sits at/above the line in all seeds
(+0.02…+0.23) while the −κ₀ well dives in s0/s2 (−0.71/−1.17); s3 both up (+0.51/+0.23); s1's
−κ₀ attractor is NOT FOUND by the finder (NO PAIR) though behaviour is intact (sep 1.00,
dual_dpa 0.999) — likely a slow manifold or narrow basin; flagged, not diagnosed. Dual-stage
n₁ᵀm₀ grows to +1.4…+1.8 in s0–s2, same as in nocue — a Dual effect, not a cue effect.

*Where this leaves the program:* foundation + cue1 cleanly demonstrate the safeguard PREMISE —
the cue parks nogo in the lick region through the late delay because nothing asks otherwise, and
the memory tolerates it at this dose. Next levers (Leon's call): the no-lick demand on this
foundation (the safeguard experiment proper — watch wells/relaxation vs pairing cost), or a
cue_scale ladder (2.0, 4.0) to find where the push alone breaks retention/routing.
Figures: gallery `rnn/sweep_r2cue1` (70 PNGs, ±2.5 box).

**27d. ★ FOUNDATION ADDON 2 — `sweep_r2cue2` (2026-09-03): the dose–response of the push; the
wells drift INTO the lick region.** = cue1 with **cue_scale=2.0**; same nocue DPA ckpts, same
everything else. Note first an internal consistency check: the GNG stage has NO supervised step
after cue onset (the hold ends there; no response/decay), so cue amplitude has no gradient path in
that stage — cue1/cue2 naive checkpoints agree up to float nondeterminism (leak matches to two
decimals), and every dose effect below lives in the DUAL stage (where the cue precedes the
pairing supervision).

The three-point ladder (per-seed ranges, one shared memory start per seed):

| | cue 0 | cue 1 | cue 2 |
|---|---|---|---|
| Dual cue Δκ₁ nogo | — | +0.73…+0.76 | **+1.25…+1.35** |
| nogo late-delay κ₁ (steps>0) | −0.15…−0.37 (.18–.41) | +0.07…+0.17 (.49–.76) | **+0.25…+0.46 (.81–.96)** |
| untrained no-lick criterion (dual_gng) | ~1.00 | 0.71–0.77 | **0.50–0.52 (chance)** |
| after_gng/dpa | 0.996–1.00 | 0.987–1.00 | 0.987–1.00 |
| pairing by-gng (none/go/nogo) | 1/1/1 | 1/1/1 | **1/1/1** |
| +κ₀ expert well κ₁ | ≈0 | +0.02…+0.23 | **+0.27…+0.68** |

- **What doubles with the dose:** the push, the residue (at cue 2 nogo spends 81–96% of the late
  delay above the line — every nogo trial reads as a lick at the untrained criterion), and the
  well UP-DRIFT: with nothing opposing it, Dual training re-equilibrates the wells where the
  cue-displaced states live, pulling the memory structure INTO the lick region (s3: both wells at
  +0.68/+0.69; s0 grows a third memory-carrying attractor (+0.82,−1.39) — proliferation; s1 again
  single-well by the finder, behaviour intact — the same fragile seed at every dose). The exact
  inverse of the target geometry, for exactly the §27a Gap-2 reason: no down-pressure exists.
  **⚠ SUPERSEDED by §27e (Leon, same day): "well up-drift" is point-attractor language for the top
  end of what is actually an elongating SLOW MANIFOLD — the FP finder was sampling near-roots
  along a groove (that is also what the "proliferation" and the "missing well" were). Read §27e.**
- **What refuses to break:** retention (mem HELD 4/4, sep 1.00 at every dose) and — against the
  §27-era prediction — PAIRING: by-condition 1.00/1.00/1.00 with κ₀ at test ±0.94–1.10 even with
  nogo parked at +0.4 through the late delay. At dose ≤2 the κ₁ displacement does NOT misroute
  the test decision. The routing-damage mechanism is NOT yet observed; only the no-lick vacancy is.

Figures: gallery `rnn/sweep_r2cue2`. Next arm (designed, not launched): the no-lick demand on top
of cue 2, where the vacancy is maximal — prediction: it must reverse the well up-drift, and any
suboptimality should finally appear in pairing.

**27e. ★ FOUNDATION ADDON 3 — the cue does not move wells, it ELONGATES the slow manifold across
the lick boundary (Leon's reading, confirmed 2026-09-03).** Leon: "increasing cue_scale does not
push the attractors down but reshapes the manifold… a slow manifold that with increasing cue
strength occupies more and more of the no-lick region." Measured with
`scratchpad/slow_manifold_dose.py`: the slow set (|F| < 0.05, exact analytic autonomous field,
attention on) of every expert checkpoint across the dose ladder, transversal stability at slow
points, and the mean A-nogo trajectory (cue-on → test-on) overlaid. Figure:
`results/figures/sweep_r2cue2/slow_manifold_dose.png` (gallery `rnn/sweep_r2cue2/flow`).

| slow set at expert | cue 0 | cue 1 | cue 2 |
|---|---|---|---|
| total area (κ²/seed) | 0.06–0.25 | 0.20–0.29 | 0.22–0.50 |
| κ₁-extent, +κ₀ side | ~[−0.6, +0.1] (compact) | [−1.2, +0.2] | **[−1.6, +0.8]** |
| manifold signature (fast eig < −0.2, slow ≈ 0) | 0–17% of slow pts | 30–57% | up to 56% |

- **Confirmed:** each compact well elongates with dose into a slow GROOVE spanning ≈2 κ-units of
  κ₁; the transversal check says a genuine 1-D slow manifold (attracting across, near-marginal
  along), not a shallow basin. The §27d "up-drift", s0's "proliferation" and s1's "missing well"
  were all the FP finder sampling near-roots along this groove — point language the data outgrew.
- **Refinement to the claim:** the groove grows into the no-lick region AND above the line — a
  symmetric stretching of the slow direction across the boundary, not a downward migration. The
  deep reach is monotone in dose in every seed (slow-set bottom ≈ −0.6 → −1.2 → −1.5), while the
  FRACTION below the line can fall at cue 2 (91→60% s1, 87→50% s2) because the up-side invades too.
- **The trajectory rides the groove:** the cue drives nogo from the rule hold (≈−0.85) UP the slow
  direction; at test-on the state sits at the groove's upper end, above the line — the flow there
  is <0.05, so the descent timescale ≫ the 1.5 s late delay. **The late-delay residue is TRANSIT
  TIME, not missing structure: the no-lick branch of the manifold already exists at cue 2.**
- Consequence for the next lever: a no-lick demand on this substrate does not need to CREATE
  κ₁<0 attractors — it needs to move the state's LANDING POINT down an existing slow direction
  (or steepen the return flow). Falsifiable form: a SMALL nolick weight should suffice, and its
  effect should appear as a repositioning along the groove rather than new structure.
- Caveats: ε=0.05 is one threshold (the dose ORDERING is robust; absolute areas scale with ε);
  clean-field maps (σ_eff≈0.37 blurs transit, not the slow direction). Watch item: at cue 2 the
  lower branches nearly CONNECT the two memory sides through the no-lick region (a partial
  lower arc, cf. the old 270°-U) — a potential A↔B drift channel; sep is 1.00 at this delay
  length, so it is slow enough for now.

**27e′. ⚠ REVIEW CORRECTION (2026-09-03, fresh-eyes pass): the "transit time" claim in §27e is
WRONG under the trained noise, and only partly true deterministically.** Tested by integrating the
TRUE autonomous dynamics (delay condition, attention on) for 10 s from the actual nogo test-on
states (`integrate_kappa_trajectories`, 64 trials/seed):

| nogo κ₁ from test-on | t=0 | 1.5 s | 10 s deterministic | 10 s at trained σ |
|---|---|---|---|---|
| cue1 s0/s1/s2 | +0.13/+0.10/+0.36 | +0.05/−0.01/+0.18 | −0.11/−0.50/−0.35 | **+0.17/+0.09/+0.29** |
| cue1 s3 | +0.32 | +0.35 | **+0.35 (up-attractor)** | +0.43 |
| cue2 s0/s1/s2 | +0.40/+0.52/+0.69 | +0.35/+0.45/+0.52 | +0.17/+0.18/+0.22 (never <0) | **+0.26/+0.33/+0.47** |
| cue2 s3 | +0.67 | +0.69 | **+0.68 (up-attractor)** | +0.64 |

- Deterministically, cue1 s0–s2 DO drift down — on a 5–10 s timescale (≫ the 1.5 s late delay),
  so "transit time" was right there; cue2 s0–s2 descend so slowly they never cross 0 in 10 s; and
  s3 at both doses sits in a GENUINE up-attractor (flow_verdict had verified it: (+1.24,+0.68)).
- **Under the trained noise, NO seed at either dose ever comes down** (κ₁ = +0.09…+0.64 at 10 s).
  The upper end of the groove is an attracting plateau that noise cannot leave. Waiting would
  never fix the false lick.
- Corrected reading: the cue TRANSPORTS the state (input-driven, not along the autonomous flow)
  from the lower slow region to the groove's UPPER END, which is a shallow (cue1) to firm (cue2)
  attractor. The deep no-lick branch exists as slow structure but is NOT the state's fate. So the
  next lever's job is RESHAPING (destabilise/remove the upper plateau, seat the state on the lower
  branch), not speeding transit. The "small nolick weight suffices" prediction is downgraded from
  expectation to hypothesis — it may still hold because the down-structure exists, but the
  mechanism claimed for it was wrong.
- Also verified this pass (naive-dual, §27f below): the go-trial pairing failure is specifically
  NONMATCH-on-go (nocue 0.52/0.52/0.60/0.98; cue2 0.02–0.26 — systematically WRONG, the held/pushed
  +1 answers "match"), and at cue2 nonmatch-on-nogo also fails (0.67–0.70) as the cue lifts nogo.
  The pairing readout is a κ₁-position readout; whatever sits up at test says "match".

**27f. The interference the standard eval never showed (Leon: "dpa accuracy is not dropping…
I would expect it to drop together with nogo").** `_eval` probes only the standalone tasks. On the
DUAL task at the NAIVE checkpoint: dpa = 0.91–0.97 (nocue), 0.79–0.88 (cue2); by condition none
0.99–1.00 / go 0.76–0.99 (cue2: 0.51–0.66) / nogo 0.92–1.00 (cue2: 0.87–0.96); nogo response at
cue2 0.00–0.10. On the standalone DPA task the drop exists as MARGIN erosion under the 1.00
ceiling: pairing d′ 5–13 → 2.7–4.8, mean margin +0.77 → +0.62…+0.71. Why the standalone number
stays 1.00: margins ≫ decision noise (sign criterion saturates); the standalone task never
populates the rule structure GNG built; everything DPA reads is frozen in GNG. The conflict only
exists when the rule is actually held — the dual task. Consequence: `after_gng/dpa` is saturated
in the foundation regime; **naive-dual dpa (and its go:nonmatch cell) is the interference metric
for this line** and should be added to `_eval` and to traj_verdict as a standard stage.

**27g. FOUNDATION ADDONS 4–6 (2026-09-03/04): gain 2 · the cue's push · the lick-vs-hold degeneracy.**
- **`sweep_r2g2cue1`** (foundation at gain 2.0, cue 1.0, DPA retrained): WEIGHT-level interference
  returns in 2/4 seeds — s2: pure-DPA acc 1.00→0.73, pairing d′ 5.3→0.33, m₁ restructured 48 %
  (cos 0.90), memory hold 0.61, −κ₀ well missing at naive, even 'none' dual trials fail nonmatch
  (0.46); s1: d′ 12→2.7; s0/s3 foundation-like. Attribution is clean: the GNG stage is
  gradient-blind to the cue epoch (no supervised step after cue onset), so everything at naive is
  gain-only. Expert: attractor proliferation (s1/s2 four wells, s2 a spiral). ⇒ §27a's saturation
  mechanism is causal on its own; gain 1 = state-level interference only, gain 2 = weight-level.
- **Cue-push factorial** (weights × attention × memory): at NAIVE weights the nogo push is the same
  ~+0.3 in the GNG task and the Dual task; the +0.75 (§27c) is made by DUAL TRAINING (×2.5 with the
  input column frozen — the retrained κ₁ mode un-stiffens the held states). Attention through the
  cue adds ~+0.3 at expert, ~0 at naive; the held κ₀ memory adds nothing (removing the sample slightly
  INCREASES the push).
- **`attention_through_cue`** (new flag; GNG-task gated attention runs to cue-OFF instead of dropping
  at cue onset — the generic 'off at last-stim onset' rule made GNG inconsistent with Dual) +
  `sweep_r2atc1` (GNG stage only, `epochs_dual=0`): naive weights IDENTICAL to cue1's to 4 decimals
  (m₁, n₁, Wi) — the GNG stage sees nothing after cue onset; push +0.30 vs +0.25. A task-consistency
  fix, consequential only once a post-cue target exists. (Trap caught: `--run_filter` is a
  substring match; `attn1` collided with the old `wp_attn1` arm — check the match list before launch.)
- **Drive decomposition (why the cue — which IS the go input — barely pushes nogo):** exact via the
  analytic field, direct drive = F₁(κ; attn+cue) − F₁(κ; attn). At rest +0.65 (and the unstable
  origin amplifies: the go stimulus only TIPS, a 0.5 s stimulus reaches +0.65 and the attractor
  carries it to +1.2); at the nogo hold (0,−1.4) +0.21 with the well pulling back −0.1 → net ≈ +0.1/τ
  ≈ the measured +0.25 over the 0.5 s cue; at the go hold (0,+1.3) +0.05–0.11 — the column is
  gain-dead at the very state it built. Same input, effect = input × local φ′ + local flow: the
  §25d gain-steal on the κ₁ axis.
- **`rwd_go_thresh`** (new: response-window go hinge threshold decoupled from the hold) +
  **`sweep_r2cuego`** (lick ≥ 2 in-cue, hold ±1 one-sided, nogo free, GNG only): the network parked
  the GO HOLD at +1.66…+1.84 and let the cue idle — push go +0.1, nogo +0.32…+0.39 (vs +0.30),
  Wi[:,4] unchanged, drive decomposition unchanged. One-sided holds are free above θ, so "lick above
  hold" degenerated into "hold above lick threshold". SIDE FINDING: inflating the κ₁ hold damaged
  pure DPA at gain 1 (choice 0.65–0.95, unpair side; κ₁ delay rest +0.26…+0.52; s2 mem DECAY, leak
  0.67) — AMPLITUDE is a second route to the same m₁κ₁ saturation that gain 2 gives.
- **`gng_hold_pin`** (new: hold scored two-sided at ±θ) + arm `cuegop` — implemented, unit-tested,
  NOT launched. Leon: a go response is correct whenever ≥ 1 (a lick is a threshold event), so
  pinning is the wrong shape. The real issue: on a single κ₁ axis a held go rule ≥1 already IS a
  lick-level state; "lick at the cue" is undefined without an upper bound on the pre-cue state.
  **Open fork:** (1) premature-lick ceiling — one-sided hinge from above on go trials before the
  cue (κ₁ ≤ 1+δ), lick ≥ 2 in-cue; or (2) rank 3 — rule on κ₁, lick on κ₂ (`target_rank=3`, a new
  foundation). Decision pending.
  **RESOLVED 2026-09-07 (Leon): rank 2 only, and no engineered constraint on the loss** — see §27h.
  `gng_hold_ceiling` (the premature-lick ceiling, arm `cuegoc`) was implemented, then judged the wrong
  question: the cue push is an INPUT property (naive GNG-task nogo push +0.25 at cue 1 → +0.65 at
  cue 2), so a loss term that forces a push the amplitude gives for free only buys distortion. It
  stays in the tree default-off, unused.

**27h. ★★ THE SAFEGUARD ARM — `sweep_r2sgd2` (2026-09-07): the no-lick rule seats the wells AT the
line, not below.** Leon's framing, which is now the standing rule for this line: "prevent the network
from being in the κ₁>0 region when it should not; the pushes (go stimulus, cue on both trial types)
are the task's and are never traded against; the network should tune the location of the wells."
Implemented as ONE one-sided hinge κ₁ ≤ 0 (relu², weight 1, threshold 0) wherever a lick would be
wrong and nothing else: nogo rows from CUE ONSET → test-on (new flag `nolick_nogo_in_cue`; rows
classified from their own hold target in the gng span, needs `dual_gng_memory`), DPA rows the whole
delay (`nolick_full_delay`), go rows FREE after the cue (Leon: a post-cue stop-lick on go is a
transience demand on the go attractor — the decay-pin lineage — and would confound attribution).
= cue2 + those flags; nocue DPA ckpts; GNG retrained but the term is inert there at dose 2 (nogo
lands at −0.4 at cue-off): naive ≡ cue2's to two decimals. 4/4 DONE; gallery `rnn/sweep_r2sgd2`.

| Dual task, expert (mean κ₁; fraction of steps > 0) | cue2 | sgd2 |
|---|---|---|
| nogo pre-cue hold | −0.92 … −1.06 | −1.04 … −1.10 |
| cue push on nogo | +1.25 … +1.35 | +1.11 … +1.21 |
| nogo at cue-end | +0.20 … +0.37 (.73–.89) | **+0.03 … +0.11** (.55–.65) |
| nogo late delay | +0.39 … +0.61 (.94–1.00) | **+0.17 … +0.20** (.76–.81) |
| DPA rows, whole delay | +0.07 … +0.22 (.60–.76) | **−0.02 … −0.07** (.42–.51) |
| go hold pre-cue | +0.98 … +1.03 | **+0.81 … +0.95** (below θ) |
| pairing by-gng · retention · memory | 1/1/1 · 0.99–1.00 · HELD (s2 GROW) | 1/1/1 · 0.99–1.00 · HELD 4/4 |
| flow_verdict (+κ₀ FP κ₁) | 0/4 (+0.27/+0.48/+0.32/+0.68) | 0/4 (+0.18/+0.75/+0.20/+0.30) |

- The push is intact and the nogo hold did not deepen: the network did NOT trade against the task's
  pushes. It relocated the LANDING: the cue-end point dropped ≈0.25 and the DPA rows sit exactly on
  the line, straddling it half the time. That is the relu² hinge at 0 (§25e): no force once κ₁ ≤ 0,
  so the wells equilibrate at 0⁻ and never acquire depth. The nogo rows are the residual violator,
  on BOTH memory sides equally (A +0.15…+0.26, B +0.10…+0.23): after the cue they drift from +0.05 up
  to +0.2 — the §27e′ upper plateau, still there a few tenths lower (slow-manifold figure
  `slow_manifold_sgd2.png`: the A-nogo path parks at (+1.1, +0.2)). The lower arc at κ₁≈−1.5
  connecting the two memory sides is now prominent in s0–s2 (the A↔B drift channel flagged in §27e;
  sep still 1.00). The +κ₀ slow set reaches +2.2…+2.5 (go structure).
- The cost Leon predicted appeared on the GO side, not in pairing: on one κ₁ axis the wells and the
  go rule move together, so the go hold is below θ in every seed (gng_pos residual 0.017–0.085).
- ⚠ NOT CONVERGED: Dual val loss 0.11–0.16 at 300 (cue2 0.07–0.10), still falling, nolick the largest
  residual in 3/4. A snapshot of a descent. `sgd2x` (continue from sgd2's expert ckpts, +300 epochs,
  identical loss) is in sweep.py, NOT launched. Epochs note: every foundation Dual loss is still
  falling at 300 (0.36–0.52 @100 → 0.07–0.10 @300); stop_loss 0.005 never fires at noise 1.0.
- "Impose no-lick in the early delay too?" — measured, not needed: the early delay (sample-off →
  go/nogo on) already sits at −0.05…−0.12 in every seed, even in cue2 (−0.02…−0.07); the only
  violating epoch is the post-cue nogo. A hinge there has no gradient to give.
- Tool: `$CLAUDE_JOB_DIR/tmp/nolick_split.py` (per-class + A/B split of κ₁ on the Dual task, mirrors
  `traj_verdict.probe`); `traj_verdict` tags `nolick-nogo-from-CUE-ON`.

**27i. `sweep_r2sgd2g` (2026-09-07): the go post-cue stop-lick is a measured NEGATIVE — settled.**
Leon asked to try it after all: = sgd2 + `nolick_late_delay` (go rows hinged cue-off → test-on; GNG
stage cue-off → end), both stages. 4/4 DONE; gallery `rnn/sweep_r2sgd2g`.
- **GNG stage breaks retention: `after_gng/dpa` 0.69 / 0.92 / 0.90 / 0.36** (sgd2 1.00 ×4); memory
  DECAY ×3 (hold 0.47/0.79/0.75), **FLIP in s3** (hold −0.41, sep 0.27, DPA 0.34 below chance — the
  §26 inversion); go rule made TRANSIENT (relax 0.02–0.35), go hold +0.67…+0.88 < θ. Mechanism = the
  §27a entanglement: g·n₀ᵀm₁/N = −2.5 … −3.6 at naive (sgd2 −0.01…−0.64; every foundation arm
  ≤ 0.7 in magnitude), m₁ restructured 39–61 % (cos 0.83–0.87). The decay-pin lineage reappearing at
  GAIN 1 from a task-shaped term.
- **Dual rebuilds the memory (dpa 0.995–1.00 — the primary-metric trap) and nothing improves:** go
  late delay +0.60…+0.75 with 100 % of steps > 0 (the hinge is simply unmet), DPA rows +0.02…+0.07
  (WORSE than sgd2), nogo late +0.08…+0.19, go hold +0.58…+0.70, mem GROW s1/s2, flow_verdict 0/4
  with NO PAIR in s0/s2 (single +κ₀ well + a κ₀≈0 basement) and 4 wells in s3; Dual val loss
  0.27–0.42 at 300 (3× sgd2), nolick residual 0.09–0.15. Go rows stay free after the cue.

## §28 — Transfer-function control: the foundation with relu is a RUNAWAY, and a loss cannot make it a well (2026-09-08/09)

**28a. Review of the implemented φ.** Eight functions in `src/models.py`, numpy mirrors in
`src/flow_field.py` verified to machine precision incl. φ′: tanh, erf, relu, softplus, elu,
**lif = Φ(x) = ½(1+erf(x/√2))** (the foundation's φ, gain 1: range (0,1), REST RATE 0.5, slope 0.40
at rest, φ′→0 at both ends), lif_sc (Φ rescaled, slope 1), tanh_asym (+γ tanh²). `lif` is the
Gaussian CDF: the exact mean-field rate of binary/threshold units with Gaussian input (van Vreeswijk
& Sompolinsky), the high-noise erfc limit of the LIF Siegert rate (Amit & Brunel 1997, Brunel 2000),
the probit. "lif" is a slight misnomer. Consequences: the chaos line "gain·λ = 1" in
`architecture.md` assumes φ′(0)=1 and is off by 0.4 for lif; saturation at BOTH ends is the
gain-steal mechanism (cue drive +0.65 at rest vs +0.21 at the nogo hold); sign in κ comes entirely
from n (rates ≥ 0). **`nonlinearities.md`'s old conclusion ("LIF cannot encode B as −κ₀") was STALE**
— every foundation seed has wells at κ₀ ≈ ±1.2; corrected there. Literature check (Barbosa): his
2026 Nat Commun model is a SPIKING LIF bump attractor (Brian2, AMPA/NMDA/GABA + Mongillo STP); his
low-rank RNN work (2023) uses tanh; the 2026 multi-area lrRNN work has no located preprint.

**28b. `sweep_r2fdrelu` — the foundation with relu (one-field delta, DPA retrained), 4/4, gallery
`rnn/sweep_r2fdrelu` (κ runs to ±20).** DPA acc 1.000 but NO memory well: κ₀ +2 at sample-off →
+28…+41 at test-on, mean unit rate 16–22 (max 220–330; lif 0.5/1.0), autonomous F₀/κ₀ → +0.25…+0.37
at large κ₀ (lif: zero crossing at κ₀≈1.25, → −1 beyond). **Why (exact):** relu is positively
homogeneous, so on the memory axis the active set depends only on sign(m₀ᵢκ₀) and the field is
LINEAR on each half-line, F₀ = (λ⁺ − 1)κ₀ with λ⁺ = g·Σ_{m₀ᵢ>0} n₀ᵢm₀ᵢ/N — verified: λ⁺ = 1.25–1.37
from the weights equals the measured asymptotic slope to 3 decimals. A linear field has one fixed
point (the origin): λ⁺<1 decay, λ⁺>1 escape, λ⁺=1 a measure-zero line attractor. The structured init
gives λ⁺≈1.5, the one-sided hinge (κ₀ ≥ 1, free above) never prices growth (DPA loss 0.08 vs lif
0.23 — margin is free), and the linear pairing readout works at any amplitude. The tonic attention
input breaks homogeneity only near the origin (F₀/κ₀ +0.05…+0.23 at κ₀≈0.5–1). **In this loss the
well exists because φ saturates.** GNG: `after_gng/dpa` 0.58/0.64/0.61/0.69, mem GROW ×3 + LOST ×1,
pairing 0.58–0.67 — WITHOUT coupling growth (n₀ᵀm₁ within ±0.4): retraining m₁/n₁ changes the active
set, hence the κ₀ growth rate, hence the test-time amplitude the non-scale-invariant readout was
calibrated on. Dual rebuilds both tasks to 1.00 on a still-runaway substrate (F₀/κ₀ → +0.4, rates
to 190). Not a comparison substrate.

**28c. relu + ACTIVITY L2 (`rr01` w=0.01 · `rr1` w=0.1 · `rrb01` w=0.01 + trainable unit biases),
DPA stage only, 2026-09-09, 4 seeds each, galleries `rnn/sweep_r2rr01` etc. + the comparison figure
`field_profiles_relu_vs_lif.png` (`scratchpad/relu_field_profiles.py`).** New machinery:
`rate_reg_weight` → `Optimization(rate_reg=w)` adds w·⟨rates²⟩ to train AND val loss (⟨r²⟩ printed
per epoch); `max_val_loss` is now a RunConfig field (default 100 — at the unstable relu init
⟨r²⟩≈1e4, so the penalty alone aborted the first launch at epoch 1; these arms use 1e6).

| DPA ckpt, DPA task | fdrelu | rr01 | rr1 | rrb01 |
|---|---|---|---|---|
| DPA acc | 1.00 | 1.00 (s3 .92) | **0.50–0.55** | 1.00 (s3 .88) |
| κ₀ sample-off → test-on | +2 → +28…41 | +1.0…1.3 → +2.6…3.4 | +0.6…1.0 → +0.7…0.9 (DECAY) | +1.0…1.8 → +2.3…2.8 |
| ⟨r²⟩ (max unit rate) | 1200–4400 (1500–2900) | 14–25 (190–330) | 2–5 (110–240) | 17–22 (180–340) |
| λ⁺ | 1.25–1.37 | 1.09–1.30 | 1.06–1.34 | 1.12–1.33 |
| F₀/κ₀ zero crossing + → − | none | none | none | none |

- **Bounded ≠ attractor.** w=0.01 slows the escape (15× → 2.5× over the delay) but λ⁺ stays > 1 and
  F₀/κ₀ > 0 for all κ₀ ≥ 1 — still an escape. w=0.1 kills the task: the profile is negative near
  the origin and positive beyond κ₀≈0.5–1, i.e. the tonic input + penalty made a REPELLER (a
  threshold), the state parks below it and decays, pairing at chance. The L2 is on the MEAN, so a
  few units still fire at 200–300. Trainable biases changed nothing: the mechanism that would give
  relu a genuine well (active set shrinking with amplitude ⇒ λ_full > 1 > λ⁺, the threshold-linear
  bump) needs λ⁺ < 1, and no term asks for zero growth AT the target amplitude — the one-sided hinge
  still pays for margin. Open (not run): a two-sided memory pin |κ₀| ≈ θ, a weight-side constraint
  on the half-population overlap, or EI inhibition — each engineers the bound Φ provides for free.

**28d. The terminal A/B hold window (`dpa_hold_window`, 2026-09-09) — absorbed by Φ, inert for
relu.** Leon: "run relu but where the hinge target for dpa is ±1 only during the last 0.5 s of late
delay — that would be consistent with the gng targets." He is right that the A/B memory was the odd
one out: the GNG identity hold is a 0.25 s window ENDING at cue onset (`generate_gng_trials`,
`hold_full_delay=False`), while the memory was supervised from SAMPLE ONSET to test onset (6 s, the
sample included). New field `dpa_hold_window` (seconds; 0 = legacy) → `generate_dpa_trials(hold_window=)`;
DPA stage only — the Dual generator sets no κ₀ target at all (`mem_*` ≡ 0 there).
The argument for it: the legacy span demands |κ₀| ≥ θ *already during the sample*, so it prices the
RISE TIME as well as the amplitude, and self-amplification (λ⁺>1) is the cheapest way to cross 0→θ
inside a 1 s sample; a terminal window prices only "be there when it is read", which should make a
subcritical λ⁺<1 solution affordable. **Half right.** Arms (DPA stage only, 4 seeds, `stop_loss` 0.1):
`w5relu` = fdrelu + window · `w5rl2` = + activity L2 0.01 · `w5lif` = nocue + window.

| DPA ckpt, DPA task | fdrelu | w5relu | rr01 | w5rl2 | nocue | w5lif |
|---|---|---|---|---|---|---|
| λ⁺ | 1.25–1.37 | 1.31–1.39 | 1.09–1.30 | 1.11–1.30 | 3.19–3.54 | 3.02–3.76 |
| F₀/κ₀ zero crossing | none | none | none | none | κ₀ 1.19–1.23 | κ₀ 1.06–1.14 |
| κ₀ 3 s → test (ratio) | ×11–15 | ×15–20 | ×1.9–2.2 | ×1.9–2.3 | ×0.98–1.01 | ×1.25–1.51 |
| dpa acc | 1.00 | 1.00 | 1.00 (s3 .92) | 1.00 (s3 .94) | 1.00 | 0.99–1.00 |

- **The epoch-matched test is `w5rl2` vs `rr01`** (both ran the full 250; the relu-only arms
  early-stopped at 100–195 under `stop_loss` 0.1 — a confound introduced by that change, since
  fdrelu's DPA val was ~0.08 and would also have stopped early). The window changes NOTHING there,
  seed by seed. For relu it is inert: λ⁺ > 1, no crossing, escape at every amplitude.
- **Why the prediction failed, precisely.** A supervision window controls what is CONSTRAINED, not
  what is INCENTIVISED. Under a one-sided hinge, growth satisfies every window equally, and the
  pairing readout at test actively rewards a larger κ₀ (free d′). Removing supervision from the delay
  removed the last term that looked at the delay at all, so λ⁺ floated — λ⁺<1 got no cheaper, its
  competitor got cheaper too.
- **Φ absorbs it and demonstrates the intended dynamical effect:** wells intact, rates unchanged
  (⟨r²⟩ 0.4 / max 1.0), dpa 0.99–1.00 — but κ₀ is no longer clamped from the sample on, it now RISES
  INTO the well across the delay (0.67–0.83 at 3 s → 0.93–1.09 at test; nocue is flat at ×0.98–1.01).
  Strictly less painted supervision at no cost. Whether it should become the foundation default is
  open — it changes the baseline every arm builds on.
- Tools: `scratchpad/w5_table.py` (the §28c table + F₀/κ₀ at κ₀=1,3 so a near-origin crossing cannot
  be misread as a memory well), `scratchpad/w5_field_profiles.py`.

## §29 — The INITIAL CONDITION: λ⁺=1 attracts training from both sides; Φ doesn't care; the subcritical memory is born ENTANGLED (2026-09-09/10)

**29a. Leon's challenge: "is that not an initial condition problem?"** — and it was. Every relu arm
through §28d used the foundation's `memory_lambda`=3.0. The structured init splits units symmetrically,
so **λ⁺ = g·λ₀/2 exactly at init** (measured, 4 seeds): memory_lambda 0.8/1.2/1.6/1.8/2.0/2.5/3.0 →
λ⁺ 0.40/0.60/0.80/0.90/1.00/1.25/1.50. So all 20 relu seed-runs STARTED at λ⁺ = 1.50, deep in the
escape regime; training walked λ⁺ down (→1.25–1.39 bare, →1.09–1.30 with L2) and stalled just above 1.
That is a barrier signature, and nothing had ever been run on the other side of it.
The homogeneity argument (F₀ = (λ⁺−1)κ₀, origin the only fixed point) holds only for a STRICTLY
homogeneous net; `attention_gated=True` puts a tonic input on for exactly the delay, and a subcritical
net with a constant drive has a stable FP at κ* = c/(1−λ⁺) ≠ 0 — with different λ±, c± on the two
half-lines, two input-sustained fixed points are possible. So "a loss cannot make relu a well" (§28c)
was overstated; the untested question was reachability.

**29b. `sc12`/`sc16`/`sc18` (relu, λ⁺ init 0.60/0.80/0.90, terminal window, DPA only).** Training
climbs from BELOW to λ⁺ 0.99–1.02 / 1.06–1.08 / 1.09–1.12. **λ⁺ = 1 is an attractor of the TRAINING
dynamics from both directions**, not a barrier. The subcritical start is a large practical win — max
unit rate 35–103 (w5relu 2000–3900), κ₀ growth ×2.8–4.0 (×15–20), ⟨r²⟩ 18–49 (3100–7700), dpa 1.000 —
and the field profile is no longer flat: it DECAYS with amplitude toward a small positive asymptote
(sc12 → +0.02, essentially marginal). But no zero crossing: still no well, and the input-sustained
FP did not appear (a settled state would give ratio ≈ 1). ⚠ ALL TWELVE stopped at epoch 10–15
(`stop_loss` 0.1 fired; subcritical + terminal window makes the loss trivially easy) — a snapshot,
not an equilibrium. The `sq*` rerun at `stop_loss` 0.005 is coded and NOT launched.

**29c. `scl12`/`scl16`/`scl18` (Φ, λ₀ init 1.2/1.6/1.8, DPA only): the foundation is init-INDEPENDENT.**
Measured at init, the pitchforks differ: **relu's is at memory_lambda 2.0** (λ⁺=1), **Φ's at ≈2.9**
(its slope at rest is 0.40 and the tonic input sits the operating point further onto the flat part) —
so the foundation's 3.0 is only BARELY supercritical (F₀/κ₀ = +0.042, wells at init κ₀* ≈ 0.03) and
1.2/1.6/1.8 are subcritical for both φ. Result: training climbs λ₀ 1.2 → 4.5–6.9, through the
pitchfork, and rebuilds the foundation's well pair in **12/12 seeds** (κ₀* ±1.04–1.15 vs nocue
±1.19–1.32), same rates, dpa 0.99–1.00. For Φ the crossing is an ordinary supercritical pitchfork —
wells grow continuously past it and saturation bounds them, a smooth uphill the whole way. relu has
no such thing, which is why it parks ON λ⁺=1 instead of passing through.
⚠ Two measurement traps found here, both of which cost a wrong first reading: (i) the 1-D profile
along κ₁=0 misses wells that sit OFF that axis — `s1_scl12` reads "no well" but has both, at
(+1.06,−0.58)/(−1.09,+0.48); use the 2-D `find_wells`. (ii) With a tonic input F(0) = c ≠ 0, so the
origin is not a fixed point and F₀/κ₀ ≈ c/κ₀ diverges near 0 — the near-origin sign is an offset
artifact, NOT a rest-state stability readout. No seed in 20 has a near-origin attractor.

**29d. ★ The subcritical memory is born ENTANGLED — and that, not subcriticality, is what costs.**
Full sequence on the λ₀=1.6 solution (`sclf16` = GNG 100 + Dual 300 from `scl16`'s DPA ckpts,
`stop_loss` back to 0.005 — at 0.1 the GNG stage would stop at ~epoch 25, since nocue's GNG val
crosses 0.1 between epoch 10 and 50, and `after_gng/dpa` would be measured on a half-trained net).

| | `after_gng/dpa` | `after_gng/gng` |
|---|---|---|
| nocue (λ₀ 3.0, no cue) | 0.996–1.000 | 1.000 |
| cue1 / cue2 (λ₀ 3.0) | 0.987–1.000 | 0.943–0.984 |
| **sclf16** (λ₀ 1.6, no cue) | **0.886 / 0.939 / 1.000 / 0.929** | 1.000 |
| **sclc2** (λ₀ 1.6, cue 2) | 0.886 / 0.939 / 1.000 / 0.929 | **0.802 / 0.700 / 0.988 / 0.958** |
| **sclnl** (+ no-lick on nogo) | **0.802 / 0.761 / 1.000 / 0.946** | **0.981 / 0.972 / 0.991 / 0.982** |

- **`sclf16`: retention degrades, and it is NOT memory loss.** `traj_verdict` GNG stage: memory
  HELD ×2 / GROW ×2, separation 1.00 → 0.94–1.00, no FLIP, no DECAY. What fails is `leak`
  (0.91/1.43/0.20/0.41), with κ₁ pushed in OPPOSITE directions by A and B (s1: A −0.80, B +0.64);
  the pairing readout lives on κ₁, so `choice@0` (0.90/0.93/1.00/0.93) tracks the accuracy exactly.
- **The cause is visible in the DPA solution, before GNG runs.** Memory-well tilt off the no-lick
  axis, mean |κ₁|: nocue 0.11 · w5lif 0.17 · scl16 0.31 (s0 0.34, s1 0.49, s2 0.12, s3 0.29), the
  wells ANTI-SYMMETRIC (+κ₀ below the line, −κ₀ above). Equivalently, and this is the project's
  standard diagnostic: **g·n₀ᵀm₁ at the DPA ckpt = −2.18 / −3.76 / +0.28 / −0.61** (s0/s1/s2/s3) —
  s0 and s1 are already at the magnitudes flagged as pathological (foundation ≤0.4; the broken
  `sgd2g` arm −2.5…−3.6) BEFORE GNG starts. Tilt and coupling are one thing, two views: a well
  displaced off κ₁=0 IS a rank-0 mode carrying a rank-1 component.
- **`sclc2` (cue 2): the cue exposes the tilt.** `after_gng/dpa` is bit-identical to sclf16 — forced,
  since the GNG stage is gradient-blind to cue amplitude (§27d) and the DPA probe has no cue window;
  it CANNOT move. The damage lands on `after_gng/gng`, in exact tilt order (tilt 0.12→0.988,
  0.29→0.958, 0.34→0.802, 0.49→0.700), while the same cue costs the foundation only 2–5 points.
  Mechanism: the cue pushes κ₁ up on BOTH trial types (+1.3 at dose 2); on an untilted pair that is
  symmetric and absorbed, on a tilted pair it drives the −κ₀ side deep into κ₁>0.
  ⚠ Statistics: ρ = −1.000 within the subcritical arm (n=4, p=1/24=0.042) but **ρ = −0.563 pooled
  with the foundation seeds, permutation p = 0.077** — suggestive, not established. The strongest
  single point is s2: tilt 0.12, and it scores 0.988, the highest of all eight.
- **`sclnl` (+ no-lick hinge on NOGO rows only, `nolick_nogo_in_cue`, weight 1, threshold 0; go rows
  free per §27i; DPA rows excluded — not `nolick_full_delay`): the causal test succeeds, one way.**
  GNG recovers to 0.97–0.99 across the board, in exact tilt order (Δ +0.00/+0.02/+0.18/+0.27), and
  `after_dual/gng` improves (0.48–0.82 → 0.58–0.94). **But the memory pays, in the same order:**
  `after_gng/dpa` Δ +0.00/+0.02/−0.08/−0.18. On one κ₁ axis, forcing nogo below the line drags the
  memory with it — the §27h cost, sharper.
- **★ The hinge does NOT relocate the wells, and does not disentangle.** Memory-well tilt at naive:
  0.38→0.41, 0.43→0.45, 0.13→0.13, 0.26→0.27 (marginally WORSE); coupling −0.99→−0.97, −1.54→−1.64.
  Leon's framing for the safeguard rule is that the network should relocate the wells to satisfy the
  constraint; measured here, it does not — it satisfies the constraint another way and bills the
  memory. Note this arm also breaks the GNG stage's gradient-blindness to cue amplitude (the nolick
  term IS supervised after cue onset), so `after_gng/dpa` is a real readout here, unlike in sclc2.
- **Where this points:** the coupling is born at the DPA stage, so the lever must act there, not in
  GNG. Open: (i) more seeds — how often does a subcritical init land clean (s2-like, +0.28) vs
  entangled? currently 1 in 4, the single most useful unknown; (ii) `w5lif` full sequence — still
  unrun, and the confound that `sclf16` differs from nocue in TWO things (`memory_lambda` 1.6 AND
  `dpa_hold_window` 0.5; the tilt ladder nocue 0.11 < w5lif 0.17 < scl16 0.31 suggests the window
  contributes part of it); (iii) the `sq*` relu rerun at `stop_loss` 0.005.
- Tools: `scratchpad/w5_table.py`, `scratchpad/sc_field_profiles.py`, `scratchpad/scl_field_profiles.py`,
  `scratchpad/tilt_vs_cue.py`, `scratchpad/nolick_nogo_effect.py`, `scratchpad/publish_gallery.sh`.

**29e. ★★ The κ₁ PIN: cutting the coupling at the DPA stage, where it is born (2026-09-10).**
§29d located the cause upstream, so the lever had to act there. `dpa_prelick_free=False` — the
LEGACY two-sided 0-pin, already in the tree, `targets[:, :n_on[1], -1] = 0.0` in
`generate_dpa_trials` — holds the readout κ₁ at 0 from trial start through the whole delay and
releases it only at test onset. Arm `k1zero` = scl16 + that one field, DPA RETRAINED (its loss
changes), then the full sequence (GNG 100 + Dual 300, `stop_loss` 0.005). Verified: κ₁ pinned over
t = 0…7.99 s, released at test onset 8.0 s (the foundation's pin ends at 1.98 s).

| DPA ckpt | scl16 (free κ₁) | k1zero (pinned) | `after_gng/dpa` sclf16 → k1zero |
|---|---|---|---|
| s0 | tilt 0.34 · n₀ᵀm₁ −2.18 | 0.11 · **−0.51** | 0.886 → **0.988** |
| s1 | 0.49 · −3.76 | 0.19 · **−2.56** | 0.939 → **0.534** |
| s2 | 0.12 · +0.28 | 0.02 · +0.28 | 1.000 → 0.994 |
| s3 | 0.29 · −0.61 | 0.06 · **−0.08** | 0.929 → **0.998** |

- **The leak is eliminated in 4/4** (0.91/1.43/0.20/0.41 → 0.21/0.08/0.29/0.05) — the exact failure
  mode of §29d. Three seeds reach foundation retention (0.988–0.998); the DPA stage is unharmed
  (`after_dpa` 0.996–1.000), so the pin does NOT fight the pairing readout, a risk that did not
  materialise.
- **★ The two coupling directions are not the same quantity.** The pin cuts **n₀ᵀm₁** (rank-1
  activity fed BACK INTO the memory) −2.18→−0.51, −0.61→−0.08, and barely touches **n₁ᵀm₀** (memory
  read by the decision mode) 0.87→0.85, 0.75→0.65, 1.08→0.82. The visible well TILT is the n₁ᵀm₀
  symptom; what actually costs retention is n₀ᵀm₁.
- **s1 is the control that proves it.** Its tilt fell to 0.19 like the others but n₀ᵀm₁ survived at
  −2.56 — the only seed above 1 — and it is the only one that LOST the memory (`traj_verdict`: hold
  +0.61→−0.00, sep 1.00→0.45, choice 0.54 = chance). Worse than with free κ₁, and mechanistically
  why: previously the memory carried a κ₁ component and sat in a tilted but stable geometry; pinned,
  it never builds that, so when GNG drives κ₁ for the rule the surviving pathway empties κ₀.
- ⚠ The flag's own docstring is right that a two-sided pin "clamps wells ON the line": it forbids the
  tilt but equally removes the freedom to sit BELOW κ₁=0, which is the project's actual goal. This
  is a DIAGNOSTIC that identifies the culprit, not the target geometry. The one-sided variant
  (`dpa_nolick_weight` with `dpa_prelick_free=True`) applies the same pressure while leaving κ₁<0
  free — NOT RUN, and the obvious next arm.

**29f. ★★ The 2×2: what the no-lick rule COSTS depends on how the memory was built.** Four cells,
all at cue 2, all loading a DPA ckpt so the RNG streams match (this matters — see the trap below).
Means over the three clean seeds (s0,s2,s3); s1 excluded, its n₀ᵀm₁ survived the pin.

| | `after_gng/dpa` | `after_gng/gng` |
|---|---|---|
| tilted wells, no hinge (`sclc2`) | 0.938 | 0.916 |
| tilted wells, + no-lick nogo (`sclnl`) | **0.916** (−0.022) | 0.985 (+0.069) |
| κ₁ pinned, no hinge (`k1zcue`) | 0.987 | 0.983 |
| **κ₁ pinned, + no-lick nogo** (`k1zcnl`) | **0.987** (−0.000) | **0.989** (+0.006) |

- On TILTED wells the hinge is a trade: +0.069 rule, −0.022 memory (and per §29d it neither moves
  the wells nor cuts the coupling — it treats the symptom). On the PINNED substrate **it costs
  nothing** and still adds a little. The §27h/`sclnl` cost was a property of the ENTANGLED substrate,
  not of the hinge. Once the memory is built right, the task's own no-lick contingency is free.
- The full recipe (subcritical init + terminal memory window + κ₁ pinned in DPA + cue + no-lick on
  nogo) reaches **0.987 retention / 0.989 rule**, vs the foundation with the same cue at
  0.987–1.000 / 0.943–0.984 — it MATCHES the foundation's memory and BEATS its rule, from a
  subcritical init. `nolick` residual 0.005–0.019 pinned vs 0.020–0.024 tilted: less to fix.
- **`k1zcue` also shows the pin makes the substrate CUE-ROBUST** (vs `sclc2`, RNG-matched): the cue
  that cost the tilted substrate up to 30 points of rule accuracy is absorbed — gng 0.802→0.979,
  0.958→0.986, and the gain scales with the tilt removed (s2 0.12: +0.00, s3 0.29: +0.03,
  s0 0.34: +0.18). Retention 0.886→0.964, 0.929→0.998.
- ⚠ **RNG TRAP, found by checking rather than assuming.** `sclf16` vs `sclc2` are BIT-IDENTICAL
  (max|Δweight| = 0), which is what licenses calling their equal retention "structurally forced"
  (§29d). But `k1zero` vs `k1zcue` differ by up to 2.45 in weights — NOT from the cue: `k1zero`
  TRAINED its DPA stage and consumed RNG while `k1zcue` LOADED it, so their GNG stages see different
  batches and noise. Never compare a trained-DPA arm with a loaded-DPA arm and attribute the
  difference to the field you changed; match the ckpt-loading status first.

**29g. Cue DOSE 1 and cue DURATION: the recipe is robust, and duration ≠ amplitude (2026-09-10).**
Two arms on the full recipe (κ₁ pinned in DPA + cue + no-lick on nogo), both reusing the cue-2
recipe's DPA checkpoints. New field `cue_duration` (seconds, default 0.5) widens the cue window in
the GNG and Dual timings; the cue ONSET is unchanged, so every loss window keyed to cue-on is
identical (`nolick_nogo_in_cue` still spans cue-on → test-on, the pre-cue rule hold still ends at
cue-on) and only cue-off moves — which matters solely for the gng response window and
`nolick_late_delay`, both off in this line. (First launch crashed: the bound assumed the cue always
has a following stimulus, false in GNG where the cue is last; now bounded by trial end.)

| arm | `after_gng/dpa` | `after_gng/gng` | GNG epochs |
|---|---|---|---|
| cue 2, 0.5 s (`k1zcnl`) | 0.967 / 0.670 / 0.999 / 0.996 | 0.984 / 0.966 / 0.988 / 0.994 | 100 |
| cue 1, 0.5 s (`pin_cue1_nolick`) | 0.998 / 0.706 / 1.000 / 0.986 | 0.995 / 0.999 / 0.995 / 0.999 | ~21–41 |
| cue 1, 1.0 s (`pin_cue1_long_nolick`) | 0.999 / 0.729 / 1.000 / 0.986 | 0.989 / 0.970 / 0.986 / 0.989 | ~21–41 |

- **★ Duration is a much weaker dose than amplitude.** nogo κ₁ push during the cue, naive ckpt, mean
  over seeds: cue 1 / 0.5 s **+0.11** · cue 1 / **1.0 s** **+0.19** · cue 2 / 0.5 s **+0.54**.
  Doubling the amplitude buys +0.43; doubling the duration buys +0.08, and the SECOND half-second
  adds less than the first (+0.11 then +0.08). The push SATURATES IN TIME: with τ = 0.3 s the state
  reaches its cue-driven equilibrium within a few hundred ms, so holding the cue longer keeps it
  where it already is rather than driving it further. Amplitude sets WHERE that equilibrium sits;
  duration only sets how long you sit there. The §27g gain-limit (drive +0.65 at rest vs +0.21 at
  the nogo hold) is therefore NOT bypassed by integrating for longer — a hypothesis this kills.
- Behaviour follows: the two cue-1 arms are matched on training length (same truncation, 6/8/5/4
  logged GNG epochs) and retention is identical to within noise (clean-seed mean 0.995 both), while
  the rule is marginally WORSE with the longer cue (0.996 → 0.988) — more of the trial spent above
  the lick line for the hinge to fight, with no compensating gain. **The recipe is insensitive to
  cue duration**, which is a robustness result even though the mechanistic hypothesis was refuted.
- ⚠⚠ **TRAP found by Leon reading the figure (2026-09-10): a config field that changes TASK TIMING
  must be threaded into `plot_sweep.py`, or the figures silently probe a different task than the one
  trained.** `plot_sweep` built `TIMINGS = make_timings(DT)` at module level and never read
  `cue_duration`, so every figure for this arm regenerated its own trials with a 0.5 s cue while the
  model had been trained with a 1 s one — the plotted cue-driven activity lasted 0.5 s, which is
  exactly what Leon spotted. Verified directly afterwards: the training input carries the cue over
  t = 5.98–7.00 s (1.01 s) and the plotting input over t = 5.98–6.48 s (0.49 s). The ACCURACIES were
  never affected (`run_single` passes its widened `dual_timing`/`gng_timing` to the evals) and neither
  was the push measurement (that script widened the timing itself) — only the figures, and ALL of
  them for that arm, since the fixed-point and accuracy-by-trialtype panels also generate their own
  trials. Fixed: `plot_sweep._timings_for(meta)` widens `TIMINGS` by the run's `cue_duration`,
  `RunMeta` reads the field from the config, and all seven call sites go through it — no bare
  `TIMINGS[...]` lookup survives outside the helper. Figures re-rendered and republished.
  **The general lesson:** checking that the training LOG reports the right timing is not sufficient;
  check that the INPUT in the figure matches the input in training.
- ⚠ **The cue-1 vs cue-2 comparison is CONFOUNDED and must not be read as a dose effect.** The cue-1
  arms use `stop_loss` 0.1 (Leon's request), which fired in every seed and truncated their GNG stage
  to ~21–41 of 100 epochs (final GNG val 0.087–0.098); the cue-2 recipe ran the full 100 at 0.005.
  Less GNG training means less opportunity to disturb the memory — the very interference
  `after_gng/dpa` measures — so the apparent +0.008 retention gain at dose 1 cannot be attributed to
  dose. Dual was unaffected (its val floor is 0.10–0.18, so it ran 300/300). The clean control —
  cue 1 at `stop_loss` 0.005 — is NOT RUN.

**29h. Two measurement corrections, and what the network actually does at test (2026-09-10/11).**
Leon, reading the trajectories: *"instead of pushing the autonomous wells down, the networks adjust
the effect of the test odors (vertical flows) to compensate for the go/nogo perturbation."* He was
right, and getting to that took two wrong turns worth recording so they are not repeated.

- **★ THE COMPENSATION IS REAL, and it is φ′ gating.** Split by go/nogo condition at the expert ckpt,
  Dual task: the go/nogo state shifts κ₁ at TEST ONSET by up to 0.83 across conditions, the
  test-driven displacement moves the OPPOSITE way by nearly as much, and the landing point in the
  pairing window is held constant to within 0.02–0.28 — **66–87 % of the perturbation is cancelled**.
  s0 unpair is clearest: offsets +0.29/+0.54/−0.30 (none/go/nogo) → displacements −1.50/−1.67/−0.99
  → landings −1.20/−1.14/−1.28. `Wi` is FROZEN during Dual, so the test-odour weights cannot change:
  the condition-dependence must come from φ′ gating — the state sits at a different operating point
  on go vs nogo trials, so the SAME input produces a different κ₁ excursion. That is the §25d/§27g
  gain-steal, here working for the network. Exception: s1 unpair, 7 % cancelled (the seed whose
  n₀ᵀm₁ survived the pin).
- **⚠ WRONG TURN 1 — a circular measurement.** My first "confirmation" measured κ₁(test-off) −
  κ₁(test-on) on pure-DPA rows and reported it growing ×2 from naive to expert. That quantity IS the
  trained pairing readout, and the Dual stage is the stage that trains it (±0.7 untrained → ±1.2 at
  target ±1). It grows by construction and tests nothing. The condition-split above is the
  measurement that actually distinguishes the hypotheses.
- **⚠ WRONG TURN 2 — a bogus gauge "correction".** I then divided κ₁ by ‖n₁‖, arguing that
  n₁ → c·n₁, m₁ → m₁/c leaves W_rec = m₁n₁ᵀ/N and the dynamics invariant while scaling κ₁. The
  transformation is real BUT THE GAUGE IS ANCHORED BY THE LOSS: targets are at κ₁ = ±1, and the
  supervised pairing window measures ±0.72 (naive, untrained) → ±1.22 (expert), i.e. it sits at the
  target. Dividing by ‖n₁‖ therefore removes a real effect. **Do not normalise κ by ‖n‖; the task
  pins the scale.** (‖n₁‖ doubles 87→192 across Dual, so the temptation is strong.)
- **★ WELL SELECTION: always take the attractor NEAREST THE OCCUPIED STATE, never the largest κ₀.**
  Selecting by max κ₀ picked, in s2, an upper attractor at (+1.02,+0.73) that the network does not
  use — it sits on a lower one at (+1.01,+0.23), split from it by a saddle at +0.54 — and produced a
  spurious "the state never reaches its well, 46 s approach" reading. The local Jacobian at the
  OCCUPIED attractor is fast (τ_slow 0.5–1.0 s against a 5 s delay), and the state sits on it to
  within d ≈ 0.00–0.03. Report d alongside, and exclude/flag anything with d > 0.2.

**29i. Pricing the memory wells: the hinge asymptotes AT the line, at any weight (2026-09-11/14).**
Nothing in this line had ever priced the memory wells' κ₁ during the delay — `nolick_nogo_in_cue`
covers nogo rows from cue onset, `nolick_full_delay` (DPA rows, whole delay) was False in EVERY arm,
and the DPA-stage κ₁ pin is two-sided at 0, which clamps them ON the line by construction. Adding the
DPA-row hinge (`nolick_full_delay=True`) and dosing it:

| `nolick_weight` | occupied-attractor κ₁ | below the line | after_gng/dpa |
|---|---|---|---|
| nogo rows only | +0.338 | 2/8 | 0.967/0.670/0.999/0.996 |
| 1 | **+0.074** | 2/7 | 0.967/0.670/0.999/0.996 |
| 3 | **+0.032** | 3/8 | 0.966/0.688/0.999/0.996 |
| 5 | **+0.001** | 2/6 | 0.980/0.705/1.000/0.996 |

- The hinge moves the attractors monotonically and TIGHTENS the spread (w=1 spans −0.11…+0.82,
  w=3 spans −0.16…+0.20 with all eight located), and it is FREE — retention and rule flat or better
  at w=5. But μ ∝ 1/w decaying to ZERO: extrapolated, even w=20 sits at ≈+0.004. **Weight buys
  tightening, not depth.** §25e confirmed on a new substrate and in a stronger form.
- My force-balance prediction (−0.15 at w=3, −0.27 at w=5, from the noise-smoothed hinge gradient
  2[μΦ(μ/σ)+σφ(μ/σ)] with a CONSTANT opposing force) is falsified. The opposition instead grows
  steeply as μ → 0⁻ and scales with the push — and it costs nothing behaviourally, so it is
  structural rather than a trade-off. What supplies it is NOT identified; candidates are the go
  rule's +1 hold sharing the κ₁ axis in rank 2, and the pairing swing being cheaper from nearer 0.
- Also settled: delaying the pairing readout would NOT fix this (Leon's instinct, confirmed by
  argument). The wells are a DELAY-period property; the pairing decision is a CONJUNCTION (A_C pair
  vs A_D unpair share a memory well) so the well's κ₁ cannot carry it. Making the decision persist
  prices the decision states, not the wells.
- Depth therefore needs a displaced threshold (`nolick_thresh`>0) — which engineers the answer and
  is ruled out by [[feedback-safeguard-rules]] — or removal of whatever supplies the upward force.

## §30 — The integration scheme: dt is not negotiable, and the two-filter cascade is load-bearing (2026-09-14)

**30a. New flag `integrate` ("both" | "rates" | "rec").** The model integrates TWO variables and the
external input is NOT one of them (`src/models.py`):
`rec_inputs ← e^(−α_rec)·rec_inputs + (1−e^(−α_rec))·W_rec·rates` (τ_rec = 0.225 s), then
`rates ← e^(−α)·rates + (1−e^(−α))·φ(g·(input + rec_inputs))` (τ = 0.3 s), with `input_drive` added
inside φ at full strength on the step it arrives. So recurrent drive passes TWO cascaded filters,
external drive ONE — stimuli act faster than recurrence by construction.
The literature standard is a SINGLE filter: current-based `τ ẋ = −x + W·φ(x) + I` (Mante 2013,
Song/Yang/Wang PyCog, Yang 2019, Mastrogiuseppe & Ostojic 2018, Dubreuil 2022) or rate-based
`τ ṙ = −r + φ(W·r + I)`. Two-filter cascades belong to the biophysical line (Wang 2002, Wong & Wang
2006) and to NeuroFlame, which is where ours came from. `integrate` selects among all three;
**"both" is verified BIT-IDENTICAL to the pre-flag model** (max|Δ| = 0 on readout/rates/currents
against a reference captured beforehand), so nothing already measured moves.
⚠ `tau_rec_frac` CANNOT remove the synaptic filter: α_rec = dt/τ_rec = (dt_base·frac)/(τ·frac) =
**dt_base/τ**, independent of the frac — which only scales dt (and hence α). To change τ_rec at fixed
dt, change `tau`. ⚠ Noise is injected into the recurrent current; it is now passed explicitly to
`update_dynamics` because in "rates" mode `rec_inputs` is overwritten each step and a caller-preadded
noise term would be silently discarded.

**30b. ★ dt IS NOT NEGOTIABLE — and the reason is not numerical accuracy.** Two independent tests:
- **Out-of-distribution** (same trained weights, re-simulated at 2dt): the pairing amplitude drops
  30–40 % (1.23→0.73, 1.28→0.89, 1.30→0.75) and the delay κ₁ moves by up to 0.27 — larger than the
  entire weight-ladder effect of §29i.
- **Retrained from scratch at dt 0.045 (α_rec 0.20, the literature-standard value)**: the DPA stage
  still learns (0.957–0.997) and the GNG rule still learns (0.93–0.99), but **retention collapses**,
  `after_gng/dpa` 0.504/0.735/0.645/0.708 vs 0.967/0.670/0.999/0.996 at fine dt (clean-seed mean
  0.987 → 0.619). Wall clock 24 min vs ~65 — the 2.7× is real and not worth having.

**★★ THE MECHANISM, and the best evidence in this thread for §29d's coupling story.** The coarse step
does not merely integrate the same solution less accurately — it produces a DIFFERENT DPA solution,
**born far more entangled**: `g·n₀ᵀm₁` at the DPA ckpt −12.04 / +0.17 / −7.42 / −2.00 versus
−0.51 / −2.56 / +0.28 / −0.08 at fine dt, against a foundation that stays within ±0.4. Retention
tracks it seed by seed **including the reversal**: s1 is the only seed whose coupling improved
(−2.56 → +0.17) and the only one whose retention improved (0.670 → 0.735). Every earlier test of the
coupling→retention link was a within-arm correlation on n=4; this is an INDEPENDENT manipulation,
with nothing to do with κ₁ or the no-lick rule, moving the coupling in both directions and retention
following. Much harder to explain away.

**30c. The three modes compared (4 seeds each, full sequence retrained, tau_rec raised via `tau`).**

| mode | filtered | after_DPA/dpa | after_gng/dpa | after_gng/gng |
|---|---|---|---|---|
| "both" cascade | both | 0.994–0.999 | 0.967/0.670/0.999/0.996 | 0.984–0.994 |
| "rec" τ_rec 0.225 | current | **1.000/0.511/0.496/0.744** | 0.261–0.736 | 0.996–1.000 |
| "rec" τ_rec 0.300 | current | **1.000/0.511/0.997/0.752** | 0.486–0.990 | 0.997–1.000 |
| "rates" τ 0.300 | rates | 0.799/1.000/1.000/1.000 | 0.474/0.900/1.000/0.896 | 0.811–0.963 |

- **"rec" fails to LEARN the delay memory** in 2–3 of 4 seeds (chance after DPA) while learning the
  rule perfectly every time. Raising τ_rec 0.225 → 0.300 helps (Leon's prediction) but does not
  rescue it. ⚠ Note the FIXED POINTS ARE IDENTICAL ACROSS MODES — steady state is
  rates* = φ(g(I + W·rates*)) either way — so the landscape available is the same and this is purely
  a LEARNABILITY result about what BPTT can find.
- **Hypothesis for why (not yet tested):** noise is injected into the recurrent current, and "rec" is
  the only mode where it reaches the readout UNFILTERED (rates follow φ instantaneously). A
  first-order filter at α = 0.075 cuts readout noise ≈5×. Prediction: "rec" recovers at ~1/5 the
  input noise. One field, 4 seeds, NOT RUN.
- **"rates" restores memory learnability** (3/4 perfect) but the rule degrades to 0.81–0.96 and
  retention is mixed. ⚠ Partly confounded: its GNG stage stopped at ~70/100 epochs under stop_loss
  0.1 while the "both" baseline ran 100 at 0.005.
- **"rates" does NOT unlock a bigger dt** either: at dt 0.045 all four seeds learn DPA (0.974–0.997,
  more consistent than fine dt) but retention falls 0.818 → 0.676 and the coupling blows up the same
  way (−0.97→−7.43, +0.30→−5.38). **So my hypothesis that the cascade was responsible for the dt
  fragility is WRONG — a single filter does it too.** Coarse integration produces a more entangled
  DPA solution regardless of filter count. ⚠ And s1 breaks the coupling↔retention correspondence here
  (coupling improved, retention fell), so the link is strong but not deterministic.
- **Verdict:** the cascade is load-bearing for the science (best on both axes), dt stays 0.0225 in
  every mode, and the remaining speed levers are the measured ones — concurrency (8 processes =
  5.52× aggregate) and batch size (launch-bound: s/step flat from batch 64 to 1024, so more trials
  per step is free compute but fewer optimiser steps per epoch).


## §31 — Attention off, the τ×noise grid, and the two end-of-delay demands (2026-09-14/15)

All arms below are 4-seed exploratory runs on the subcritical-lif recipe, stop_loss 0.1 at every
stage, Adam, `integrate="both"`, dt = dt_base·0.75 with α = dt/τ = 0.075 held fixed when τ changes.
Gallery titles = sweep names minus `sweep_`. Protocol readout = `scratchpad/readout_arm.py`
(per seed, DPA ckpt and expert, ALL memory attractors of the input-noise-averaged field, landings
simulated under the training noise; see `analysis.md` protocol rules 1–11).

### §31a — Removing the tonic attention input gives the best retention yet; test-driven pairing collapses the U

`sweep_lif_sub_noattn` (`attention_input=False`, τ 0.3, noise 1.0): after_gng/dpa **0.996 / 0.999 /
0.972 / 0.998** — the attention channel was what fed rank-1 back into the memory (§29e: n₀ᵀm₁ is
built through the tonic input). But dual_gng 0.59–0.67: the rule cannot be learned on top. With
`response_in_cue=True` (`sweep_lif_sub_noattn_ric`, the pairing decision scored in the last 0.5 s of
the TEST): retention 0.992 / 0.991 / 0.985 / 0.991 AND dual_gng 0.91–0.97. Geometry: without ric the
memory is a U-shaped continuous attractor (flow_verdict/find_all_fixed_points cannot see it — they
report "no fixed point"; the traj_grid shows the state sliding along a ridge); ric collapses it into
two isolated wells, both ON the line. **`noattn_ric` at τ 0.3 is the reference substrate.**

### §31b — Noise and τ: noise carves deep wells, τ buys nothing, and the state never gets there

Noise ladder on the reference (`sweep_lif_sub_noise`, τ 0.3): noise 1.5 → retention 0.78 / 0.47 /
0.55 / 0.54; noise 2.0 → 0.55 / 0.53 / 0.49 / 0.47. Then the τ×noise grid (`sweep_lif_sub_tau_noise`,
16 runs, τ ∈ {0.2, 0.15} × noise ∈ {1.0, 1.5}, dt compensated, staged 8 + 8):

| cell | after_gng/dpa (s0–s3) | memory attractors (input-noise field, expert) |
|---|---|---|
| τ 0.2 · n 1.0 (`tau20_n10`) | **1.000 / 0.999 / 0.773 / 0.996** | on-line pair at −0.2σ, occupied; deep well (+0.54, −1.15) = −3.1σ in s0, EMPTY |
| τ 0.2 · n 1.5 (`tau20_n15`) | 0.966 / 0.995 / 0.594 / 0.986 | s0 a full sub-line PAIR (−1.8σ / −1.6σ), s1 one (−1.9σ): all EMPTY; states on the on-line pair |
| τ 0.15 · n 1.0 (`tau15_n10`) | 0.992 / 0.924 / 0.393 / 0.502 | on-line pair; s3 no memory attractor |
| τ 0.15 · n 1.5 (`tau15_n15`) | 0.929 / 0.492 / 0.690 / 0.331 | s0 pair (−1.9σ / −1.5σ), s2 one (−1.6σ), s1 one at −0.9σ (occupied by B, d 0.23) |

σ_eff = noise·√(1−e^(−2α)) = 0.37 (noise 1.0) / 0.56 (1.5). Readings: (1) shorter τ is worse on
retention (1–2/4 vs 3/4) and does not change where the occupied wells sit; (2) noise 1.5 is what
creates deep sub-line wells, at both τ, exactly as the noise-smoothed hinge predicts (its residual
force at the line ∝ σ) — but the wells form NEXT TO the occupied one, not under it; (3) in
`tau15_n15` s0/s2 the A side has NO on-line well and drifts down the whole delay (κ₁ +0.11 → −0.33
over 5 s) toward a well at −0.8…−0.9 it never reaches — a slow-manifold/speed limit rather than a
basin boundary. **The landscape has the target geometry in most seeds; the state is not
transported into it.** `tau20_n10` is the working substrate (best retention, τ 0.2 = 1.5× faster
than 0.3).

### §31c — Which DPA-stage demand holds the wells on the line? Two hypotheses, three arms

At the DPA checkpoint every arm to date has its wells at |κ₁| ≤ 0.05 and the states exactly on them.
Two things in the DPA loss can do that (verified by dumping the targets per window):
- the **two-sided κ₁ pin** (`dpa_prelick_free=False` → `_pin(p)` on every `tgt == 0` step of the
  delay), which forbids κ₁ < 0 during the stage that builds the memory;
- the **end-of-delay |κ₀| ≥ θ = 1 hinge** (`dpa_hold_window` 0.5, ending at test onset). Leon's
  argument (2026-09-15): "the only way to have κ₀ at 1 is … with κ₁ at 0" — descending costs |κ₀|
  (in every net with a deep well, the deep well has smaller |κ₀| than the on-line one of the same
  net: 0.54 vs 0.72, 0.65/0.85 vs 0.75/0.95), and the DPA-ckpt wells sit at r = 0.64–0.99 < θ, so
  the hinge is unsaturated and pulls |κ₀| outward all through DPA training.
The Dual stage has NO κ₀ target at all (every window after the pre-sample baseline is NaN) and its
only delay term is the one-sided no-lick, satisfied at 0 — so nothing in Dual can move a well that
DPA built on the line. The test is therefore at the DPA checkpoint.

| arm (single-field delta from `tau20_n10`) | DPA-ckpt wells | expert | retention |
|---|---|---|---|
| `onesided`: pin → one-sided κ₁ ≤ 0 (`dpa_prelick_free=True`, `dpa_nolick_weight=1`) | 7/8 within ±0.2 of the line, one at −0.6σ (s0-B) | on-line wells −0.2σ; one empty deep well (s0, −1.7σ) | 0.999 / 0.999 / 0.796 / 1.000 |
| `mem_early`: κ₀ hold moved to the 0.5 s AFTER sample offset (`dpa_hold_anchor="sample"`), pin kept | on the line, \|κ₁\| ≤ 0.03, and \|κ₀\| UP to 1.02–1.10 | **deep pair in 4/4** (−2.2…−3.1σ); s3-A OCCUPIES (+0.93, −0.23) = −0.6σ, s2-B OCCUPIES (−0.72, −0.31) = −0.8σ; s0-A between line and deep well (κ₁ −0.30 ± 0.30); s1 has no line well, state hovers ±0.35 | 0.995 / 0.896 / 0.964 / 0.989 |
| `mem_free`: no κ₀ target (`dpa_hold_anchor="none"`) | NO memory attractors (s0: one well at (+0.53, +0.11), B on a non-attractor; s1–s3 none) | none | 0.52 / 0.49 / 0.54 / 0.51 (chance) |
| `onesided_early`: both (`dpa_prelick_free=True`, `dpa_nolick_weight=1`, `dpa_hold_anchor="sample"`) | RUNNING (launched 17:52) | | |

Readings. (1) Each single lever leaves the other demand in place, so neither DPA checkpoint could
move — by construction; `onesided` changed nothing in the geometry and did NOT bring the coupling
back (s2 is the fragile seed in both arms, n₀ᵀm₁ 2.6 vs 12.5). (2) `mem_early` is the first arm in
which trials OCCUPY sub-line wells (2 of 8 state-sides) and the deep pair exists in every seed; the
κ₁ spread on the line widens (sd 0.30–0.41 vs 0.19) — the line is a shelf rather than a well for
these states. The memory does NOT decay across the delay (noise-free κ₀ flat at 1.2–1.3 at the DPA
ckpt; I first claimed a decay from trial means and retracted it — that was the post-sample
relaxation under noise, identical in the baseline). (3) `mem_free` shows the κ₀ target is what makes
the memory an ATTRACTOR: without it even the seed that learns DPA perfectly encodes the sample as a
tilted transient (κ₀ ±0.5, identity partly in κ₁, n₁ᵀm₀ 0.9–3.8) that GNG erases in every seed. A
calibrated-stop_loss rerun would not change that (s0 was fully trained). (4) Noise and no-lick
weight are Dual-stage forces against a DPA-built geometry: noise carves wells beside the occupied
one and kills retention past σ ≈ 0.5; weight ≥ 3 sharpens the occupied well on the line and costs
the rule (dual_gng 0.69–0.80). If they are to be used, it is on top of the DPA recipe that
`onesided_early` selects (weight 2 first — it targets the occupied well).

### §31d — ★ The noise-model error, and protocol rule 11 corrected

These nets are trained with INPUT noise (`RunConfig.noise`, σ_eff per channel per step) and ZERO
recurrent noise (`model_noise = 0`). From the grid onward my "under trained noise" landings set
`model.noise = σ_eff` — isotropic recurrent noise the nets never saw — on top of the input noise.
The WELL tables were fine (`find_wells(noise_sigma=σ)` is the input-noise-averaged field:
`low_rank_field_np` averages each neuron's drive over variance g²Aᵢ²σ²‖wᵢ‖², the same object as
`plot_sweep --field_input_noise`); the LANDINGS were pushed ≈0.1–0.15 too far below the line
(mem_early s0-A "−0.43" is −0.30). And the plot_sweep trajectories were never noise-free — they draw
the batch with `meta.noise_sigma()` and recurrent 0, i.e. exactly the trained system. Rule 11 now
says: wells with `find_wells(noise_sigma=σ_eff)`; landings with `model.noise = 0` and input noise
σ_eff; never `model.noise = σ_eff`. All §31 numbers above are protocol-correct.


## §32 — The design: bowl + tail + no hold. Both memory wells below the line, occupied, task perfect (2026-09-16/17)

All arms 4 seeds on the `tau20_n10` substrate (no attention, test-driven pairing, τ 0.2, noise 1.0, Adam,
stop_loss 0.1, `integrate="both"`), protocol readouts (`scratchpad/readout_arm.py`: input-noise field
wells, landings under input noise, per side, at the DPA / naive / expert checkpoints).

### §32a — The loss split by sample, and the two DPA end-of-delay demands together

- **`nolick_split_sample`** (Leon): the no-lick hinge scored as TWO masked means, A-sample rows + B-sample
  rows, so the side that already satisfies the hinge cannot dilute the pressure on the other. Identity
  from the sign of the row's κ₀ target; Dual trials get the A/B hold written into their targets
  (`dual_mem_targets=True`) but the memory terms stay OFF (`dual_mem_supervise=False`) — Dual remains
  supervised through pairing only. Also splits the legacy κ₁ = 0 pin and the DPA one-sided hinge
  (`dpa_nolick_split`). GNG has no sample → no split there.
- **`onesided_early`** (pin → one-sided hinge AND κ₀ hold moved to after the sample): the FIRST arm whose
  DPA-ckpt wells all sit below the line (−0.3…−1.0σ, 7/8, states on them), retention 0.95–0.99. GNG
  keeps/deepens them; DUAL flattens the A side (3/4 lose the A well; s2-A ends on an upper well).
- **`onesided_early_w1split_dpa`** (+ split at both stages): DPA wells symmetric within 0.1σ in 3/4
  (s0 −0.5/−0.5, s1 −0.9/−1.0, s3 −0.4/−0.2), retention 0.999/0.923/0.948/0.984; Dual still reshapes
  but the A side descends further (−0.50/−0.10/+0.13/−0.36) and nobody ends above the line.
- **`free_early_w1split`** (NO κ₁ term in DPA at all): retention 0.99/0.97/0.99/1.00 — the best — but the
  DPA wells scatter ±0.9σ, 5/8 ABOVE the line: with nothing imposed the κ₁ of a DPA well is
  undetermined (sign per seed and per side). The one-sided hinge is not fighting an upward force; it
  resolves an indifference — which is also why it stops at −0.3…−1σ once the sign is chosen.
- Dual never converged in any of these (train 0.2–0.4 at 300 epochs, still falling); the dominant
  residual is the go hold (`gng_pos` 0.26–0.36) and nolick ~0.02–0.10 (≈ the relu² noise floor).

### §32b — Why every hinge saturates: two caps

1. DPA (inputs free): relu(κ₁)² has force 2[μΦ(μ/s)+sφ(μ/s)] — 0.8s at the line, 0.17s at −1s,
   0.02s at −2s. It switches itself off a noise-width below zero. Weight scales it, not where it dies
   (§29i's "weight buys tightening, not depth", now with the reason).
2. Dual (inputs frozen): the response is a fixed additive kick — measured pairing kick 0.6–1.0, go kick
   0.5–0.9 in every arm — so the paired trial must reach +1 from the well and a well below ≈ −0.2 costs
   the pairing directly. "The net adjusts the vertical flows instead of moving the wells" (Leon, 09-10)
   is this cap.

### §32c — Leon's specification and its three ingredients

Loss as a function of the well height w: **DPA a bowl at w = 0** (moving the wells away deteriorates
pairing); **GNG an inverted sigmoid** (the cue pushes up, a lick is wrong on nogo → lower wells better,
with a tail); **Dual = the sum → w* < 0**, no threshold anywhere. Reduced model
(`scratchpad/landscape.py`, kicks fixed at measured values, state noise 0.3): w* −0.37 (relu²) …
−0.5 (softplus) at weight 1, deeper with weight; DPA-only a bowl, GNG-only min at −0.5.
Each ingredient alone, measured:

| arm | what | result |
|---|---|---|
| `w1split_dpa_softplus` (A) | no-lick shape → softplus (logistic lick cost), both stages | DPA wells at (±0.85, **−3.2**) = −8.5σ, DPA 1.0 — with one-sided pairing NOTHING opposes the tail in DPA (the kick just grows); after GNG the pairing readout is dead (0.63/0.54/0.57/0.50) |
| `w1split_dpa_nohold` | no go/nogo delay hold (`gng_weight` 0 — the go LICK had to be added: `gng_response=True`, the old recipe had NO cue-time go target, the hold WAS the lick) | go/nogo become displacements of the well (go +0.35…+0.39, nogo −0.23…−0.50, none ≈ 0) instead of absolute (+0.4 / −1); GNG learns it (0.73–0.96) but Dual LIFTS the wells (+0.6…+1.6σ): the go hinge (unsatisfied at 0.74) pulls up, relu² nolick pushes nothing at 0 |
| `design1_w1/w4` (from the nohold DPA ckpt): `pair_pin` (two-sided ±1 pairing) + softplus + no hold | **w1: 3/4 seeds with BOTH wells at −1.8…−2.7σ, symmetric, OCCUPIED (d ≤ 0.44), dual_dpa 0.996–1.0, go 0.99–1.0, nogo 1.0.** w4: −2.8…−3.9σ, nogo 1.0, go 0.85–0.96. The target geometry, no painted value. BUT the softplus in GNG (nothing opposes it there) killed retention after GNG (0.51–0.64); Dual rebuilt it |
| `design2_w1/w4` (full sequence; DPA no push, GNG relu², Dual bowl+softplus+no hold) | retention after GNG 0.94–1.00 (GNG clean). DPA wells scattered ±0.9σ (§32d); Dual pushes ONE side per seed (−1.0…−1.4σ) and leaves/raises the other; w4: s0/s1 symmetric (−1.5/−1.9σ, −3.3/−3.4σ) |

Reduced model vs design1 (kicks re-measured): structure right everywhere (interior optimum, depth ∝
weight, go pays at w4); quantitatively right at w4 (−0.84…−1.00 vs −0.88…−1.31) and a lower bound at
w1 (−0.22…−0.34 vs −0.39…−0.82): with frozen inputs the net still GREW the effective kicks (go d+C
1.0–1.8, K 1.2–1.9 vs 0.75/1.0 in nohold) through the recurrent dynamics — the "free inputs" row of
the model is the better predictor.

### §32d — The bowl is NOT available in DPA (a claim of mine, falsified)

I argued that two-sided ±1 pairing forces w = 0 (w + K = 1, w − K = −1). `design2` with `pair_pin`
confirmed active in DPA gives the SAME scattered wells as `free_early`, to two decimals. The pairing
response is not a fixed symmetric kick: it is computed by the recurrence from the (sample × test)
conjunction and can be asymmetric (+0.7 / −1.3 from a well at +0.3). In DPA, with inputs and
recurrence free, κ₁ of the wells is undetermined by the task — with lif there is no ring to give
κ₁ = 0 for free (§33), and no target does either. The bowl exists only in Dual, where the kicks are
frozen. Consequence: the Dual stage propagates the DPA start's symmetry — design1 (symmetric hinge
wells) → symmetric deep pairs; design2 (scattered) → one side per seed. **Best recipe on the table =
design1's DPA (one-sided hinge, split) + design2's GNG (relu²) + Dual (bowl + softplus + no hold) —
`design3`, not yet run.**

### §32e — Fig. 4c of the dual project, for the RNN (`scratchpad/fig4c_rnn.py`)

Per seed × sample: Δ well depth (κ₁ of the OCCUPIED attractor of the input-noise field, expert −
naive) vs Δ accuracy (DPA on DPA-only trials; NoGo on dual trials — go is at ceiling, pooled GNG hid
the effect). Naive nogo accuracy per sample is binary on well side (0.00 above the line, 0.93–1.00
below); the sample-sides that go down gain nogo (0.00 → 0.87–0.98), those that go up lose it — the
mouse sign on the NoGo arm. The DPA arm is flat at ceiling (naive DPA ≈ 1.0), unlike the mice; and
the arms with a broken naive stage (design1) have no naive wells to measure Δ from.

### §32f — Method notes
- Dual with softplus never reaches stop_loss 0.1 (floor 2·ln2·w = 1.39 at w=1, 5.5 at w=4 at the
  line) → `epochs_dual` 150 for design2 (Leon). A plateau stop is the cleaner fix (not built).
- The go/nogo "hold" (`gng_weight`) and the cue-time lick (`gng_response` → `rwd_go`) are different
  terms; the recipe up to 09-16 had no cue-time go target at all.
- Trained noise = INPUT noise (rule 11): never `model.noise = σ`.


## §33 — The ring: covariance, not the transfer function (2026-09-17/18)

Leon: "in the theory the ring should not depend on the transfer function but on the covariance
between n₀ and n₁." Correct, and now measured in-house; the `nonlinearities.md` line "ring-capable:
tanh and erf only (odd + saturating)" is an empirical statement from structured inits, not the theory.

### §33a — Mean-field
F(κ) = −κ + Σ·⟨φ′(Δ(κ))⟩·κ with Δ(κ) = κᵀΣ_m κ (Stein's lemma, zero-mean Gaussian vectors): φ
enters only through the averaged gain; the field is rotationally symmetric iff the rank-2 covariance
is isotropic — equal σ_m across modes, equal σ_n, zero cross-covariances (m₀·m₁, n₀·n₁, m₀·n₁,
m₁·n₀), zero means. The overlap matrix **J_ij = g·n_iᵀm_j/N** (the linearised κ-gain; origin unstable
when g·J·φ′(0) > 1, i.e. J > 2.5 for lif φ′(0) = 0.399 at gain 1; ≈ 3.4 under the input-noise-averaged
gain) is necessary but NOT sufficient: it constrains only the products.
Per mode (`init.py`: m = a·u + s·p_m, n = a·u + s·p_n): λ = a² = nᵀm/N, σ² = a² + s² = ‖m‖²/N =
‖n‖²/N, ρ = λ/σ² → **two free parameters per mode (λ, ρ), σ² = λ/ρ**, plus the gain. The literature
convention (unit-variance Gaussian entries) is the σ = 1 slice, ρ = λ ≤ 1.

### §33b — Surrogate fields on a trained DPA net (`s0_design2_w1`)
Anisotropy index = range of the radial field component around a circle (input-noise-averaged field):

| field | R 0.8 | R 1.1 | R 1.5 |
|---|---|---|---|
| trained lif vectors (J₀₀ 6.8, J₁₁ 7.4, |J₀₁|,|J₁₀| ≤ 0.6, m₀⊥m₁, n₀⊥n₁) | 0.21 | 0.41 | 0.69 |
| Gaussian surrogate, same 4×4 covariance + means, lif | 0.38 | 0.44 | 0.50 |
| Gaussian surrogate, same covariance, zero means | 0.30 | 0.34 | 0.39 |
| Gaussian surrogate, ISOTROPISED covariance, zero means, lif | 0.16 | 0.16 | 0.16 (finite-N floor) |

The anisotropy is second-order structure, not higher-order sculpting, and lif rings once the covariance
is isotropic. The trained nets have isotropic J but unequal FACTORS: σ(m₀) 2.56 vs σ(m₁) 2.08, σ(n₀)
2.91 vs σ(n₁) 3.91 — DPA training equalises the products (the task cares about J) and never the
factors. The seed is `init.py`: memory mode σ_m = σ_n = √(λ₀/ρ) = 1.41; decision mode n₁ unit-variance
(`readout_scale`) and m₁ = λ₁/ρ₁ ≈ 0.63 — the two modes are not exchangeable from the start.
(July's tanh ring nets: checkpoints no longer on disk; the criterion could not be checked on them.)

### §33c — The init grids (`scratchpad/init_flow_grid.py`, gallery `init_flow_grids`)
Autonomous fields of rank-2 lif nets AT INIT rendered with the trained-net panel code (input-noise
field σ_eff 0.37, magma, ● attractor / × saddle / △ repeller), gain 1, N 1024, box ±1.5.
- **Grid 1** (isotropic construction, cols λ ∈ {1.6, 3, 5, 7, 10}, rows ρ ∈ {0.3, 0.5, 0.8, 1.0}): single
  stable origin below λ_c (our init λ₀ = 1.6 is subcritical — every DPA stage crosses criticality; the
  trained nets sit at J ≈ 7 because the ±1 targets need radius 1); above it a closed low-speed
  ANNULUS around a repelling origin — the ring — radius growing with λ and shrinking with ρ (σ² = λ/ρ:
  ρ 0.3 r 0.4→0.6, ρ 0.8 r 0.6→1.0, ρ 1.0 r 0.7→1.1 over λ 5→10). Finite N picks 2 ● + 2 × on it
  (tangential eigenvalue 0 → any residual anisotropy chooses); at ρ 0.3–0.5, λ 7 the finder reports NO
  discrete point on the annulus — the most isotropic cells.
- **Grid 2** (memory mode at J₀₀ 7, ρ 0.8; decision λ₁/λ₀ ∈ {0.25, 0.5, 1, 2}; rows = `init.py`
  construction vs isotropic): ratio ≤ 0.5 two memory wells on κ₀ in both; ratio 1 isotropic = annulus,
  `init.py` = NO annulus, a tight ●●×× cross with ● on κ₁ at r ≈ 0.3 — the ring is broken at equal
  overlaps by σ(n₁) = 1 vs σ(n₀) = 2.96; ratio 2 wells on κ₁ (r 0.3 vs 1.3).
- **Grid 3** (λ₀ = λ₁ = 7, ρ 0.8 both; rows σ(n₀) ∈ {1, 1.7, 2.96}, cols σ(n₁) ∈ {1, 1.7, 2.96, 4};
  general construction n = σ_n(√ρ u + √(1−ρ) p_n), m = σ_m(√ρ u + √(1−ρ) p_m), σ_m = λ/(ρσ_n)):
  **σ_n sets the radius of its mode at fixed J** (n only via the overlap, σ_m via saturation
  Δ = σ_m²R²): σ_n 1 → r 0.3, 1.7 → 0.5, 2.96 → 0.85, 4 → 1.1. Off the diagonal the ring becomes an
  ELLIPSE and breaks into **four wells at the ends of both axes** with saddles between — the
  four-cardinal-wells geometry of the trained DPA nets (σ(n₀) 2.91 vs σ(n₁) 3.91 → predicted radii 0.85
  vs 1.1 ≈ measured). Equal σ_n restores the circle.

Open: whether a DPA ring is wanted, and whether isotropy would survive training (isotropic init,
DPA-only probe, 4 seeds — not run). Any isotropy term would be a constraint on the representation.

### §33d — ★ Ring vs wells is a QUANTITATIVE call, and the multiplier is the wrong yardstick (2026-09-18)

Leon: "our solver finds fixed points on the ring; I feel like these are incorrect, because the whole
ring is a fixed point." Right in the mean-field limit, and the solver is right about the field it is
given — the two are reconciled by finite N, and the measurement forced a change of diagnostic.

**Finite-size scaling** (isotropic construction λ = 7, ρ = 0.5 both modes; noise-free field; 3 seeds):

| N | max \|F_θ\| on the ring | angular modulation of F_r | ring radius |
|---|---|---|---|
| 512 | 0.059–0.104 | 0.071–0.075 | 0.67–0.76 |
| 2048 | 0.025–0.029 | 0.030–0.036 | 0.69–0.70 |
| 8192 | 0.013–0.019 | 0.021–0.024 | 0.69–0.70 |

The corrugation falls as ~1/√N (16× more units → ≈4× smaller) while the radius is stable: it is
sampling noise in the covariance, not structure. The fixed points the solver returns ARE genuine zeros
of the finite-N field (residual < 1e-8); they are the discrete residue of a continuous attractor.
**Why exactly four, on the mode axes:** the leading perturbation is the covariance mismatch, a
quadratic form in κ, hence a SECOND harmonic cos 2(θ−θ₀) — which pins exactly 2 attractors + 2 saddles,
at the eigenvectors of the anisotropy (= the κ₀/κ₁ axes when it is the σ_m/σ_n split). Same mechanism,
larger amplitude, gives grid 3's ellipse-with-four-wells.

**The eigenvalue evidence, and why the raw multiplier is useless.** A map multiplier is
λ_i = 1 − dt/τ_i, so with dt = 0.015 and τ ≈ 0.2 s (α = 0.075) EVERY attractor reads |λ| ≈ 0.99 and a
threshold on it says nothing. In relaxation times τ_i = dt/(1−|λ_i|):

| field | multipliers (slow/fast) | τ_slow / τ_fast | anisotropy |
|---|---|---|---|
| isotropic init ring (λ 7, ρ 0.8) | 0.9973 / 0.8417 | **5.6 s** / 0.095 s | 59× |
| trained DPA nets (`design2_w1`, 4 seeds) | 0.9912–0.9962 / 0.843–0.847 | **1.0–3.9 s** / 0.10 s | 17–41× |

So the trained "wells" are not qualitatively different from the ring: a fast radial contraction
(0.1 s) onto a slow set, with a tangential time constant of 1–4 s against a **5 s delay**. The state
does not finish settling along the slow direction within a trial. This is the same object as §31a's
U-shaped continuous attractor and §31b's states drifting the whole delay toward a well they never
reach — those were not anomalies, they are what a 1–4 s tangential time constant looks like.
`classify_fixed_points`' `marginal_tol` = 2e-3 (on the multiplier) only catches exact degeneracy, so
everything here is labelled "stable attractor".

**Changes made (so future flow plots characterise FPs correctly):**
- `bifurcation_probe.find_wells(..., with_eigs=True)` → `(κ, kind, (τ_slow, τ_fast))` in SECONDS
  (backward compatible; default return unchanged). Docstring states the rule.
- `scratchpad/readout_arm.py` prints `τ slow/fast s` per attractor and flags `SLOW` when
  τ_slow > the delay.
- `plot_sweep.py --mark_slow` is now **default ON** (`--no_mark_slow` restores the old behaviour):
  shallow/slow attractors get the orange ring in every flow panel, so a ring remnant is never drawn
  like a well. `--slow_tol` 0.06 on the multiplier remains the marker's threshold (it flags what is
  slow at a glance; the seconds are the quantitative statement).
- **Rule:** never report "wells" from labels alone — report τ_slow in seconds against the delay.


## §34 — What DPA training builds, as a function of the init (λ scan at ρ = 1, isotropic) (2026-09-18)

`sweep_lif_dpa_lambda_scan_rho1` — DPA stage ONLY, 2 seeds × λ ∈ {1.6, 3.5, 7, 12}, ρ = 1 on both modes
and the **isotropic construction** (new `RunConfig.readout_scale` = √(λ₁/ρ) makes σ(m) = σ(n) = √(λ/ρ)
on both modes; without it `init.py` forces σ(n₁) = 1 and the modes are not exchangeable, §33). Loss =
pairing + the A/B hold in the 0.5 s after the sample, **no κ₁ term**. Figures: gallery
`lif_dpa_lambda_scan_rho1` (`fp_lambda_x_condition_dpa`, `flow_rows_lambda`, `flow_grid_lambda`).

**Three things are task-locked, one is not.**
1. **J ≈ 7 whatever the init.** J₀₀/J₁₁ after DPA: 6.97/6.57 and 6.54/7.44 from λ 1.6; 7.08/6.92, 7.14/6.89
   from 3.5; 6.81/7.10, 7.06/5.59 from 7; 9.14/7.68, 8.59/9.48 from 12. The low end is forgotten
   entirely (λ 1.6 is SUBCRITICAL — a single stable origin at init, λ_c = 1/(g·φ′(0)) = 2.5, ≈3.4 under
   the noise-averaged gain), the high end keeps a little memory. Confirms §14a/§15c.
2. **The radius is task-locked at |κ| ≈ 1** — every attractor at r = 0.96–1.18, the ±1 target amplitude.
3. **What DPA builds is a RING of radius ≈1 with 2–3 slow wells on it**, not four cardinal wells: e.g.
   s0 λ 3.5 → (+1.09, 0.00), (+0.68, +0.84), (−1.01, −0.34), all at r ≈ 1.07.
4. **λ controls the corrugation, and it runs opposite to the factor isotropy.** λ = 12 keeps σ(m), σ(n)
   nearly equal across modes (iso ratio 1.04–1.08) yet gives the most anisotropic field (0.42–0.58 vs a
   1/√N floor of 0.062) and the fastest wells (τ_slow 0.6–1.4 s) — its anisotropy moved into the
   CROSS-overlaps (J₀₁ = −0.41, −1.21). λ = 1.6/3.5 must grow their overlaps 4×, which re-breaks the
   factor isotropy (σn₁ 3.6–3.9 vs σn₀ 2.8–2.9) but leaves the field nearly flat (0.085–0.17) with
   genuinely ring-like slow directions (τ_slow 5.5–6.5 s > the 5 s delay). Cleanest ring: λ 3.5 seed 1.

**Input-driven structure (same at every λ).** A and B each collapse the whole plane to a single point
(the write); C and D are mirror-image basin choices with a saddle between (the read, rotating the
memory axis onto κ₁); the sample drive is 2–5× stronger than go/nogo.

### §34a — GNG on the same nets, with and without the delay memory
`sweep_lif_gng_lambda_scan_rho1` (`gng_weight` 0 + `gng_response`: the rule as a transient, no κ₁ hold)
and `sweep_lif_gngmem_lambda_scan_rho1` (the same + `gng_weight` 1: the hold IS learnt), both from the
λ-scan DPA checkpoints.

| λ | GNG acc: no hold → hold | retention: no hold → hold |
|---|---|---|
| 1.6 | 0.966, 0.938 → 0.996, 1.000 | 1.000, 0.885 → 0.999, 0.968 |
| 3.5 | 0.944, 0.957 → 0.996, 0.994 | 0.994, 1.000 → 1.000, 1.000 |
| 7 | 0.984, 0.989 → 1.000, 1.000 | 0.997, 0.999 → 0.999, 0.998 |
| 12 | 1.000, 1.000 → 0.993, 1.000 | 0.740, 0.979 → **0.491**, 0.979 |

- **No hold:** GNG shrinks the decision mode (σ(m₁) 2.07→1.87, 2.59→2.22, 2.97→2.49; J₁₁ drops ~1 in 6/8)
  so the isotropy ratio WORSENS (1.04→1.19, 1.26→1.39, 1.06→1.16) — learning the rule corrugates the
  ring further. τ_slow falls from ≤6.5 s to 1.0–5.0 s: the slow ring becomes discrete wells.
- **With the hold:** the field grows a SECOND attractor pair on the **κ₁ axis** ((+0.13, +1.47) and
  (+0.20, −1.09) at λ 1.6; (+0.32, +0.92) at 3.5; (+0.19, +0.89), (−0.26, −0.98) at 7) — the go/nogo
  memory stored as its own wells at r ≈ 1, i.e. the four-well structure built deliberately. The rule is
  then perfect at every λ and retention is better, but κ₁ carries a second memory through the delay.
- λ = 12 seed 0 is the failure mode in both: after GNG its autonomous field has **no stable attractor**
  and retention collapses (0.74 / 0.49). Rule perfect, memory gone.
- ⚠ The go/nogo drive measured at the origin is 2–5× weaker than the sample drive and NOT orthogonal to
  the memory mode (Go +55…+100°, NoGo −58…−151°, where 90° is pure up), so the clamped-input panels show
  the memory structure surviving inside them. The trajectories agree with the field on DIRECTION
  (Go +62…+82°, NoGo −59…−147°); what differs is the endpoint — a GNG trial starts near the ORIGIN and
  travels only ≈0.4–0.7 in 1 s, so it never reaches the clamped fixed point at r ≈ 1.

### §34b — ★ Dual on the no-hold nets: both memory wells below the line, and λ decides how often
`sweep_lif_dual_lambda_scan_rho1` — Dual = design2's (two-sided ±1 pairing, softplus no-lick in Dual
only, no hold, split by sample, w = 1, 150 epochs) from the gngscan naive checkpoints. All 8 learn the
dual task (dual_dpa 0.990–1.000, dual_gng 0.866–1.000).

| λ | s0 attractors (expert) | s1 |
|---|---|---|
| 1.6 | (+1.05, −0.26), (−0.81, **+0.57**) | (+0.88, **+0.56**), (−0.90, **+0.50**) — both ABOVE |
| 3.5 | (+0.93, −0.37), (−0.73, −0.46), (+0.89, +0.58) | (+0.73, −0.52), (−0.97, **+0.51**) |
| 7 | (−1.00, −0.37) τ 6.6 s, (+0.89, −0.31), (−0.99, +0.10) | ★ **(+0.94, −0.55), (−0.94, −0.50)** + (+0.98, +0.45) |
| 12 | **(+1.10, −0.75) = −2.0σ, (−0.94, −0.88) = −2.4σ** + a pair near the line | **(+0.95, −0.24), (−1.01, −0.37)** |

**Both wells below the line in 3–4 of 8, every one of them at λ ≥ 7.** Bigger λ → wells further out along
κ₀ → the Dual stage's κ₁ displacement does not merge them and both can descend; at λ 1.6 the stage
splits them one-up-one-down instead. Training breaks the factor isotropy hard here (ratio 1.36–1.72,
σn₁ 4.3–5.0 vs σn₀ ≈3.0).

**★ `s1_dualscan_7` is the target solution.** Memory wells at (±0.94, −0.5) = −1.35σ / −1.5σ, both
below, occupied: on DPA-only trials the state is held at κ₁ ≈ −0.45 for the whole delay (DPA ckpt +0.10/
−0.05 → naive 0/−0.15 → **expert −0.45**), then snaps to ±1 at the test. after_gng/dpa 0.998,
dual_dpa 1.000, dual_gng 1.000 (go 1.000, nogo 1.000). Input-driven columns show why it is stable: Go
drives to (±0.8, +1.2) from either side, Cue to (0, +1.6), C/D are mirror basin choices — the pair sits
exactly where the cue's upward push is absorbed without a false lick. **Nothing is painted: the DPA
stage has no κ₁ term at all.** `recipe7` (8 seeds, identical config end-to-end) queued to test how often
it reproduces. ⚠ Dual runs given only `gng_ckpt` have no `dpa_*.pth` — copy it from the originating run
before plotting, or the DPA row of every figure is missing (done for this sweep).

### §34c — A `sweep.py` bug: `gng_ckpt` did not skip the DPA stage
The DPA stage was gated only on `dpa_ckpt`, so a run given just a `gng_ckpt` silently retrained DPA for
250 epochs and then had it overwritten by the checkpoint load. Results were correct, ~40 min/run were
wasted and `after_dpa` / `dpa_*.pth` were misleading. Fixed: `gng_ckpt` without `dpa_ckpt` now skips DPA
entirely, as its docstring always claimed.


## §35 — The init ensemble's symmetry is what forbids both wells below the line (2026-09-18)

Leon: "right now m and n are unimodal gaussians at init — what if we made them bivariate? what if we
included symmetries otherwise?"

### §35a — The theorem, and the measurement
Each neuron's (m₀, m₁, n₀, n₁) is one draw from a **zero-mean 4-D Gaussian** — unimodal, and symmetric
under the joint sign flip (m, n) → (−m, −n). For ANY transfer function that implies
⟨n φ(g m·(−κ))⟩ = −⟨n φ(g m·κ)⟩, i.e. **F(−κ) = −F(κ)**: the autonomous field is exactly odd, so every
attractor has a partner at −κ, and **both memory wells below the lick line is symmetry-forbidden at
init**. Measured (‖F(κ)+F(−κ)‖/‖F(κ)‖ on a circle): init **0.000**; after DPA 0.32; after GNG 0.26;
`s0_design1_w1` expert (both wells below) **1.67**. So the solutions we want exist only by destroying
that symmetry during training, slowly and unreliably — which is precisely the one-up-one-down failure of
§34b. (This also restates §12a's deadlock correctly: it is a property of the ENSEMBLE, not of odd φ.)

### §35b — Two populations, and which symmetry to keep
A mixture of Gaussians (multiple populations) is the standard way out — Beiran et al. 2021 and Dubreuil
et al. 2022 show the number of populations sets which computations a low-rank net can implement. But a
±μ pair of populations is still sign-flip symmetric; what matters is WHICH symmetry survives:
- **inversion** κ → −κ (zero-mean Gaussian): wells at (κ₀, −w) and (−κ₀, **+w**) — one up, one down.
- **reflection** κ₀ → −κ₀ with κ₁ fixed (the task's own A↔B exchange): wells at (±κ₀, w) with a COMMON
  w — both down or both up, never one each.
Implementation (`mirror_tying`): two halves related by an involution — memory mode sign-flipped,
decision mode copied, A↔B and C↔D input columns swapped, go/nogo/cue shared. Exact at finite N:
n₀ᵀm₀/N and n₁ᵀm₁/N preserved, **n₀ᵀm₁ = n₁ᵀm₀ = 0**. `Optimization._mirror_tie()` re-imposes it after
every optimiser step by PROJECTING onto the symmetric subspace (averaging the halves, so it is
gradient-consistent, not a copy). Verified after real training steps: max deviation 0.00e+00 on all four
components, cross-overlaps 5e−8.

### §35c — The init grids (gallery `init_flow_grids`)
- **grid4** (2-pop reflection, λ × ρ): exactly two mirror attractors ON the κ₀ axis at (±κ₀*, 0),
  cross-overlaps 0.00, odd violation 0.000. No ring — the reflection IS an anisotropy in the (m₀, m₁)
  joint distribution; that is the trade (a ring gives a free κ₁ but couples the wells antisymmetrically).
- **grid5** (+⟨n₁⟩ = −1): far too strong — the pair merges into a single sink at (0, ≈−1), memory gone.
- **grid6 / the numeric map** (dose of ⟨n₁⟩ × λ at ρ 0.8; the resting decision is κ₁ ≈ φ(0)·⟨n₁⟩ for a
  non-negative φ, the only displacement the reflection allows):

| λ | ⟨n₁⟩ −0.05 | −0.10 | −0.15 | −0.20 | −0.30 |
|---|---|---|---|---|---|
| 7 | none | none | single −2.3σ | single | single |
| 10 | PAIR (±0.75, −0.55) −1.5σ | none | none | single −3.0σ | single |
| 14 | PAIR −1.7σ | **PAIR (±0.59, −1.14) −3.1σ** | PAIR −3.2σ τ6.4 | single | single |
| 20 | PAIR −1.9σ | PAIR −2.2σ | PAIR (±0.69, −1.41) −3.8σ | PAIR −3.9σ | single |

  There is a window, and it widens with λ: at λ = 14–20 with ⟨n₁⟩ ≈ −0.1…−0.2 the UNTRAINED net already
  has two mirror memory attractors 2–4σ below the lick line. Too little λ or too much mean and the pair
  merges into one sink on the κ₁ axis.

### §35d — The experiment (`sweep_lif_mirror_dpa`, running)
DPA stage only, 4 + 4 seeds, λ = 14, ρ = 0.8, `readout_scale` = √(λ/ρ) = 4.18, ⟨n₁⟩ = −0.1, DPA loss with
**no κ₁ term**: `mirror_lam14` (tying held through training) vs `decmean_lam14` (identical, tying off) —
isolating the symmetry from the mean. Question: do both wells stay below the line once the task is
learned, and is the tying or the mean doing the work.

---

## §36 — Symmetry as a curriculum: hold it through DPA, release it, and both wells go down 8/8 (2026-09-18)

Three sweeps close the §35 thread. All three use the `s1_dualscan_7` recipe as the base (lif, gain 1,
λ = 7, ρ = 1, isotropic init, `pair_pin` two-sided, `nolick_shape` softplus, no-hold, `nolick_split_sample`)
and **no κ₁ term anywhere in the DPA loss**.

### §36a — `sweep_lif_mirror_dpa` (λ = 14, ⟨n₁⟩ = −0.1): the tying is what does it
DPA stage only, 4 + 4 seeds. `mirror_lam14` (σ₁ tied through training) vs `decmean_lam14` (identical
init and mean, tying off).
- **Tied: 4/4 both wells below**, −0.47…−0.89σ, occupied (d ≤ 0.04), fast (τ_slow 0.4–0.9 s against a
  5 s delay, so genuine wells and not ring remnants), cross-overlaps n₀ᵀm₁ = n₁ᵀm₀ = **0.00 exactly**,
  DPA 0.993–1.000.
- **Untied: 3/4 fail** — asymmetric pairs, or one memory well missing altogether (one odor with nowhere
  to sit). Same init statistics, same ⟨n₁⟩. So it is the symmetry, not the mean, that produces the
  matched pair.

### §36b — `sweep_lif_recipe7` (8 seeds, free): the target solution reproduces 6/8
Eight seeds of the exact `s1_dualscan_7` configuration, end to end, nothing tied.
- 8/8 learn the whole task: dual_dpa 0.995–1.000, go 1.000, nogo 0.932–1.000, retention 0.990–1.000.
- **6/8** end with BOTH memory states in wells below the line. Mean depth −1.43σ, mean left–right
  imbalance **0.231σ** (worst 0.48σ). The 2 misses are the σ₂ pattern: one well down, one up.

### §36c — `sweep_lif_symdpa`: σ₁ vs the full Klein group, held in DPA only, then released
8 runs, 4 seeds each of `symdpa_pair` (σ₁, 2 blocks) and `symdpa_klein` (V, 4 blocks), `symmetry_stages`
= DPA only, released before GNG. This is the sharp test of §9.3, because **σ₁ and V differ in exactly
one way**: σ₁ = diag(−1,+1) ties the two wells to a COMMON κ₁ but leaves that height free; adding
σ₃ = diag(+1,−1) closes the fixed-point set under κ₁ → −κ₁, so two wells must be **pinned at κ₁ = 0**
and any extra attractor must come with its reflection.

**At the DPA checkpoint (constraint held) — the prediction is confirmed on both halves:**

| arm | memory pair κ₁ | in σ | attractor count, 4 seeds | extra attractors |
|---|---|---|---|---|
| `symdpa_pair` (σ₁) | −0.11 … −0.14 | −0.29 … −0.37σ | **3, 3, 3, 4** | a LONE sink at (≈0, +1.1…+1.3) — its own σ₁ image, so odd counts are allowed |
| `symdpa_klein` (V) | −0.02 … +0.02 | ≤ 0.1σ | **2, 4, 4, 2** | when present, a σ₃ PAIR at (≈0, ±0.93…±1.02) — equal and opposite |

Cross-overlaps **0.000 exactly** in both arms; DPA 0.998–1.000. The orbit relation is exact unit by
unit: max |actual − D_σ·predicted| = **0.00e+00** on all four components at the DPA checkpoint, and
4.1 (pair) / 9.5 (klein) at the expert checkpoint, with J off-diagonals moving 0.000 → 0.10–0.23. That
residual is the symmetry breaking, measured in the parameters rather than in the flow.

The −0.3σ the σ₁ arm shows at DPA, with no κ₁ term in the loss, is the memory task's own mild
preference for sitting low — a preference the full group is strong enough to forbid.

**At the expert checkpoint (released, GNG + Dual trained freely) — 8/8:**

| curriculum | both states below | mean depth | mean \|left−right\| | worst |
|---|---|---|---|---|
| free throughout (`recipe7`, 8 seeds) | 6 / 8 | −1.43σ | 0.231σ | 0.48σ |
| symmetry held in DPA, then released (`symdpa`, 8 seeds) | **8 / 8** | −1.49σ | **0.077σ** | 0.23σ |

Behavior after release: dual_dpa 0.999–1.000, dual_gng 0.974–1.000, go 0.994–1.000, nogo 0.947–0.997.

**The mechanism is not that the constraint pushes anything down** — it is gone by the Dual stage, and
while it was on it either held the wells level (σ₁) or pinned them on the line (V). What it does is
hand GNG a configuration with **no residual σ₂**, so the breaking field acts on both memories the same
way and they descend together. Depth is unchanged; reliability and balance are what improve. Symmetry
here is scaffolding: it selects which solution the later stages are pushed away from.

Figures: gallery `rnn/lif_symdpa` — `summary_symmetry_flow_grid_dpa/expert.png` (4 seeds × pair/klein,
lick line dashed), `summary_orbit_block_check.png` (unit-by-unit identity check, held vs released),
`s0_mn_pairs_symdpa_*_dpa.png`. Artifact: https://claude.ai/artifact/ANVVa4bWp1bzB4fByKJFwW

---

## §37 — The symmetry artifact reviewed from scratch: the bias condition, the exact oddness, and the ledger (2026-09-21)

Leon: "review from scratch the group theory artefact ... make sure the document is coherent, the math
make sense the figures too." Four things were wrong or missing; all are fixed in v2 of the artifact
(https://claude.ai/artifact/ANVVa4bWp1bzB4fByKJFwW) and recorded here.

### §37a — The bias is a fourth equivariance condition, and the projection had left it free
`W_in` has a trainable per-unit bias `wi.bias` (rms 0.5–1.6 after training). Equivariance needs
**P b = b** alongside P m = m D, P n = n D, P W_in = W_in S. `project_symmetry` / `symmetrize_init` tied
the first three only. Consequence for `sweep_lif_symdpa` and `sweep_lif_mirror_dpa`: the low-rank
structure is exact (max |actual − D_σ·pred| = 0, cross-overlaps 0.000) but the FIELD carries a residual
from the bias — measured ‖F(Dκ) − D F(κ)‖/‖F‖ on the disk |κ| ≤ 1.5 at the DPA checkpoint: pair σ₁
0.011–0.035; klein 0.002–0.054. **Fixed 2026-09-21**: both functions now tie the bias (block average /
copy). Verified on the trained klein checkpoint: projecting the bias takes all three residuals from
0.04–0.05 to 6e-15. Runs before this date are unaffected in their conclusions (the residual is a few
percent) but are not exactly equivariant; a definitive rerun should use the fixed projection.

### §37b — Why the inversion is EXACT at init: φ − ½ is odd, ⟨n⟩ = 0, b = 0 (not the ensemble theorem)
lif is the Gaussian CDF, φ(u) = ½ + ½ erf(u/√2). For the autonomous field, exactly,
  Ψ(κ) + Ψ(−κ) = ⟨n⟩ + (1/N) Σᵢ nᵢ [Φ(g(mᵢ·κ + bᵢ)) − Φ(g(mᵢ·κ − bᵢ))].
The even part of the field is a constant, the unit-mean of n, plus a term odd in the bias. The structured
init z-scores n (⟨n⟩ = 0, or = `decision_readout_mean`) and zeroes the bias ⇒ **F(−κ) = −F(κ) for any
draw of (m, n), any N** — which is why the measured violation at init is 0.000 in all 8 recipe7 seeds.
The §35a ensemble theorem (population symmetric under (m,n) → (−m,−n)) is the general-φ statement and
is NOT what makes it exact here (a finite Gaussian sample is not sign-symmetric). Corollary: for an odd φ
(tanh, erf) the constant vanishes, so ⟨n⟩ cannot break the inversion and ONLY the bias can — with b = 0
a tanh memory is a quadruple, never two wells below the line (§3 of theory_landscape); with a trained
bias it is odd only to the extent the bias term is small (Ψ(0) ≈ 0 was checked then). Training breaks σ₂ through exactly the two parameters named: ⟨n⟩ and b; both
contributions are comparable in the trained nets (E_const vs E_rest in `symviol2.tsv`).

### §37c — The deafness lemma: a σ₃- (or σ₂-) equivariant net cannot hear go/nogo
Under σ₃ the go/nogo/cue columns are block-shared (P w = w) while n₁ flips sign between blocks, so
n₁ᵀw_go = n₁ᵀw_nogo = n₁ᵀw_cue = 0 identically; under σ₁ it is n₀ that flips, so n₀ᵀw_go = 0 (the
leak a tied net sets to zero). Measured (`scratchpad/input_overlaps.py`, channel 4 = go+cue, 5 = nogo):
klein DPA ckpt n₁·w_go/N = n₁·w_nogo/N = **0.000** in 4/4; after GNG +0.93…+1.22 / −1.37…−1.60; after
Dual +1.82…+2.99 / −1.95…−2.61. So GNG MUST break σ₃ (and σ₂) in the parameters to learn the rule —
independently of the one-sided cost. n₀·w_nogo leaks of +0.9/+1.1 appear after release in klein s2/s3
(forbidden under σ₁; s2's σ₁ residual jumps to 0.39 after GNG). Free nets show the same leak (s0 +1.66).

### §37d — The symmetry ledger (`scratchpad/sym_violation2.py`, `sym_ledger_fig.py`)
v_σ = ‖F(Dκ) − D F(κ)‖/‖F‖ over the disk |κ| ≤ 1.5 (NOT the unit circle: there the field is ≈ 0 on the
ring and the ratio is inflated), at init (rebuilt from config+seed), DPA, GNG, Dual:

| nets | init σ₁ / σ₂ | DPA σ₁ / σ₂ | GNG σ₁ / σ₂ | Dual σ₁ / σ₂ |
|---|---|---|---|---|
| free recipe7 (8) | 0.03–0.12 / **0.000** | 0.09–0.70 / 0.11–0.95 | 0.15–0.73 / 0.14–0.90 | 0.20–0.42 / 0.40–1.06 |
| σ₁ tied in DPA (4) | 0.000 / 0.15–0.58 ⁽¹⁾ | 0.01–0.04 / 0.52–0.96 | 0.01–0.11 / 0.47–1.01 | 0.14–0.29 / 0.51–1.09 |
| V tied in DPA (4) | 0.000 / 0.000 | 0.02–0.05 / 0.02–0.05 | 0.05–0.39 / 0.07–0.13 | 0.17–0.40 / 0.60–0.75 |

⁽¹⁾ `symmetrize_init("pair")` copies the first half of n₁ onto the second, so ⟨n₁⟩ ≠ 0 (−0.04…−0.17)
and σ₂ is broken by the pair init itself. Readings: (i) the DPA objective is V-invariant, yet the free
nets break V during DPA in a seed-dependent way (spontaneous, from a 3–12% finite-N seed); (ii) GNG
barely moves the autonomous field — the rule lives in the input columns, which the autonomous field does
not see, so the wells stay on the line; (iii) Dual breaks σ₂/σ₃ hard (0.4–1.1) while σ₁ only drifts
(0.14–0.42). (iv) ★ **The two recipe7 seeds that end one-up-one-down (s0, s3) are the two that left DPA
with the inversion most intact relative to the pair exchange**: v_σ₂/v_σ₁ after DPA = 0.22, 0.30 (fails)
< 0.49 (s4, the most lopsided success, 0.39σ) < 1.06 < 2.2 < 2.5 < 4.4 < 9.0. n = 8; the ordering among
successes is not clean. Pattern, not law.

### §37e — Verified premises (recorded so the next reader need not re-derive them)
- DPA delay decision target is NaN in all runs used (`dpa_prelick_free=True`, `dpa_nolick_weight=0`):
  generated trials show zero pre-sample, NaN sample→test, ±1 in the response window; memory ±1 for 0.5 s
  after the sample then NaN. The DPA objective is invariant under the whole Klein group.
- Channels: 0 A, 1 B, 2 C, 3 D, **4 go (+cue)**, 5 nogo (`go_on_rwd_input=False`).
- Lineage of the target net: s1_dualscan_7 = DPA of s1_lamscan_7 (identical to s1_recipe7's DPA — same
  seed and config) → GNG of s1_gngscan_7 → Dual. σ(n₁) 3.58 (DPA) → 5.04 (Dual); the artifact's 3.68 was
  a typo.
- `sweep_lif_mirror_dpa` had ⟨n₁⟩ = −0.1 built into the init: its depths are the bias, not the tie; it
  is a control for the tie (tied 4/4 level pairs, untied 3/4 asymmetric or missing a well).
- Dropped from the artifact: the `mn_mirror` pairs plot (its mixture fit found a cross, not the lattice,
  and it added nothing the block check does not show).

### §37f — Companion artifact: the derivations (2026-09-21)
"Klein Group Derivations", https://claude.ai/artifact/TVnuduX42gSBzCWvxDeSyd — group/action/orbit/
character definitions recalled; the four characters of V as the sign matrices; Theorem 4.2 (field
equivariance incl. bias and noise); orbit–stabilizer → the counting rule and pinning; Theorem 6.1 (even
part of the field for φ = c + odd: 2c⟨n⟩ + bias term; c = ½ for lif, 0 for tanh) + Lemma 6.2 (Gaussian
averaging keeps the form) + Theorem 6.3 (ensemble condition, any φ); Theorem 7.1 (isotypic
orthogonality ⇒ zero cross-overlaps AND deafness, one theorem); the 4-block parametrization derived from
(4.1) and the Reynolds projection (2.1) = `project_symmetry`; Theorem 9.1 (gradients of an invariant loss
are equivariant; Adam preserves it) + the per-term invariance table of the three objectives; O(2) in the
population limit → radial field, λ_c = 1/(gφ'(0)) = √(2π) = 2.507 for lif, exact zero tangential
eigenvalue (differentiate the equivariance along the orbit), even-harmonic corrugation with alternating
attractor/saddle zeros. Source: `docs/artifacts/symmetry/derivations.html` (MathJax SVG from cdnjs); the main note's source and
figures are in `docs/artifacts/symmetry/` too (`symmetry.src.html` + `fig/` → `python build.py` → publish).
Also fixed today: `symmetrize_init` does NOT preserve λ exactly — the copy gives the prototype block's
overlap (5.8/7.3 for λ = 7 in one seed); docstring, architecture.md and the artifact corrected.
