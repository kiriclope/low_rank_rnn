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
