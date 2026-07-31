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
