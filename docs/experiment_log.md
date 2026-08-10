# Experiment Log

Chronological log of all sweeps. Each entry: config highlights, key result, status.

---

## Phase 1 — Baseline and basic strategies

### sweep_fixedW
- `use_fixed_weights=True`, `fixed_weight_scale=0.8`, `gng_nogo_weight=2.0`
- `freeze_input_stages=["dual"]`
- Result: partial improvement over baseline.

### sweep_nogo2
- `use_fixed_weights=False`, `gng_nogo_weight=2.0`, `go_hinge_thresh=None`
- `freeze_input_stages=["dual"]`
- Result: nogo weighting helps but go bimodality persists.

### sweep_goH1
- `go_hinge_thresh=1.0`, `gng_nogo_weight=2.0`, `freeze_input_stages=["dual"]`
- Result: hinge caps rank-1 eigenvalue growth during GNG.

### sweep_frzGNG ← key milestone
- `freeze_input_stages=["gng","dual"]`, `freeze_gng_input_during_dpa=True`
- AdamW, scheduler, `gng_nogo_weight=2.0`, `go_hinge_thresh=1.0`
- **Key finding**: seed 5 found compositional solution with real (decoupled) κ-Jacobian
  eigenvalues after DPA.

### sweep_adam / sweep_adam2
- Same as frzGNG but `optimizer="adam"`, `use_scheduler=False`

---

## Phase 2 — Dual stage tuning

### sweep_noHinge / sweep_noHinge2
- Copy DPA+GNG from sweep_adam, retrain Dual without go hinge.
- Result: almost half-ring U-shape, but κ₁ tips are stable wells (not saddle points).

### sweep_cont
- Reload after_dual, run 100 more Dual epochs.

### sweep_hingeGNG
- Go hinge applied in GNG stage specifically.

### sweep_nogo1 / sweep_nogo5
- Copy DPA+GNG, retrain Dual with `gng_nogo_weight=1.0` / `5.0`.

### sweep_cue5
- `cue_scale=5.0` — stronger GNG cue signal.

---

## Phase 3 — Nonlinearity exploration

### sweep_relu
- `nonlinearity="relu"`, `cue_scale=2.0`, `gng_nogo_weight=2.0`, seeds 0–9
- `freeze_input_stages=["gng","dual"]`, `freeze_gng_input_during_dpa=True`
- **Finding**: No ring after DPA (relu asymmetric). Crisp go/nogo κ₁ attractors after Dual.
- Figures: `results/figures/sweep_relu/` (XLIM=±5)

### sweep_relu2
- Same as sweep_relu + 100 more Dual epochs from expert checkpoints.

### sweep_softplus
- `nonlinearity="softplus"`, seeds 0–9
- No saturation → no ring.
- Figures: `results/figures/sweep_softplus/` (XLIM=±5)

### sweep_tanh_reg
- `nonlinearity="tanh"`, `gain=2.0`, `freeze_rank0_dual=True`, `kappa1_reg_weight=1.0`
- `freeze_input_stages=["gng","dual"]`, `cue_scale=2.0`, seeds 0–4
- Goal: tanh ring preserved by freeze; κ₁ kept sub-critical by regularizer.
- Figures: `results/figures/sweep_tanh_reg/` (XLIM=±2)

### sweep_erf
- `nonlinearity="erf"` (Brunel LIF approximation — odd, saturates), `gain=2.0`
- `freeze_rank0_dual=True`, `kappa1_reg_weight=0.0`, seeds 0–4
- Figures: `results/figures/sweep_erf/` (XLIM=±2 — within ±1.2)

### sweep_relu_struct
- `nonlinearity="relu"`, `init_style="structured"`, `gain=2.0`, `freeze_rank0_dual=True`
- Hypothesis: structured init still can't give ring with relu (asymmetry is intrinsic).
- Figures: `results/figures/sweep_relu_struct/` (XLIM=±5)

### sweep_elu
- `nonlinearity="elu"` (ELU), `gain=2.0`, `freeze_rank0_dual=True`, seeds 0–4
- Hypothesis: negative saturation → stable −κ*; positive linear → crisp go.
- **Result**: ring still failed (positive side unsaturated).
- Figures: `results/figures/sweep_elu/` (XLIM=±5)

### sweep_lif
- `nonlinearity="lif"` (Gaussian CDF, φ∈[0,1]), `gain=3.0`, seeds 0–4
- **Result**: vanishing gradients; dual stage completely failed (loss flat ~1.0).
- Root cause: gain=3 → saturated everywhere → φ' ≈ 0.
- Figures: `results/figures/sweep_lif/` (XLIM=±1.5)

### sweep_lif2
- `nonlinearity="lif"`, `gain=2.0` — lower gain to widen linear regime.
- **Result**: DPA converges (4/5 seeds); GNG partial; dual stuck ~0.8–1.1, no convergence.
- Root cause: one-sided [0,1] range → can't represent negative DPA targets.
- Figures: `results/figures/sweep_lif2/` (XLIM=±1.5)

### sweep_lif_sc
- `nonlinearity="lif_sc"` (rescaled LIF, φ'(0)=1), `gain=2.0`
- GNG converges faster (several seeds stop early). Dual still fails same way as lif2.
- Figures: `results/figures/sweep_lif_sc/` (XLIM=±1.5)

---

## Phase 4 — Structured init + gain separation

### sweep_tanh_struct
- `nonlinearity="tanh"`, `init_style="structured"`, `gain=1.5`
- `freeze_rank0_dual=False`, `kappa1_reg_weight=0.0` (no explicit protection)
- With `memory_lambda=0.8`: gain×λ₀ = 1.2 > 1 (ring) and gain×λ₁ = 0.75 < 1 (sub-critical κ₁)
- Structure enforces the separation — no regularizer needed.
- Status: complete. Figures: `results/figures/sweep_tanh_struct/` (XLIM=±1.5)

---

## Phase 5 — Static backbone & EI/STP (toward lower-plane wells)

Full detail in `docs/ring_lowerplane_log.md` (§9 static backbone, §11 EISTP). Summary of the
key sweeps:

### sweep_tanh_static (+ _sparse, _ng0*, _hinge variants)
- `model_type="lowrank"`, `use_fixed_weights=True`, `orthogonalize=False`, tanh, gain 1.0.
- Best retention yet (after_gng≈0.89, dual_dpa 1.0) BUT symmetry intact (odd tanh + linear
  backbone ⇒ ± pairs, mean κ₁=0). `_sparse`: `fixed_weight_sparsity=0.1` — identical result.
- Conclusion: static backbone = a *retention* ingredient, not a *lowering* one.

### sweep_ei_v1 / v2_full / v2_stp  (`model_type="ei"`, EILowRankModel — FAILED)
- Dale backbone + additive low-rank (E→E or full graph) ± STP. Memory **collapses to origin**
  (after_gng/dpa ≈ 0.55–0.63). The additive-on-balanced-backbone design is the failure;
  full-graph low-rank and my first STP both diverged/stuck. Superseded by EISTPModel.

### sweep_eistp (`model_type="eistp"`, `eistp_lr_scale="N"`) — chance
- NeuroFlame port, N=1000/K=125, hinge targets, 60 & 100 epochs. **DPA at chance (0.50)** at every
  stage; κ1 encodes the sample then decays over the delay (transient). Cause: `/N` ⇒ memory-mode
  gain g_mem≈0.015 (dead). GNG (immediate decision) learns fine (0.96).

### sweep_eistp_sqrtK (`eistp_lr_scale="sqrtK"`) — ★ BREAKTHROUGH
- Same as above but low-rank divisor `/√K` ⇒ g_mem≈1 (critical), STP-gated.
- **3/5 seeds perfect**: after_dpa/dpa 1.0, after_gng/dpa 0.92, dual_dpa 1.0, dual_gng 1.0.
  **2/5 NaN** (supercritical runaway — stabilise with lower lr / clip / lr_ini<1).
- κ1 **holds flat at ±10 through the delay** (persistent memory); flows show **Go upper-plane,
  NoGo lower-plane** wells. First model to give both. Figures: `results/figures/sweep_eistp_sqrtK/`
  (eistp flows via `ei_flow.py`, default magma style).

### sweep_eistp_sqrtK_stab — clean 5/5 (stabilised) ← matched-init reference
- Same as `_sqrtK` + anti-runaway combo: rate cap `eistp_r_max=200` (≈5× the ~43 operating peak),
  NaN-skip in `Optimization`, lr 0.1→0.05, grad_clip 1.0→0.5.
- **5/5 seeds** (was 3/5): after_dpa/dpa 1.0, after_gng/dpa 0.81, dual_dpa 1.0, dual_gng 1.0. No
  divergence; persistence + Go-upper/NoGo-lower wells preserved. Matched init (`lr_ueqv=True`).

### sweep_eistp_rand_clip — random init, no/with clipping ablation
- `eistp_lr_ueqv=False` (random init), `eistp_r_max=500`, lr=0.05. nogo=−1.
- **Random init works**: after_dpa/dpa 1.0 (training builds the m–n correlation from overlap≈0.01).
- **Grad clipping matters**: with NO clipping (first attempt) 1/5 destabilised in Dual and hit a
  NaN-skip edge case (fixed: `_run_epoch` returns nan → graceful stop, not a "zero examples" crash).
  With `grad_clip=1.0`: 5/5 recorded, dual_dpa ≈0.96, but 2 seeds still diverged in **Dual** (handled
  gracefully, best-state kept). Instability is now isolated to the Dual stage. NoGo well lower-plane.

### sweep_eistp_rand_clip_ng0 — random init + nogo_target=0 (the NoGo-well knob)
- Same as above + `--nogo_target 0.0`. Clean **5/5**, dual_dpa **0.987**, after_gng/dpa 0.84 — more
  stable than nogo=−1. Flows: the **NoGo well moves to the κ₁≈0 midline** (vs lower-plane at nogo=−1)
  → the NoGo well depth is set by the target value.

### sweep_eistp_jstp5_lr01 — j_stp=5 + lr=0.01 ★ best DPA retention yet (2026-06-23)
- Random init, nogo=−1 (lower-plane wells), sqrtK, N=1000/K=125, 5 seeds, 100 ep/stage. Only change
  vs `_rand_clip`: **`j_stp` 1.0→5.0** (5× recurrent gain incl. memory mode) and **lr 0.05→0.01**.
- Clean **5/5, no NaN** — the 5× gain did *not* destabilise (rate cap `eistp_r_max=500` + `grad_clip=1.0`
  + the gentler lr held it). after_dpa/dpa 0.997, dual_dpa **1.0** all seeds.
- **after_gng/dpa = 0.93** (sem 0.034) — **best DPA-through-GNG retention in the project** (vs 0.81
  for `_sqrtK_stab`, 0.84 for the nogo=0 ref). Lower-plane NoGo wells visible in expert flows.
- Figures: `results/figures/sweep_eistp_jstp5_lr01/` (summary + `ei_flow` κ-plane flows). NOTE:
  `plot_sweep` per-run trajectory/scatter crash on eistp (`EISTPModel` has no `alpha` — analytic
  κ-reduction path unsupported; summaries + `ei_flow` flows are fine).

### sweep_eistp_ablate_all — ÷N_E ('all') works in the original regime (2026-06-30)
- The original notebook uses `TRAIN_SCALE='all'` (÷N_E) = our `lr_scale="N"` (the "dead" setting).
  Reproduced it **working** by matching the notebook regime: `lr_scale="N"`, **lr=0.1**, **no grad
  clip**, `j_stp=1`, 200 DPA epochs. Clean **5/5**: DPA 1.0, after_gng/dpa 0.91, dual_dpa 0.999.
- ⇒ "÷N is dead" was a **regime artifact** (lr≤0.05 + clip throttle the ‖m,n‖ growth), not a scaling
  barrier. `'all'`(÷N_E) and `'sqrtK'`(÷√K) are two routes to the same fixed point. See
  `ring_lowerplane_log.md` §11e.

### sweep_eistp_frozen — frozen-input ablation (eistp_init_noise=0)
- Same regime as `_ablate_all` + `eistp_init_noise=0` → fully deterministic frozen forward (no
  resampled noise across epochs). **5/5**, DPA 1.0, dual_dpa 0.999, after_gng/dpa 0.87.
- **No convergence speedup** (~150 DPA epochs, same as resampled) and no generalization loss (fresh
  eval still 1.0) ⇒ a frozen dataset is NOT why the notebook converges in ~10 epochs.

### sweep_eistp_ueqv_adam — faithful notebook init (m=n) + plain Adam (2026-07-01)
- Full notebook recipe: `eistp_lr_ueqv=True` (**m = n at init**, NeuroFlame `LR_UeqV=1`; init overlap
  `n^Tm/N ≈ +1.0`, so a nonzero memory mode from step 0), `low_rank_scale=1.0` (LR_INI), `lr_scale="N"`
  (÷N_E = `'all'`), **plain `Adam`** (no weight decay), lr=0.1, no grad clip, 200/100/100 epochs,
  squared-hinge DPA (`ThresholdLoss`). N=1000/K=125, nogo=−1. Clean **5/5**.
- DPA 1.0, **after_gng/dpa 0.843 ± 0.072**, after_dual/dual_dpa 0.998, dual_gng 0.991.
- Note the DPA loss *starts* at ~3.6 (not 1.0): the squared hinge is summed over 2 channels × the
  pos/neg/zero target windows, each ≈1 near `pred≈0` — it's the multi-component sum, not squaring
  amplification (confirmed by the linear variant starting at ~the same value).
- Figures: `results/figures/sweep_eistp_ueqv_adam/` (full set via auto-routed eistp sim FP + flow).

### sweep_eistp_ueqv_adam_linhinge — linear-hinge A/B vs the above (2026-07-01)
- Identical to `sweep_eistp_ueqv_adam` except the DPA `ThresholdLoss` uses a **linear margin**
  `relu(thresh−·)` instead of squared (new `hinge_squared=False` config / `--hinge_squared 0`). 5/5.
- DPA 1.0, **after_gng/dpa 0.886 ± 0.076**, after_dual/dual_dpa 0.999, dual_gng 0.984.
- **No meaningful difference from squared** — every metric within one SEM; near-identical κ-plane
  fp_scatter. Both hinges share the same zero-loss margin region → same fixed point; the shape only
  changes the transient + loss scale. Squared ⇒ bigger early gradients; linear ⇒ notebook-matching
  start (~1.9 at a wrong-side pred), gentler grads.
- Ops note: 2 seeds (s2,s4) OOM'd when both sweeps ran concurrently on 2 GPUs (10 BPTT jobs); re-ran
  solo → clean 5/5. Plotting requires `MPLBACKEND=Agg` on this box (DISPLAY set → matplotlib hangs
  on a dead X server at TCP :6013).

### E→E connectivity × nogo_target grid — what protects the DPA memory (2026-07-02)
Two new E→E flags (`eistp_lr_additive`, `eistp_dense_cee` — CLI `--lr_additive`, `--dense_cee`),
each swept at nogo=−1 and nogo=0, all else at the `sweep_eistp_ueqv_adam` recipe (m=n init, plain
Adam lr=0.1, ÷N_E, squared hinge, N=1000/K=125). 5/5 each. after_gng/dpa (DPA retention through GNG):

| E→E form | W_EE | nogo=−1 | nogo=0 | sweep dirs |
|---|---|---|---|---|
| **multiplicative** (faithful) | `C·(1+lr)` | **0.843 ± 0.072** | **0.925 ± 0.046** ★ | `sweep_eistp_ueqv_adam` / `_ng0` |
| additive (sparse) | `C + lr` | 0.534 ± 0.062 | 0.636 ± 0.067 | `_additive` / `_additive_ng0` |
| dense + additive | `1/N_E + lr` | 0.586 ± 0.113 | 0.679 ± 0.073 | `_dense_add` / `_dense_add_ng0` |

- **Multiplicative coupling is the load-bearing mechanism.** Only `C·(1+lr)` (low-rank *riding on* the
  sparse STP synapses) holds DPA through GNG (0.84–0.93); both **additive** forms — adding the low-rank
  as a separate term, sparse `C+lr` or dense `1/N_E+lr` — collapse retention to ~chance (0.53–0.68),
  because GNG training freely rewrites the un-gated low-rank. Confirms the old `EILowRankModel`
  intuition ("low-rank added to a balanced backbone loses persistence") now A/B'd cleanly.
- **nogo=0 helps retention in every connectivity** (+0.05…+0.10 vs nogo=−1); best overall is
  **multiplicative + nogo=0 = 0.925**, the strongest DPA-through-GNG retention in the project.
- DPA 1.0 and after_dual/dual_dpa ≈1.0 in all six (Dual always recovers DPA); the differentiator is
  purely GNG-stage retention. Dense+additive early-stops fast (high-gain regime hits stop_loss quickly).
- Flows/scatters built with the matching `W_EE` (verified: additive model → 234k dense off-C weights;
  `_build_model`/`ei_flow.build_model`/`plot_sweep` all thread `lr_additive`+`dense_cee`). Figures:
  `results/figures/sweep_eistp_ueqv_adam_{additive,dense_add,ng0,additive_ng0,dense_add_ng0}/` (111 each).

**Stabilisation knobs (all in EISTPModel / Optimization, on by default in make_configs):**
`eistp_r_max` (rate cap), `grad_clip_norm` (keep ≥0.5), NaN-skip + graceful-epoch-divergence in
`Optimization`. `eistp_lr_ueqv=False` (random) ≈ as good as matched init.

---

### ★ Vanilla isolation — the ring→two-low-wells breakthrough (2026-07-05)
Goal (docs/ring_lowerplane_log.md §13): two **isolated** A/B memory wells, both at κ₁<0. Confirmed
the §12 two-ingredient theory — LOWER (directional break) + ISOLATE (g·λ₁→1) — in a live net.

| sweep | recipe | isolation? | result |
|---|---|---|---|
| `sweep_curriculum` | DPA→GNG→Dual-paired→Dual | ✗ | best lowering (held-κ₁=−0.44) but ring (~3 wells) |
| `sweep_recscale` | trainable per-mode `rec_scale` | ✗ | net grew s₁→1.4 (wants strong decision); ring |
| `sweep_slowtau` | τ×{1,2,4} | ✗ | g·λ₁ *grew* with τ; falsifies the slow-transient idea |
| `sweep_fasttau` | τ×{1,½,⅓}, tanh+attention | ✗ | both wells κ₁<0 but ring; τ<0.3 breaks convergence |
| **`sweep_kappa1reg`** | **fast1 + Dual `kappa1_reg_weight`** | **✓** | **w=1 → g·λ₁=1.0, two isolated wells κ₁≈−0.9** |

**`sweep_kappa1reg` (the winner).** Base = fast1: τ=0.3, `nonlinearity="tanh"`, `attention_input=True`,
`nolick_weight=0.5`, `hinge_gng=True`, `decision_lambda=0.25`, `memory_lambda=0.8`, gain 2, 100/100/100.
Arms `kappa1_reg_weight ∈ {0,1,3,6}` (reg0/1/3/6), 3 seeds.

| w | g·λ₁ | #wells | mem-κ₁ | DPA | match/nonmatch | go | nogo | Dual loss |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.47 | 3 (ring) | −0.1 | 1.0 | 0.97 | 1.0 | 0.81 | 0.19 |
| **1** | **1.01** | **2** | **−0.9** | 1.0 | **1.0** | 1.0 | **0.79** | **0.11** |
| 3 | 1.01 | 2 | −0.7 | 1.0 | 1.0 | 1.0 | 0.43 | 0.21 |
| 6 | 1.01 | 2 | −0.5 | 1.0 | 1.0 | 1.0 | 0.15 | 0.36 |

- **w=1 = operating point:** isolation + deep low wells + task intact (only nogo softens to 0.79, the
  criticality cost). More penalty buys no more isolation (g·λ₁ already 1) — only kills nogo. Next: finer
  sweep w∈{0.5,1,1.5,2}.
- Robust to input noise (`--field_input_noise`, noise-averaged field): reg1 wells hold (shift ~0.1).
- Figures: `results/figures/sweep_kappa1reg/` (flows re-rendered noise-averaged).

**Loss changes (this session, see docs/running.md "Isolation recipe"):** `hinge_gng` single switch
(True=hinge all 3 stages / False=MSE); go/nogo asymmetric one-sided (nogo≤−1 mem / ≤0 post-cue),
match/nonmatch symmetric ±th (same in DPA + Dual); `nolick` excludes the sample window; per-side
`dual_go`/`dual_nogo` accuracy; trainable `rec_scale`; `--field_input_noise` noise-averaged flows.

---

## Config reference (common base, current sweeps)

```python
shared = dict(
    noise=1.0, model_noise=0.0,
    cue_on_go_input=True, go_on_rwd_input=False,
    freeze_input_stages=["gng", "dual"],
    freeze_gng_input_during_dpa=True,
    nogo_target=0.0, cue_scale=2.0,
    stop_loss=0.1, dual_loss="separated",
    epochs_dpa=100, epochs_gng=100, epochs_dual=200,
    gng_nogo_weight=2.0, go_hinge_thresh=1.0,
    optimizer="adam", use_scheduler=False,
)
```

---

## 2026-07-13 — self-gain / task-side sweeps (all tanh unless noted, gain 1.0, noise 0.25, nolick 0.5, reg 0 unless noted)

Base = §13 fast1 recipe (structured init, attention, hinge_gng, tau 0.30, adam, 100/100/100). Init
self-gains held fixed across arms via λ-scaling (`memory_lambda=1.6/gain`, `decision_lambda=0.5/gain`)
where a gain axis is swept. Read-out via `bifurcation_probe.py` (g·λ₀/g·λ₁, wells, nogo). Science in
`docs/ring_lowerplane_log.md` §14.

| sweep | axis | key result |
|---|---|---|
| `sweep_gainscan` | gain {0.5,1,1.5} | g·λ₁ tracks gain (1.8→2.8) but stays supercritical; wells at κ₁≈0, seed-inconsistent. Self-gain is task-locked. |
| `sweep_noise_g10` | noise {0.1,0.25,0.5,1} @ gain 1 | high noise → higher g·λ₁ (robustness); low noise plateaus ~2.2 / collapses to single nogo well. No sweet spot. |
| `sweep_nolick0` | reg {0,0.5,1} @ nolick **0** | without nolick the isolated wells float UP to κ₁≈+0.7 (lick side, 2/3 seeds), nogo collapses. nolick does the lowering. |
| `sweep_nolick1` | reg {0,0.5,1} @ nolick **1** (Dual-only) | reg1 seed 2 nails two low wells at (±1,−0.9) nogo 0.90; but 2/3 seeds lose the memory attractor entirely (isolation fragile). |
| `sweep_nogopole` | `nogo_hinge_thresh` {0,−0.5,−1} | raising toward 0 does NOT remove the nogo pole — it becomes the sole global attractor (memory poles → saddles). The pole = decision's autonomous down-well. |
| `sweep_relu_noreg` | relu, reg 0 | relu → asymmetric spiral, **0 point attractors** (near-marginal integrator), broken A/B symmetry. Not ring-capable. |
| `sweep_relu_ml` | relu, memory_lambda {0.6..1.6} | trained g·λ₀ regrows to ~1.6–2.0 regardless; Re₀>0 everywhere (unstable), no attractors. Verified reduced field = exact 2-timescale stability (h is the fast var, α_rec>α). |
| **`sweep_mem`** ★ | **memory_lambda {1.6,2.5,3.5,5}** | **mem50 (memory_lambda=5): two isolated low wells at (±1,−0.8), robust supercritical decision (no isolation), DPA/go=1.0, nogo up to 1.0, 2/3 seeds.** The winning robust route. |
| `sweep_ramping` | `ramping_gng=True`, decision_lambda {0.1,0.5} | cue-driven decision did NOT go subcritical (g·λ₁~2.2); go/nogo delay-memory forces supercritical. Nogo often worse. |
| `sweep_cuescale` | cue_scale {2,4,6,8} | stronger cue amplifies decision poles + hurts nogo (0.91→0.71). Ceiling: cue-driven κ₁ peaks ~cue 2 then decreases (strong cue overrides recurrence). |
| `sweep_n1024` | **N=1024** @ cue_scale 2, 11 seeds | doubling N (vs 512) leaves self-gains unchanged (N-independent init) but firms up the statistics: **DPA retention rock-solid (dual_dpa mean 0.985, min 0.915); nogo the weaker axis (dual_gng mean 0.883, range 0.78–0.95)**. The earlier 3-seed nogo floor (0.65) was a small-sample artifact — across 11 seeds the floor is ~0.78. |

**Takeaway:** `sweep_mem` mem50 (deep memory, robust decision, no isolation) is the two-low-wells recipe.
More neurons (`sweep_n1024`) don't change the geometry (self-gains are N-independent) but shrink seed variance: DPA is essentially perfect, nogo reliably 0.82–0.95.

**Plotting infra fixes (2026-07-13):** `plot_sweep._build_model` now reads `hidden_size` from each
run's `config.json` (was hardcoded 512 → crashed loading N≠512 checkpoints); `RunMeta.hidden_size`
added. `--meanflow_overlay kde` now writes `fp_meanflow_{stage}_kde.pdf` (own filename, never clobbers
the scatter-overlay `fp_meanflow_{stage}.pdf`).

### `sweep_n1024_mem5` — winner combo (N=1024 × mem50 depth), 11 seeds (2026-07-13)

`memory_lambda=5` at N=1024 (else = `sweep_n1024`). Deep memory raises `g·λ₀` from ~2.1 to **~2.7–3.0**
(g·λ₁ stays ~2.0–2.4) and **splits the central nogo pole into two low A/B wells** in the good seeds
(e.g. s0 (−0.7,−0.3)&(+0.9,−0.8); s4 nogo=1.00). Accuracy: **DPA perfect** (dual_dpa mean 1.000, min
0.999); **nogo lifted but still seed-variable** (dual_gng mean 0.908, range 0.42–1.00). Two seeds
regressed (s3=0.42, s1=0.73) — their wells landed in the **lick** half-plane (κ₁>0), so nogo collapses.
So deep memory reliably makes the two isolated wells but their κ₁ side is still seed-dependent.

### DPA/GNG accuracy vs well location — two new analysis tools (2026-07-13)

Both operate on trained checkpoints (any sweep), reusing `bifurcation_probe.load_run` + `dpa_stats`.

- **`dpa_well_accuracy.py`** — d′ (graded match/nonmatch or go/nogo separation) vs memory-well location,
  per sample (A/B) / trial-type (go/nogo), for each stage. Key result: **DPA decision is a fully-settled
  attractor** — match/nonmatch clouds are *disjoint* (AUC=1.0, overlap=0.0) with **d′≈44–117σ** regardless
  of well κ₁, so hard/AUC/overlap accuracy all saturate; only d′ (unsaturated) shows any location trend.
  GNG go/nogo is also perfectly separable but **~10× thinner margin (d′≈5–32)** — the fragile axis.
  Encoding split confirmed: **DPA sample→κ₀, GNG go/nogo→κ₁**. Per-seed d′-vs-location is weakly tuned
  (R²≈0.2); binned/poly tuning fits added (`_add_tuning`, `--poly_deg`).
- **`perturb_well_sweep.py`** — CAUSAL d′(well-location) curve from ONE network by sliding the wells and
  re-measuring, across stages. Modes: **`input`** (default) tonic drive on the go channel — moves both
  wells' κ₁ together, **weights & pairing untouched** (the clean manipulation); `n1`/`m0` weight
  perturbations (antisymmetric). Findings: **expert d′ sharply peaks (~120) at the trained κ₁≈−0.5 and
  goes negative (decision inverts) when the well is pushed to either extreme**; **Dual training sharpens
  the DPA margin ~20×** over the DPA stage. A closed-form **LDA readout-rotation SNR**
  `d′(ε)=(Δμ_n+εΔμ_m)/√([1,ε]V[1,ε]ᵀ)` is exact for readout rotation (<0.5% vs reprojection) and
  approximates the physical perturbation at the expert stage (corr 0.53) but not at DPA (dynamics
  reorganize). Caveat: the go-channel knob is untrained at the DPA stage (no learned κ₁-input there).
  A/B curves differ by seed **asymmetry**: lopsided s0 gives A/B d′ 122 vs 72; near-symmetric s6
  collapses them (60 vs 61) near the trained point, with residual split at large drive from the
  fixed-input-direction × nonlinearity.

### RANK-3 role split (κ0 sample / κ1 gng / κ2 action) — new architecture (2026-07-15)

**Code (all guarded on `rank>=3`; rank-2 unchanged):** `src/init.py` installs a 3rd self-sustaining
eigenmode for κ1 wired from the go(+)/nogo(−) input channels. `src/tasks.py` GNG & Dual generators
split the held memory (κ1) from the action (κ2); DPA already produced the right layout. `sweep.py`:
`gng_lambda` field; structured init passes `mem=0,gng=1,out=2`; progressive freeze locks κ0 after DPA,
κ0+κ1 after GNG (`dual_mem_freeze`); the κ1-reg regularizer now targets the **last** column (κ2 action
for rank-3, κ1 for rank-2). `plot_sweep.py` reads `rank`/`hidden_size` from config so rank-3 loads.

**`sweep_rank3` (structured init, 4 seeds, memory_lambda=gng_lambda=5, decision_lambda=0.5, no reg):**
the split works structurally — autonomous field holds the two memories with κ2 at rest, **DPA retained
(dual_dpa 0.76–1.00), go perfect (1.00)** — but **nogo collapses (0.00–0.07)**. Cause (from `rank3_flow`):
the action mode κ2 trained supercritical (g·λ2 ≈ 2.4–2.9 from init 0.5), so under the cue there's a
**single global lick attractor** (κ2≈+1) that the negative-κ1 nogo memory can't veto. Convergence: DPA/GNG
early-stop ~20-45 ep; the combined Dual loss never hit stop_loss in 100 ep (still descending).

**`sweep_rand3` (random init, no reg, epochs 150/150/300):** control for whether the role split emerges
without the structured scaffold. Same nogo-collapse family (action mode goes supercritical).

**`rank3_flow.py` (new tool):** 3-D fixed-point flow portraits, rank-2 conventions (magma/white/cyan),
8 dual input-driven condition rows × 3 κ-plane columns. brainpy `SlowPointFinder` (jax) for the FPs —
identical attractors to scipy/simulation, `--slow_tol`/`--marg` expose slow-manifold detection; jax
(jit) **adiabatic projection** (relax the off-plane κ to its nullcline per grid point, seeded from the
nearest FP) so streamlines converge onto the projected 3-D fixed points instead of a flat-slice artifact.

**Target-timing revision (`src/tasks.py`, commits `22ae9e4`→`a48e2a6`→final 2026-07-20 — Leon's edits):**
the clean "κ1 = pure held memory, κ2 = pure action" split no longer holds; the current rank-3 layout is:
- **κ0 = sample memory** — held ±1 only over its behaviourally-relevant window (GNG: free/NaN throughout;
  DPA: sample→test delay; Dual: `n_off[0]:n_on[3]`, sample-off to test-on), NaN during its own stim.
- **κ1 = gng-memory→action** — holds go(+1)/nogo(−1) through the gng delay, then expresses the action in
  the **reward pulse** `[n_off_cue, dt)` (`dt = n_off[2] + ½·test_window`), go→`go_target`, nogo→`nogo_target`.
  Same structure in GNG and Dual (`ramping_gng` gives the single-step ramp variant in both).
- **κ2 = lick** — in Dual it's the shared last-channel decision built by the non-rank-3 code: gng-memory
  in the delay + the same reward pulse **+ the DPA match/nonmatch pair decision after the test** (`n_off[3]:`,
  pair→+1/nonpair→−1). The rank-3 block no longer overwrites it. So κ1 and κ2 share the gng part; κ2 adds
  the pairing decision. (In GNG, with no test, κ2 = the go/nogo action only.)

Verified by `scratchpad/plot_task_io.py` → `results/figures/task_io/{dpa,gng,dual}_rank3_io.pdf` (input-channel
heatmap + κ0/κ1/κ2 target traces per condition; NaN=free shown as gaps). Motivation (inferred): give the
*subcritical* κ1 mode a share of the action so the decision no longer rests solely on the supercritical κ2 that
caused the nogo collapse. **Not yet re-swept** — `sweep_rank3`/`sweep_rand3` predate this and used the old targets.

### Baseline drift → subcritical-λ is the fix (2026-07-21)

Chased why the **pre-sample baseline** (t<n_on[0]) sits off-zero and drifts in the κ-trajectories. Three
compounding causes found and fixed, in order of importance:

1. **Deep-λ init makes 0 unstable (the real cause).** With `memory_lambda=gng_lambda=5` the self-gain
   `g·λ = gain·(n·m/N) ≈ 5 ≫ 1` — so κ=0 is a strongly *unstable* saddle before any stimulus, and the net
   slides into a well during the input-free baseline. **Fix = subcritical init `memory_lambda=0.8`** (g·λ≈0.8<1
   ⇒ 0 is a stable FP off-attention). `sweep_sub_nogate` (rank-2, tanh, N=1024, 4 seeds): baseline now **flat
   at 0** AND κ0 memory **held at ±1 through the whole delay** — additive attention (last input channel, 0 at
   baseline / 1 after stim) gates the bistability on its own via the trained `wi` projection. DPA~1.0, go 1.0
   (nogo weak 0.03–0.56, a separate decision-mode issue).
2. **`ThresholdLoss` zero-target was toothless** (`src/train.py`). Zero targets (baselines, nogo rest) were
   scored `relu(|pred|−thresh)²` with `thresh=1.0` → a free ±1 dead-zone. Added **`zero_thresh` (default 0 ⇒
   MSE-to-0)** so baselines are pinned while ±1 keeps its margin. Config: `dpa_zero_thresh`. (Affects ALL
   hinge/ThresholdLoss sweeps.)
3. **GNG generator never pinned the baseline** (`src/tasks.py`). It inits `zeros*nan` and only wrote task
   windows, leaving `[0,n_on0)` free (NaN) for every κ — DPA/dual clamp it, GNG didn't. Added
   `targets[:, :n_on[0]] = 0.0`.

**Multiplicative attention gain-gate — tried and REMOVED.** Hypothesis: attention should multiply the
recurrent self-gain (`g·(1+s·attn)·λ`) so 0 is stable off-attention (baseline) and bistable on. Implemented in
`models.py` + swept (`sweep_sub_gate`, s=2). Result: baseline flat 0 ✓ but κ0 memory **decays during the delay**
(transient, not latched) — the uniform gate on `rec_inputs` destabilises the held state. Additive attention
(nogate) does strictly better (flat baseline + held memory), so the gate was reverted; `models.py` back to plain
`gain·(input+rec)`. **Takeaway: subcritical init + additive attention is the clean recipe** for a stable-0
baseline with persistent memory; no gain-gate needed.

**Plotting/infra this thread:** `plot_sweep.py` now saves **PNG+SVG (no PDF)** via a `save_fig` helper (PNGs
publish straight to the localhost figure gallery — see the `figure-gallery` shared skill) and passes
`target_rank=meta.rank` in `individual_flow` (was defaulting to 1 → IndexError on the new rank-restricted
`generate_dual_trials`). `rank3_flow.py` likewise emits PNG+SVG. Rank-3 sweeps must plot with
`--plots acc traj` (the analytic FP finder is rank-2 only; flows come from `rank3_flow.py`).

Earlier this session (`sweep_r2_newtgt`/`sweep_r3_newtgt`, pre-subcritical): the revised rank-3 targets +
`zero_thresh` lifted rank-3 DPA retention to ~1.0 and recovered nogo on 3/4 seeds (was 0/4); rank-2 held DPA
1.0 with seed-variable nogo. Superseded as the baseline recipe by the subcritical-λ finding above.

**Response-window `dt` fix (`src/tasks.py`, 2026-07-21):** GNG & Dual expressed the go/nogo response over
`[cue_off, dt)` with `dt = cue_off + (n_off[-1]−n_on[-1])/2` = *half the last epoch* — the cue in GNG (0.25 s)
but the test in Dual (0.5 s), so the GNG response was silently half as long. Both now compute a fixed 500 ms
window explicitly: `dt = cue_off + int(round(0.5/timing.dt))` (robust to the timestep). GNG response widened
250 → 500 ms; Dual unchanged (still 0.5 s, now explicit).

### ★ ATTENTION isolates the wells — the mechanism, by ablation (2026-07-21)

Factorial ablation on the subcritical rank-2 recipe (N=1024, tanh, gain=1, `memory_lambda=0.8`,
`decision_lambda=0.5`, `cue_scale=2`, **fixed lr** `use_scheduler=False`, adam 0.01). All autonomous/
input-driven flows computed with **attention ON** (`ff[-1]=1`, incl. "Autonomous") and at **±2** limits.

| sweep | attention | nolick | DPA ep | seeds | dual_dpa | dual_nogo | autonomous flow |
|---|---|---|---|---|---|---|---|
| `sweep_subcrit` | trained | 0.5 | 100 | 11 | 0.95 | **0.40** | 2 isolated wells κ1≈−0.8 |
| `sweep_nonolick` | trained | 0 | 100 | 11 | 0.95 | 0.21 | 2 isolated wells κ1≈−0.8 |
| `sweep_subcrit_dpa300` | trained | 0.5 | 300 | 11 | ~1.0 | **0.53** | 2 isolated wells |
| `sweep_frzatt` | **frozen** | 0.5 | 100 | 4 | 1.0 | 0.29 | 2 isolated wells |
| `sweep_noatt` | **OFF** | 0 | 100 | 3 | 1.0 | **0.00** | **3-attractor ring** (lick well returns) |
| `sweep_supercrit` (λ0=2.0) | trained | 0.5 | 100 | 4 | ~1.0 | 0.42 | same as subcrit |

**Mechanism (overturns the old mental model).** The two A/B memory wells sit at **κ1≈−0.8 in every
condition** — attention does NOT "lower" them. What attention does is **destabilise the autonomous
lick/go attractor**: with attention the top node (κ1≈+1.7) is a *repeller* and only the two no-lick
wells remain (isolated, no ring); with attention OFF that lick well is a stable *attractor* → the state
falls into it on nogo trials → **nogo = 0** and the old 270° ring/U is back. So **attention is the
isolation lever** we used to attribute to `kappa1_reg` (reg is 0 here yet the ring is gone).

**What each knob actually does (by elimination):**
- **attention = necessary.** OFF ⇒ ring returns, nogo→0. This is THE lever.
- **attention need not be LEARNED.** `frzatt` (wi attention column frozen at random init via new
  `freeze_attention_input` flag) still isolates (nogo 0.29) — any fixed tonic break suffices.
- **nolick ≠ lowering/isolation.** `nnl` (nolick=0) gives the *same* well positions; nolick only buys
  a modest nogo margin (0.40 vs 0.21).
- **init criticality ≠ it.** subcritical (λ0=0.8) vs supercritical (λ0=2.0) → identical geometry
  (self-gains task-locked).
- **longer DPA helps nogo.** 100→300 DPA epochs: nogo 0.40→0.53, DPA retention cleaner. (At 100 ep DPA
  never hit stop_loss 0.1 — floored ~0.34 — because the attention-OFF baseline is unstable, see the
  2026-07-21 baseline note; the residual is the pre-sample κ1 drift, penalised but un-removable.)

**Baseline caveat (unchanged):** attention is on only from stim onset, so the pre-sample window `[0,n_on0)`
runs attention-off where κ=0 is a supercritical saddle → κ1 drifts (DPA-stage figures). The Dual stage
pins its baseline (converges <0.1); the DPA stage does not (that residual is real, not a masking bug).

**Code:** `freeze_attention_input` RunConfig flag (freezes the attention wi column in DPA+GNG; Dual
already freezes all inputs); `use_scheduler=False` (fixed lr — these nets learn better without a
scheduler); `plot_sweep` XLIM/YLIM → ±2 (all FP scatters/flows/mean-flows). Figures published to the
localhost gallery under `rnn/<sweep>/`.

### ★ Windowed transient decisions — decay vs no-decay ablation (2026-07-24)

Decision targets moved from held plateaus to **windowed transients**: a 0.5 s pre-cue hold (go=+1 /
nogo=−1, ending *at* cue onset so nogo sits at −1 *before* the go-push cue), a 0.5 s response right after
cue-off, then an **optional** 0.5 s decay back to 0. DPA pairing (match/nonmatch) expressed for 1 s after
test-off, then optional decay. **GNG nogo is no longer reset on cue onset** — its response line is dropped,
so nogo holds −1 pre-cue, is free through the cue, then decays if decay is on.

**Flag split** (`src/tasks.py` all 3 generators + `sweep.py`): the old `decay_decision` param is renamed
**`windowed_targets`** (the windowed scheme) and a new **`decay_to_zero`** (default True) gates *only* the
explicit decay-back-to-0 lines (gng response + pairing). So `windowed_targets=True, decay_to_zero=False` =
express-in-window-then-free.

**`sweep_win_decay`** (8 seeds, fresh DPA→GNG→Dual 100/100/300). Emergent recipe: rank2, N=1024, tanh,
gain1, `memory_lambda=0.8`, `decision_lambda=0.5`, `cue_scale=2`, attention ON, `nolick_weight=0`,
`nogo_push_memory=False`, `freeze_rank0_dual=True`, `ramping_gng=True`, `windowed_targets=True`, adam
fixed lr 0.01, `nogo_target=0`. Two arms differ ONLY in `decay_to_zero`.

| arm | pairing (none trials) | go / nogo | converged |
|---|---|---|---|
| **no-decay** (`decay_to_zero=False`) | **1.0 all 4** (s0 go-trial pairing 0.75) | 1.0 / 1.0 | **4/4** |
| **decay** (`decay_to_zero=True`) | s1,s3 ≈1.0 · s0 0.59 · s2 1.0 (0.5 on go-trials) | 1.0 / 1.0 | **2/4** (s0,s2 stuck) |

**Result:** dropping the decay-to-0 target makes the pairing converge far more reliably. The match
decision must hold at +1 against the no-lick-biased field; the repeated decay-to-0 target fights that, and
2/4 decay seeds fall into bad minima (s0 val 0.82; s2 val 0.96, loses pairing specifically on go trials).
No-decay is clean 4/4. go/nogo is perfect (~1.0) in both arms regardless. (Measured with the correct
α-scaled `noise_sigma`; raw-noise eval understates badly.)

**Metric fix** (`sweep._dual_accuracy` + `plot_sweep._eval_dual_by_trialtype`): score the pairing in its
**expression window** `[n_off[3], n_off[3]+1s]`, not averaged to trial end (which dilutes across the
windowed decay/free tail → understated, esp. the decay arm). Also fixed the plot_sweep pairing **label** to
come from condition names — it was `y[:,-1,-1]`, which is 0/NaN under windowing → labelled every trial
nonmatch → chance. go/nogo window tightened to the 0.5 s response window (matches `_dual_accuracy`).

Figures: `rnn/{arm_decay,arm_nodecay}/` in the localhost gallery (accuracy-by-trialtype, trajectories,
per-stage flows + KDE mean-flows, ±2). Open: reseed the two stuck decay seeds; the go-trial-pairing ceiling
is the shared-κ₁-axis tension (match=+1 vs no-lick memory) → rank-3 split. See `ring_lowerplane_log.md` §16.

### ★ Decay favours the lower ring; all-stage decision reg (2026-07-25)

**Held-well measurement** (`scratchpad/wells.py`: drive-and-release A/B on 'none' trials, κ₁ read in the
deep delay [7,8]s) on `sweep_win_decay`: the **decay arm puts BOTH memory wells in the no-lick plane
(mean κ₁ −0.73, 4/4 seeds), the no-decay arm lifts them (mean +0.30, 0/4)**. So the decay-to-0 target is
the geometry lever — it forbids κ₁ from resting anywhere but 0 after the window, so the only stable
structure left is the two no-lick memory wells. Convergence cost and geometry benefit are two faces of
one constraint.

**`sweep_win_reg`** (8 seeds) = windowed recipe + **decision-subcriticality reg at ALL stages**
(`kappa1_reg_weight=1`, `w·relu(g·λ_dec−1)²`, new `_kappa1_regularizer` factory applied DPA/GNG/dual-
paired/Dual). Two arms `regnd` (nodecay) / `regd` (decay).
- **regnd**: converged 4/4 (fastest yet) but wells lifted WORSE (κ₁≈+0.70). The reg pinned g·λ_dec≈1.1,
  but the network re-routed the parking into the **memory mode** (its g·λ grew to ~2.1–2.7) → wells
  even higher. nogo degraded (0.70–0.96). *Lesson: reg constrains a mechanism, not the incentive — with
  a free tail, parking is loss-free and the task re-routes it (§14 task-locking, new form).*
- **regd**: reg **fixed the pairing convergence** — match/nonmatch 1.0 on all 4 seeds (incl. the seeds
  that stuck at reg 0), because the marginal decision mode can't build the lick attractor that fought the
  decay. But the failure moved to go/nogo (s0 go 0.43, s2 nogo 0.50; two seeds went complex-eigenvalue
  spiral 1.7±0.5j), and wells flattened to ≈0 (reg also weakens the κ₁-tilted well structure). Tail
  parked near −1 (the ≤−1 pairing-tail bug, below), not 0.
- **★ Loss bug found (fixed):** the Dual pairing decay-to-0 zeros fell into the match/nonmatch hinge's
  `else` branch → trained as **κ₁ ≤ −1** (a hard basement shove on every trial-end), NOT "return to 0".
  New `pin_decay_zeros` (auto = `windowed_targets & decay_to_zero`) pins ALL decay zeros to 0 (MSE),
  gng + pairing, all stages; baseline keeps its own separate term. `dual_dpa`/geometry from every decay
  arm above is contaminated by this — rerun needed.
- **No arm dominates**: decay = geometry lever, reg = pairing-convergence lever, but reg flattens wells +
  destabilises go/nogo per-seed. Figures `rnn/{arm_regnd,arm_regd}/`. See `ring_lowerplane_log.md` §17.

### ★ UnifiedLoss — one value-based loss for all three stages (2026-07-25/31)

Replaced the three stage-specific losses with a single `UnifiedLoss` (`dual_loss="unified"`). Semantics
live in the TARGETS (`tasks.py` windows); the loss only enforces value classes: **+1→one-sided hinge
`relu(1−p)²` (overshoot free) · −1→`relu(p+1)²` · 0→pin `p²` (MSE-to-0) · NaN→free** (+optional nolick).
Kills the whole family of "zeros fall into the wrong branch" bugs by construction. Separately-weighted,
separately-logged terms, each its own masked_mean (short windows never diluted):
`bl` (pre-sample only, its own timing split — SEPARATE from decay) · `gng_pos/gng_neg` (pre-cue holds) ·
`gng_decay` · `rwd_go/rwd_nogo` · `pair_pos/pair_neg` · `pair_decay` · `mem_*` · `nolick`. The decision
channel splits into gng vs pair groups at test onset (`pair_start`).
- **Intended semantic changes vs the old losses** (verified, all else identical): Dual pre-cue **nogo
  hold now enforces ≤−1** (old gentle ≤0); match/nonmatch and go-holds now class-balanced (separate
  pos/neg means → pairing effectively 2× weight; `dpa_weight=0.5` restores old balance).
- **`gng_response` task flag** (default False) re-adds the 0.5 s response window after cue-off
  (go→go_target, nogo→0), scored by the **rwd group** carved out of gng: `rwd_go` (+1 hinge) and
  `rwd_nogo` (0→pin, both-sided) with independent weights → a **go/nogo imbalance knob** (e.g.
  `rwd_nogo_weight=2` to make the not-lick stick against false licks). GNG decay aligned to 1.0 s in the
  dual generator (matched the 3*half GNG-stage edit).
- New RunConfig: `gng_decay_weight, pair_decay_weight, gng_response, rwd_go_weight, rwd_nogo_weight`.
- Full verification battery in-session (equivalences to old losses; each violation → only its component;
  weight routing exact; non-windowed = no-op). Targets figure: `rnn/task_targets/`. Old `dual_loss` modes
  untouched.

### ★ UnifiedLoss feature-isolation sweep — RUNNING (2026-07-31)

Three sweeps, 4 seeds each, `dual_loss="unified"`, windowed, all weights 1, NO reg, fresh 100/100/300.
Feature isolation (differ only in the response window + decay):
- **`sweep_uni_base`**  — `gng_response=F, decay=F` (pre-cue hold + pairing only; the minimal recipe).
- **`sweep_uni_rwd`**   — `gng_response=T, decay=F` (+ the 0.5 s response window: rwd_go/rwd_nogo terms).
- **`sweep_uni_decay`** — `gng_response=F, decay=T` (+ decay-to-0, the lower-ring lever, no rwd).
rwd effect = rwd vs base; decay effect = decay vs base. All free of the ≤−1 tail bug (pinned decays).
Launched into 3 dirs via `--run_filter _base/_rwd/_decay`. **Score when done** (`scratchpad/wells.py` held
κ₁, expression-window pairing/go/nogo, per-stage g·λ, unified `dual_loss_components`).

**Two new optional loss/task knobs added this session** (default-off, prior configs byte-identical):
- **`rwd_nogo_onesided`** (UnifiedLoss): response window scores ONLY the nogo lick penalty `relu(κ₁)²`
  — NO go +1 hinge, NO nogo pin. go response + nogo no-lick value both free (only "nogo must not lick"
  is enforced, per the emergent one-sided philosophy). vs default = go +1 hinge + nogo pin-to-0.
- **`dual_gng_memory`** (task, DUAL only): supervise the go/nogo pre-cue hold or not. False = the go/nogo
  working memory is NOT re-supervised in Dual (must survive on the GNG-learned structure; note
  `freeze_rank0_dual` only protects κ₀, so this tests whether κ₁'s go/nogo memory persists on its own).
Both verified; uncommitted-results status. See `ring_lowerplane_log.md` §17e.

### ★ Unified-loss ladder RESULTS + "decay lowers wells" was the tail bug (2026-07-31)

`sweep_uni_{base,rwd,decay}` all solve the task (pairing 1.0, 4/4). Geometry (held κ₁, deep-delay 'none'):
base +0.14 · rwd +0.19 (2/4 spiral) · **decay +0.02** (2/4 go/nogo degrade). **The old buggy decay arm
gave κ₁≈−0.73; the same recipe under the unified loss (pinned tail, not the ≤−1 shove) gives +0.02** — the
−0.75 was ENTIRELY the ≤−1 pairing-tail bug. So "decay favours the lower ring" was the accidental
basement-shove, not honest return-to-0. Under the clean loss none of the arms lowers the wells (all
straddle 0); decay costs go/nogo for no geometry. Next honest lowering lever = `nolick_weight` or rank-3.
Figures `rnn/sweep_uni_{base,rwd,decay}/`. Detail: `ring_lowerplane_log.md` §17f.

- **NO-MEMORY ladder** `sweep_uni_nomem_{norwd,rwd,1s}` (4 seeds, `dual_gng_memory=False`, decay off,
  differ in response window: none / pin / one-sided) — done, scoring pending. Tests whether κ₁'s go/nogo
  memory survives Dual on the GNG-learned structure without re-supervision.
- **plot_sweep target-overlay bug fixed (§17g):** `RunMeta` now carries the target-scheme flags so
  trajectory dashed-target overlays match the trained scheme (were defaulting to non-windowed → wrong on
  EVERY windowed-run traj figure). Flows/accuracy unaffected. `sweep_uni_*` traj regenerated.

### ★★ First honest well-lowering: nolick + freeze×pressure 2×2 (2026-07-31)

Nolick (`nolick_weight·relu(κ₁)²` over free Dual decision windows, sample+baseline excluded) on the clean
nmrwd base. **2×2 held-κ₁** (deep-delay 'none' trials): frozen/no-nolick +0.06 · frozen/nolick **−0.15
(3/4 both<0)** · unfrozen/no-nolick +0.09 · unfrozen/nolick **−0.16 (4/4)**. All 16 runs task-perfect
(pairing/go/nogo 1.0). **nolick lowers the wells honestly** (no artifact, emergent) — the first intrinsic
lowering; mechanism = sustained no-lick pressure during retention. **Unfreezing κ₀ is safe** (retention
survives on the pairing alone, κ₀ sep ~2.4) but doesn't amplify (−0.16≈−0.15), only makes it consistent
(4/4). ⇒ magnitude capped by nolick weight (0.5), not the DOF; raise nolick for dose-response. Modest
(−0.15 vs old −0.8). Figures `rnn/sweep_uni_{nolick,unfrozen_nonolick,unfrozen_nolick}/`. Detail §17h.

### ★★ Rectifier probe → relu lowers the wells; FP-classifier bug (2026-08-03)

`sweep_relu_cap` (relu ×4), `sweep_softplus_sp` (softplus ×4), `sweep_nogo_pin` (tanh, L1 nogo-pin
`rwd_nogo_l1=True` w∈{0.1,1.0}, ×4). Base = unified loss, unfrozen κ₀, `attention_gated`, nmrwd
(`dual_gng_memory=False`), **no nolick**, `memory_lambda=0.8`, structured init, 100/100/300. New knob
`rwd_nogo_l1` (nogo pin form: `|κ₁|` L1 vs `κ₁²` L2).

- **FP-classifier bug**: `classify_fixed_points` (`marginal_tol=1e-2`) mislabels genuine SLOW attractors
  (map |λ|≈0.99) as "marginal" → `wells.py`/plots drop them; and the analytic finder's `XLIM=±2` box
  misses non-saturating wells (κ₀≈±5–25). Fixed: `wells.py` routes non-saturating φ → grid-sim ground
  truth; `marginal_tol=2e-3` in `wells.py` + `plot_sweep` (`MARGINAL_TOL`); re-render with wide `--xlim`.
  This bug had also mis-scored several earlier reads. Detail: `ring_lowerplane_log.md` §18.
- **relu**: task 4/4; autonomous wells discrete, **all κ₁<0** (mean −0.29, 6/6), 2/4 clean-bistable (2
  asymmetric basins). **First structural (nonlinearity) lowering** — no nolick / no painted target;
  the rectifier's even curvature tilts the field below the no-lick line emergently.
- **softplus**: task 3/4 (s1 failed); wells mostly <0 (mean −0.63, 4/6), κ₀ huge (±10–25).
- **tanh L1 pin**: w=0.1 → +0.29 (4/4 bistable, 0/8<0); **w=1.0 → −0.18** (4/4 bistable, 5/8<0). Strong
  pin lowers (mostly) WITHOUT degrading memory — earlier "1/4 degraded" was the classifier bug.
- Figures `rnn/sweep_{relu_cap,softplus_sp,nogo_pin}/`.

### Attention amplitude fails; relu landscape bad; lif+DC bifurcates the well down (2026-08-04)

New knobs: `attention_scale` (tasks.py `_attn_window`), `decision_readout_mean` (init.py: DC on n₁).

- **`sweep_wellpush`** (`--run_filter wp`): tanh × `attention_scale` 1/2/3 + lif_sc + relu deep-mem.
  **Attention amplitude does NOT lower** the wells on retraining (wells +0.07/+0.02/+0.17) — the
  trainable attention weight neutralizes the bias. **relu's low wells are a BAD landscape** (κ₀±3–6,
  2/4 bistable, spiral); `lif_sc` (bounded) → +0.30 (non-negativity alone insufficient); `reludeep`
  diverged. Clean-landscape and below-0 were mutually exclusive. Detail: `ring_lowerplane_log.md` §19.
- **`sweep_lifdc`** (`--run_filter lifdc`): **lif** (Gaussian CDF, φ(0)=½), gain 2, λ₀=3, sweep
  `decision_readout_mean ∈ {0,−0.3,−0.6,−1.0}`, clean base, no nolick, 4 seeds. **Landscape SOLVED**:
  compact bistable wells κ₀≈±1, no relu blowup/spiral, task-perfect (dual_dpa≈1, go/nogo=1). The DC
  works by **bifurcation**: each sample splits into up+down attractors; at ⟨n₁⟩ init −1.0 (trained
  −0.75) **all 4 seeds have 4 wells (2 up + 2 down), down wells at κ₁≈−0.58**. Creates no-lick-plane
  memory wells (closer); remaining = kill the up copies. Baseline-pin erodes ⟨n₁⟩. Detail: §20.
- **`wells.py` fixed**: was hard-coded to TWO wells (`_side_well`, one per κ₀ sign, averaged A&B) →
  collapsed the 4-well (2up/2down) structure and mis-averaged up+down into a meaningless number
  (the source of repeated wrong well-location claims). Now lists EVERY memory-well attractor with
  (κ₀,κ₁), up/down tally, and an "all-wells-down / seed" GOAL metric. Figures `rnn/sweep_{wellpush,lifdc}/`.

### DPA-metric bug fixed; DC×attention 2×2; the noise-margin mechanism (2026-08-05)

**★ DPA-accuracy bug:** `_dpa_accuracy*` regenerated DPA trials without `windowed_targets`/`decay` and read
the LAST-timestep target (0/NaN after decay) → `pair=nan`, `overall≈0.50` for **every** windowed run. DPA
was always solving (probe Δ(match−nonmatch)=0.89). Fixed via `_dpa_score()` (scores the supervised decision
window) → re-scored **dpa=1.0**. All prior "task-perfect" numbers reported this bug, not true DPA.

- **2×2 DC × attention** (lif, gain2, λ₀=3, one-sided, windowed, 250 DPA ep; all fixed-eval dpa≈1.0, go=0
  under one-sided): `sweep_os_ep` (DC+attn) wells **κ₁≈−0.67, 3/4 all-down** — best; `sweep_noattn` (DC only)
  **erratic, memory collapses to 1 well**, 1/4; `sweep_nodc` (attn only) clean 2-well but shallow **−0.09**,
  3/4; `sweep_nodc_noattn` (neither) straddles 0, 1/4. g·λ₁≈9–12 (supercritical) everywhere. **DC = deep
  lowering; attention = stabilizes the two-sided memory; both needed.** Detail §21.
- **`sweep_nodc_afrz`**: freeze-attention-in-GNG now DEFAULT (train attention in DPA, freeze GNG+Dual). Small
  deepening −0.09→−0.14 vs `sweep_nodc` (attention free), within seed noise.
- **`sweep_noise`** (`--run_filter nzA`, noise 0.5/0.75, DC=0, attn on, `stop_loss=0.05`): testing the
  noise-robustness-margin lever — the ONLY below-0 pressure (wells ≈−0.09 ≈ input-noise scale). Then `nzB`
  (go-preserving one-sided, new `rwd_keep_go_hinge`). *Running.*
- **Infra**: `stop_loss` 0.02→0.05 (0.02 overtrains Dual); `gng_criterion` kept two-sided (`_uw_gng`);
  STACKED 3-row flow portraits (`plot_stage_stacked_flow` + `_render_meanflow_stacked`). Figures
  `rnn/sweep_{os_ep,noattn,nodc,nodc_noattn,nodc_afrz}/`.

### Cue-driven response; the NOISE mean field; sim-trajectory tooling (2026-08-06/07)

**`sweep_cue`** — `r3cue` (rank-3) + `r2cue` (rank-2), 4 seeds each. Base = the tf-arm recipe (lif, gain2,
one-sided go-preserving `rwd_keep_go_hinge`, `decision_readout_mean=0`, noise 1.0, 250 DPA ep, stop 0.05);
rank-3 adds `gng_lambda=1.5` (bistable rule) + `decision_lambda=0.5` (subcritical κ₂). **New flag
`response_in_cue`** (all 3 generators + eval windows + RunConfig): score the go/pairing response in the LAST
0.5 s of its triggering stimulus (cue/test ON) not after it turns off; ran with `decay_to_zero=False` (purest
emergent). **RESULT: task-perfect, but cue-locking made the up-copies WORSE** — r3cue autonomous go-rule wells
κ₂≈+0.64…+0.77 (vs r3o10 +0.25…+0.44), **0/4 all-down**; r2cue 0/4. The held rule feeds an ADDITIVE lick
(κ₂≈rule+cue, not the conjunction rule∧cue), and dropping the post-cue pin removed the only downward pressure.
⇒ up-copies need an active downward force (one-sided κ₂ decay / attention-baseline), not timing alone. §22a.

**★ NOISE-corrected flows (all four sweeps: `r2go`, `r2cue`, `r3o`, `r3cue`, clean + `_noise` published).**
Production noise field = **input-only EXACT Gaussian resummation**, `low_rank_field_np(noise_sigma=σ)` =
`(1/N)Σ nⱼ φ(āⱼ/√(1+c·sⱼ²))`, sⱼ²=g²Aⱼ²σ²‖wⱼ‖², c=1 lif (effective-gain compression g→g/√(1+cs²)); validated
~2e-3 vs MC. σ_eff=noise·√(1−e^{−α})²≈0.37. Noise destabilizes MARGINAL wells (saddle-node) but is **not
directional** — clears go-rule up-copies in only **2/16 seeds** (r3cue s1 annihilates both; r2go10 s2 → 0 up),
and can kill the desired down-wells (r3o s0). `rank3_flow.py --noise` (lif added to its jax PHI), `plot_sweep
--field_input_noise` (16-draw MC of the same thing, ~16× slower), `scratchpad/wells3.py` clean-vs-σ. §22b–c.

**Self-consistent DMFT** (`solve_sc_variance` etc.) = ⚠ EXPERIMENTAL: right structure but over-predicts stiff
modes ~10–20× (two-timescale FDT factor omitted; `scratchpad/validate_sc.py`). Kept input-only exact as
production. **Dubreuil** (`~/models/dubreuil`) flow field is finite-N deterministic (no noise/self-consistent
code in the repo) — equivalent to our reduced field for pure low-rank. §22d–e.

**★ Genuine sim-trajectory tooling:** `src/dynamics.integrate_kappa_trajectories` (rank-general) + **`traj_flow.py`**
CLI (rank-2 stages×conditions / rank-3 conditions×3-planes / `--noise`) — integrate REAL trajectories from a
κ-grid. `plot_sweep --use_sim_field` is a one-step adiabatic map (≈β·analytic), NOT trajectories. Confirms the
reduced field is the true flow for pure low-rank nets. Companion figs `scratchpad/plot_cue_{targets,inputs}.py`.
§22f.

**Infra — flow-code refactor (branch `flow-refactor`).** `src/dynamics.py` (1884 lines) split into a
layered flow package (`dynamics.py` re-exports for back-compat): `flow_field.py` (shared rank-general
engine + noise + sim primitives), `flow_fixedpoints.py` (**shared rank-general `find_fixed_points`, scipy
+ brainpy backends**), `flow_rank2.py` / `flow_rank3.py` (analytic rendering by rank), `flow_traj.py`
(trajectory rendering, rank-general). `rank3_flow.py`/`traj_flow.py` → thin CLIs. All tools verified
identical output (rank3_flow 12 FPs unchanged). Tool-selection + code map in `docs/analysis.md`.
