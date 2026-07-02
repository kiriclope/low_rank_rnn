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
