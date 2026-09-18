# Nonlinearity Investigation

## Goal

Find a nonlinearity that gives:
1. A **ring attractor** in κ₀ after DPA (symmetric ±κ* FPs for A/B memory)
2. **Input-driven attractors** in κ₁ after Dual (crisp go/nogo readout)
3. The ring **persists** after Dual

## Ring formation conditions

The ring requires:
- **Odd nonlinearity**: φ(−x) = −φ(x), so both +κ* and −κ* satisfy the FP equation
- **Saturation**: gain × φ'(κ*) × λ₀ < 1 at the ring radius (stabilises the FP)
- **Super-criticality**: gain × φ'(0) × λ₀ > 1 (bifurcation from origin)

Only **tanh** and **erf** satisfy both conditions naturally. All others have at least one
failure mode.

## Nonlinearity comparison

### tanh
- Ring ✓ (odd, saturates to ±1)
- Problem: with high gain, both λ₀ and λ₁ go super-critical after Dual → isolated
  attractors replace the ring; κ₁ develops competing autonomous FPs.
- Fix: `freeze_rank0_dual=True` + `kappa1_reg_weight > 0` to prevent λ₁ > 1.

### relu
- No ring ✗: positive side is linear (no saturation → no stable FP above 0); negative
  side outputs 0 (B-sample state collapses, κ₀ = 0 not −κ*).
- κ₁ attractors are crisp and large (linear growth above threshold).
- Even with structured init: ring impossible (asymmetry is intrinsic to relu).

### softplus
- No ring ✗: everywhere positive and monotonically increasing with no upper saturation.

### erf (Brunel approximation)
- Ring ✓ (odd, saturates to ±1 faster than tanh — Gaussian vs sech² tail)
- Same fix requirements as tanh.
- Saturation profile: falls off as exp(−x²) → FP more strongly stabilised than tanh.

### ELU
- Tested as a hybrid: negative side saturates (to −1), positive side is linear (relu-like).
- Hypothesis: negative saturation → stable −κ* attractor; linear positive → crisp go.
- Result ✗: positive branch still unsaturated → positive-κ* FP not stabilised → ring
  fails on both sides in practice.

### LIF (Brunel erfc approximation)
- φ(x) = (1 + erf(x/√2)) / 2 — Gaussian CDF, range [0, 1], φ'(0) = 1/√(2π) ≈ 0.40
- One-sided (non-negative) → B-sample (negative κ₀) collapses → no ring ✗
- Same fundamental limitation as relu.
- gain=3 needed for ring bifurcation → vanishing gradients, dual stage fails completely.

### LIF with gain=2 (sweep_lif2)
- Lower gain → less saturation → some learning.
- Dual still fails (loss flat ~0.80–1.07 after 200 epochs, no convergence).
- DPA retention poor (after_gng dpa ≈ 0.48–0.77).

### LIF rescaled / lif_sc
- φ(x) = (1 + erf(x·√π)) / 2 — φ'(0) = 1 (same as tanh at origin), range [0, 1]
- One-sided → no ring ✗ (same asymmetry issue)
- GNG converges faster/better than original LIF.
- Dual still fails (loss ~0.81–1.05 at ep200, not converging).
- Root cause: one-sided range [0,1] can't represent negative DPA targets (−1) directly.

### Structured init + tanh + gain=1.5 (sweep_tanh_struct)
- Separates ring (λ₀_eff = 1.5×0.8 = 1.2 > 1) from decision (λ₁_eff = 1.5×0.5 = 0.75 < 1)
- No freeze needed — structure enforces separation.
- Ongoing: see [experiment log](experiment_log.md).

## Summary table

| Sweep | Nonlinearity | Ring after DPA | DPA retained after GNG | Dual converges |
|---|---|---|---|---|
| sweep_relu | relu | ✗ | Poor | ✓ (GNG only) |
| sweep_softplus | softplus | ✗ | Poor | partial |
| sweep_tanh_reg | tanh + reg | ✓ (frozen) | TBD | TBD |
| sweep_erf | erf | ✓ | TBD | TBD |
| sweep_elu | ELU | ✗ | Poor | Poor |
| sweep_lif | LIF gain=3 | ✗ | Poor | ✗ (vanish. grad) |
| sweep_lif2 | LIF gain=2 | ✗ | Poor | ✗ (stuck ~0.8) |
| sweep_lif_sc | LIF_sc gain=2 | ✗ | Poor | ✗ (stuck ~0.8) |
| sweep_relu_struct | relu + struct | ✗ | TBD | TBD |
| sweep_tanh_struct | tanh + struct + gain=1.5 | TBD | TBD | TBD |

## Conclusion so far

**⚠ SUPERSEDED (2026-09-08).** The paragraph below was wrong: sign in κ = rates·n/N comes from n,
not from the rates, and every foundation seed (`sweep_r2nocue`, lif, gain 1) holds a symmetric
pair of memory wells at κ₀ ≈ ±1.2 (`ring_lowerplane_log` §27b). What lif cannot do is the RING —
the current target is the two-well geometry, which it does. Kept for the record:

> The one-sided LIF nonlinearity is fundamentally incompatible with the ring attractor
> required for DPA. The B-sample cannot be encoded as a negative-κ₀ state because LIF
> output is always ≥ 0, so the recurrent drive at κ₀ < 0 is suppressed.

The most promising approaches remain:
1. **tanh/erf with freeze_rank0_dual + kappa1_reg**: explicitly protects the ring.
2. **tanh with structured init + low gain**: structure enforces λ₀ > 1 > λ₁ without
   needing a regularizer.

## 2026-09-08/09 — the foundation's φ, and relu as a control (see `ring_lowerplane_log` §28)

**What `lif` is.** Φ(x) = ½(1+erf(x/√2)), the Gaussian CDF: the exact mean-field rate of
binary/threshold units under Gaussian input (van Vreeswijk & Sompolinsky), the high-noise erfc limit
of the LIF Siegert rate (Amit & Brunel 1997; Brunel 2000), the probit sigmoid. Range (0,1), REST
RATE 0.5, slope 0.40 at rest (so the "gain·λ = 1" chaos line in `architecture.md` is off by 0.4 for
lif), φ′→0 at both ends (the gain-steal mechanism behind the cue's state-dependent push). All eight
torch φ agree with the numpy mirrors in `src/flow_field.py` to machine precision, φ′ included.

**relu at the foundation's parameters (`sweep_r2fdrelu`) has no memory well.** Relu is positively
homogeneous, so on the memory axis the active set depends only on sign(m₀ᵢκ₀) and the autonomous
field is linear on each half-line, F₀ = (λ⁺ − 1)·κ₀ with λ⁺ = g·Σ_{m₀ᵢ>0} n₀ᵢm₀ᵢ/N (measured
1.25–1.37 = the asymptotic slope to 3 decimals). One fixed point (the origin): the memory is an
exponential escape (κ₀ → 30–40 by test-on, rates to 330) that the one-sided hinge never prices.
Retention breaks in GNG (0.58–0.69) because retraining m₁/n₁ shifts the active set and hence the
growth rate the amplitude-dependent readout was calibrated on. Activity L2 (`rate_reg_weight`)
bounds it (w=0.01: slower escape; w=0.1: task dies below a repeller) but cannot create a well;
trainable unit biases did not either, because no term asks for λ⁺ < 1 at the target. A relu well
needs λ_full > 1 > λ⁺ (the threshold-linear bump mechanism) — via a two-sided memory pin, a
half-population gain constraint, or EI inhibition (what the EISTP model's depression supplies).
Figure: `results/figures/sweep_r2rr01/field_profiles_relu_vs_lif.png`.


## 2026-09-17/18 — CORRECTION: the ring is a covariance property, not a transfer-function property

The "ring-capable: odd + saturating (tanh, erf only)" verdicts above were empirical, from structured
inits. Mean-field: F(κ) = −κ + Σ·⟨φ′(κᵀΣ_m κ)⟩·κ — φ enters only through the averaged gain; the ring
exists iff the rank-2 covariance is isotropic (equal σ_m and σ_n across modes, zero cross-covariances,
zero means). Measured (`ring_lowerplane_log` §33): lif rings at init once λ > λ_c = 1/(g·φ′(0)) = 2.5
when both modes are built alike (`scratchpad/init_flow_grid.py`, grids 1–3); a Gaussian surrogate with
the trained net's covariance reproduces its anisotropy and an isotropised one rings. What breaks it in
our nets is `init.py`'s decision construction (σ(n₁) = 1 vs σ(n₀) = √(λ/ρ)) and training, which
equalises the overlaps J but not the factors σ_m, σ_n. Unequal σ_n at equal J → ellipse → four cardinal
wells. The rows above remain valid as descriptions of what those (structured-init) nets did.
