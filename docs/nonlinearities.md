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

The one-sided LIF nonlinearity is fundamentally incompatible with the ring attractor
required for DPA. The B-sample cannot be encoded as a negative-κ₀ state because LIF
output is always ≥ 0, so the recurrent drive at κ₀ < 0 is suppressed.

The most promising approaches remain:
1. **tanh/erf with freeze_rank0_dual + kappa1_reg**: explicitly protects the ring.
2. **tanh with structured init + low gain**: structure enforces λ₀ > 1 > λ₁ without
   needing a regularizer.
