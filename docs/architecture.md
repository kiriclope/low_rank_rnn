# Architecture

## Model (`src/models.py` — `LowRankModel`)

Discrete-time two-timescale RNN with rank-R recurrent factorisation.

```
W_rec = m @ n^T / N        (m, n ∈ ℝ^{N×R})
κ     = rates @ n / N      (shape B×T×R — internal readout)
```

No separate output layer (`output_size=0`). κ₁ (last rank) is the loss / accuracy
channel.

### Per-step update

```
h     ← exp(-α_rec)·h     + (1 - exp(-α_rec))·(W_rec · rates)
rates ← exp(-α)·rates     + (1 - exp(-α))·φ(gain·(Ai·Wi·x + h))
```

`gain` scales the **full** net input (feedforward + recurrent). The chaos threshold is
`gain × λ_max(W_rec) = 1`.

### Nonlinearities (`nonlinearity` param)

| Name | Formula | φ'(0) | Range | Ring-capable |
|---|---|---|---|---|
| `tanh` | tanh(x) | 1.0 | (−1, 1) | ✓ |
| `relu` | max(0,x) | 1.0 | [0, ∞) | ✗ (asymmetric) |
| `softplus` | log(1+eˣ) | 0.5 | (0, ∞) | ✗ (no saturation) |
| `erf` | erf(x) | 2/√π ≈ 1.13 | (−1, 1) | ✓ (faster sat. than tanh) |
| `elu` | x if x>0, eˣ−1 if x≤0 | 1.0 | (−1, ∞) | ✗ (positive side unsaturated) |
| `lif` | (1+erf(x/√2))/2 | 1/√(2π) ≈ 0.40 | (0, 1) | ✗ (asymmetric, low slope) |
| `lif_sc` | (1+erf(x·√π))/2 | 1.0 | (0, 1) | ✗ (asymmetric) |

**Ring-capable** requires: (1) odd function so both ±κ* are FPs, (2) saturation on both
sides so gain·φ'(κ*)·λ₀ < 1 at the ring radius. Only tanh and erf satisfy both.

### Reward feedback

`rwd_channel = −1` (last input dim). Teacher-forced: if target[...,−1]==1 and
κ₁[...,−1] > 0.5, a +1 pulse is added to the reward input on the next step.

### Key parameters

| Param | Meaning |
|---|---|
| `gain` | Scales full net input; NOT saved in state_dict |
| `alpha = dt/tau` | Rate timescale |
| `alpha_rec = dt/tau_rec` | Recurrent-input timescale |
| `noise` | Per-step recurrent noise std (prefactor) |
| `rank` | Number of recurrent modes (always 2 here) |

---

## EISTP model (`src/models.py` — `EISTPModel`, `model_type="eistp"`)

Minimal self-contained port of the **NeuroFlame dual-EI network** (`~/models/NeuroFlame`,
`conf/train_dual_EI.yml`) — the model that produces **persistent working memory** AND
**lower-plane decision wells** (see `docs/ring_lowerplane_log.md` §11). LowRankModel-compatible
interface (`forward`, `get_readout`, `update_dynamics`, `.m`, `.n`, `.wi`, `.gain`, `.noise`,
`exp_alpha`, `exp_alpha_rec`) so the sweep/plot/flow tooling works unchanged.

**Mechanism (all essential):**
- 2-pop **EI**: N = `n_neuron`, E = round(0.75·N), I = 0.25·N. **Sparse binary** connectivity `C`
  (entry 1 w.p. `K/N_pre`, mean ⟨C⟩=K/N, in-degree ~K). Dale block strengths `Jab=[1,−1.5,1,−1]`
  balanced **1/√K**. `relu` rates. (Runs use N=1000/K=125, K scaled with N to hold K/N=0.125.)
- **Two timescales**: synaptic filter (`tau_syn`, → `exp_alpha_rec`) on the recurrent current,
  then rate filter (`tau`, → `exp_alpha`) on `relu(ff + syn)`. (Per-population E/I time constants.)
- **Markram STP on E→E** (`u,x` per presynaptic-E unit; USE=`stp_use`, τ_fac, τ_rec): output
  `u·x·r`; gate sweeps from USE (rest) to ~1 (full facilitation). Differentiable (not detached).
- **Trained rank-2 low-rank `m,n` on E modulates the STP E→E weight MULTIPLICATIVELY**:
  ```
  W_EE = gain·j_stp·(C/√K)·(1 + n@mᵀ / lr_scale),   clamped ≥0 (Dale)
  κ    = rates_E @ n / N_E      (n = output/readout dir, m = presynaptic selection)
  ```
  The memory mode *rides on* the facilitating synapses — NOT an additive backbone perturbation.
- **Inputs: LINEAR** (NeuroFlame `dualStim`, not cosine). `wi`: fixed random E-pattern per channel
  ("odors"), E-only, with balanced scaling `external = gain·√K·M0·(Ja0 + Wi·code)`; Ja0 baseline to
  E and I. `forward` takes the vanilla low-dim code `(B,T,input_size)`.

**The decisive param — `lr_scale`** (NeuroFlame `train_scale`): the memory-mode gain is
`g_mem = √K·⟨mn⟩/lr_scale` (the K/N density of C cancels the N of the overlap → √K). `"N"` (=N_E)
gives g_mem≈0.015 (DEAD, DPA at chance); **`"sqrtK"`** (=√K) gives g_mem≈lr_ini²≈1 (CRITICAL) and
the memory persists. **Always use `eistp_lr_scale="sqrtK"`.**

| Param | Meaning |
|---|---|
| `n_neuron` / `eistp_K` | total units / mean in-degree K (keep K/N≈0.125) |
| `j_stp` | E→E STP weight scale (fixed 1.0; trainable in NeuroFlame but frozen) |
| `eistp_lr_scale` | low-rank divisor: `"N"` (dead) or `"sqrtK"` (critical — use this) |
| `stp_U` / `stp_tau_f` / `stp_tau_d` | Markram USE / τ_fac / τ_rec |
| `low_rank_scale` | `lr_ini` (init scale of m,n; =1 → mode starts critical) |

Stability note: the `/√K` coupling + STP can run away (~2/5 seeds NaN); mitigate with lower lr /
tighter grad-clip / `lr_ini`<1. Flows: use `ei_flow.py` (it has a dedicated eistp path), not
`plot_sweep --plots flow` (analytic reduction is invalid here).

---

## Tasks (`src/tasks.py`)

All generators return `(inputs, targets[, trial_type, condition_names])` with shape
`(n_trials, n_steps, *)`.

### Input channel layout (input_size=8, or 7 with cue_on_go_input=True)

| Channel | Stimulus |
|---|---|
| 0 | A sample |
| 1 | B sample |
| 2 | C test |
| 3 | D test |
| 4 | Go stimulus (also carries GNG cue if `cue_on_go_input=True`) |
| 5 | NoGo stimulus |
| 6 | GNG cue (removed if `cue_on_go_input=True`) |
| 7 (or 6) | Reward (always last = `input_size − 1`) |

### Target encoding (target_rank=2)

- **Channel 0 (memory κ₀):** supervised to ±1 over the delay in DPA. In Dual: left `nan`
  (unsupervised) except a pre-sample 0.
- **Channel −1 (decision κ₁):** time-multiplexed. DPA decision (±1) after test; GNG
  response (go=+1, nogo=`nogo_target`) in the cue window. `nan` masks unused timesteps.

### Timing

`TaskTiming(stim_on, stim_off, t_steps, dt)`. Dual uses 4 epochs:
`[sample, gng, cue, test]` at times `[2, 4, 6, 8]` s (on/off).

---

## Training pipeline (`sweep.py`, `src/train.py`)

Three sequential stages with selective freezing:

### Stage 1: DPA
- Train all parameters.
- Checkpoint: `dpa_{run_id}.pth`

### Stage 2: GNG (naive)
- **Freeze:** rank-0 of m/n (`freeze_low_rank_cols=[0]`) + DPA+reward input dims
  `[0,1,2,3,input_size−1]`.
- If `freeze_input_stages` includes `"gng"`: all input dims frozen.
- Checkpoint: `naive_{run_id}.pth`

### Stage 3: Dual (expert)
- **Freeze:** by default all input dims (`list(range(input_size))`).
- If `freeze_rank0_dual=True`: rank-0 of m/n also frozen.
- Checkpoint: `expert_{run_id}.pth`

### Freezing mechanism

Snapshot frozen params before optimizer step → zero their grads after `backward()` →
**restore** original values after `optimizer.step()`. The restore makes freezing exact
even with AdamW weight decay.

### Loss functions

| Loss | Used for | Notes |
|---|---|---|
| `MaskedMultiTargetLoss` | DPA, GNG | Per-channel masked MSE |
| `MaskedMultiTargetDualLoss` | Dual (default) | Splits decision channel by time window into DPA/GNG/baseline components; each independently weightable |
| `MaskedGNGLoss` | GNG variant | Nogo-zero hinge + go_hinge_thresh |

`dual_loss="separated"` selects `MaskedMultiTargetDualLoss`. Loss components logged in
`.last_components`.

### Early stopping

`stop_loss` (default 0.1 in current sweeps): halts stage when both train and val loss
drop below threshold.

### κ₁ regularizer

`kappa1_reg_weight > 0` adds `weight × relu(gain·n₁ᵀm₁/N − 1)²` to the Dual loss,
penalising λ₁ going super-critical.

---

## Initialisation (`src/init.py`)

`init_dpa_internal_readout_prepost` — structured init for DPA:

- Rank-0 (memory): eigenvalue `memory_lambda`, corr(m₀,n₀) = `target_mn_corr`.
  A/B inputs aligned with ±u_mem; C/D inputs aligned with ±u_test.
- Rank-1 (decision): eigenvalue `decision_lambda`.
  Decision direction is `mix_strength·u_mix + sqrt(1−mix_strength²)·u_noise`
  where `u_mix = u_mem ⊙ u_test`. Default `mix_strength=0` → random decision direction.

**Structured init** mainly scaffolds the **memory rank**, not the decision rank.

With `memory_lambda=0.8` and `gain=1.5`: effective λ₀ = 1.5×0.8 = 1.2 > 1 (ring
bifurcation met); with `decision_lambda=0.5`: effective λ₁ = 1.5×0.5 = 0.75 < 1
(κ₁ sub-critical → input-driven, no competing autonomous attractors).
