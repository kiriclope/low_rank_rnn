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

**What is NOT supervised in Dual** (matters when reading results — `ring_lowerplane_log` §25d):
κ₀ carries only the pre-sample 0, so the sample memory survives only via the pairing readout;
the go/nogo rule has no pre-cue hold unless `dual_gng_memory=True`. With `freeze_rank0_dual=False`
Dual can REBUILD a memory that GNG destroyed, so `dual_dpa`≈1.0 does **not** imply retention —
the project's key metric is `after_gng/dpa`, measured before that repair.

### Delay-window options (2026-08-12)

- `nolick_late_delay` — restrict the nolick term to the after-cue span (Dual `(cue-off, test-on)`,
  GNG `(cue-off, end)`), free steps only. The NeuroFlame delay-supervision mechanism: on
  'none'/pure-DPA trials the state sits ON the sample well there, so the term grades the well's κ₁.
- `gng_decay_to_zero` — GNG stage pins BOTH trial types back to 0 from response-end to trial end
  (`decay_to_end`), so nothing can park during the GNG epochs for Dual to inherit.
- `nogo_target=None` — drop the nogo response target entirely (redundant with the nolick hinge).
  ⚠ leaves the cue window unsupervised for nogo; `nogo_target=0.0` + `rwd_nogo_onesided` gives a
  one-sided hinge there instead. Also shifts the eval boundary
  (`(th_go + max(nogo_target, nogo_hinge_thresh))/2`) — compare arms with `flow_verdict.py`, which
  always scores at the fixed boundary 0.

Added 2026-09-02 (foundation-era; see `ring_lowerplane_log` §27):

- `gng_hold_full_delay` — the windowed go/nogo hold spans stim-off → cue-on (GNG; go/nogo-off →
  cue-on in Dual) instead of the 0.25 s pre-cue hold — symmetric with the A/B supervision. Used by
  the `nocue` foundation arm.
- `cue_scale=0.0` — no cue input at all, with `cue_on_go_input` keeping input_size/timings
  identical to cue versions (cue sweeps become one-scalar deltas).
- `nolick_nogo_in_cue` (2026-09-07) — the NOGO rows' don't-lick span starts at CUE ONSET (Dual:
  cue-on → test-on; GNG: cue-on → end) instead of cue-off — the cue is the lick window and a nogo
  trial must not sit in κ₁>0 during it. Rows are classified from their own hold target inside the
  gng span (negative = nogo, positive = go, none = DPA row), so it needs `dual_gng_memory=True`
  (guarded) and `nolick_weight>0`. Combine with `nolick_full_delay` (DPA rows) and, if wanted,
  `nolick_late_delay` (go rows after the cue — measured NEGATIVE, §27i). The GNG-stage nolick weight
  is gated on `nolick_late_delay OR nolick_nogo_in_cue`. Loss component: `nolick`.
- `gng_hold_ceiling` (2026-09-07, default None, UNUSED) — one-sided hinge from ABOVE on the go hold
  (`hinge(κ₁ − ceiling)` on pre-cue go steps; component `gng_ceil`). Needs `rwd_go_thresh` above it.
  Implemented for the premature-lick fork, then judged the wrong question (§27g/h).
- `rate_reg_weight` (2026-09-09) — ACTIVITY L2: `w·⟨rates²⟩` over all units and steps, added to
  the train AND val objective at every stage (`Optimization(rate_reg=w)`; the epoch line prints
  `⟨r²⟩`). Needed for a non-saturating φ (relu), whose memory field is linear on each half-line so
  nothing else prices a runaway; it bounds activity but cannot create a well (§28c).
- `max_val_loss` (2026-09-09, default 100) — the trainer's abort threshold on the val loss, now a
  RunConfig field. Relu + activity L2 needs ~1e6: at the unstable init ⟨r²⟩≈1e4, so the penalty
  alone exceeds 100 before the first step.
- `dpa_hold_window` (seconds, default 0.0 = legacy) — restrict the **DPA-stage A/B memory
  supervision** to the last N s ENDING at test onset, instead of the legacy span sample ONSET →
  test onset (6 s, the sample included). Mirrors the GNG identity hold, which is a 0.25 s window
  ending at cue onset (`generate_gng_trials`, `hold_full_delay=False`). DPA stage ONLY — the Dual
  generator sets no κ₀ target at all, so `mem_pos`/`mem_neg`/`mem_decay` are identically 0 there.
  Rationale: the legacy span demands |κ₀| ≥ θ *during the sample*, pricing the RISE TIME as well as
  the amplitude; a terminal window prices only "be there when it is read". Measured (§28d): Φ
  absorbs it (wells intact, dpa 0.99–1.00) and κ₀ then rises into the well across the delay instead
  of being clamped from the sample on; relu is unaffected (λ⁺ stays > 1). Set 0.5 in the `w5*`,
  `sc*`, `scl*` arms.
- `nolick_full_delay` — extend the Dual don't-lick to the WHOLE delay on the DPA trials only
  (rows identified by no finite decision target in the go/nogo span; guarded: needs
  `dual_gng_memory=True` and `target_rank=2`). go/nogo trials keep the late window.
- `dpa_nolick_weight` — the same one-sided hinge over the DPA-stage delay (guarded: needs
  `dpa_prelick_free=True`, else the legacy pin leaves no free steps and the term is silently inert).
- `nolick_thresh` — displace every nolick hinge to κ₁ ≤ −ε (relu/relu² stop pushing at the
  boundary, so ε is the only imposed-depth lever).
- `gng_decouple_decision` — GNG stage projects `n[:,dec] ⟂ m[:,0]` after each optimizer step
  (kills the κ₀→κ₁ readout leak). ⚠ §27a: the κ₁→κ₀ drive needs the second projection
  `m[:,dec] ⟂ n[:,0]` — not yet implemented. None of the nolick/decoupling levers has been run.

### Freezing mechanism

Snapshot frozen params before optimizer step → zero their grads after `backward()` →
**restore** original values after `optimizer.step()`. The restore makes freezing exact
even with AdamW weight decay.

### Loss functions

**`UnifiedLoss` is the only loss `sweep.py` runs** (2026-08-12). The legacy paths
(`MaskedMultiTargetLoss` "multi", `MaskedMultiTargetDualLoss` "separated", `ThresholdLoss`
"threshold", and `MaskedGNGLoss` for the GNG stage) were removed from the runner — `run_one`
raises on `dual_loss != "unified"`. The classes remain in `src/train.py` for older scripts;
`hinge_squared` is now dead config. Components logged in `.last_components`.

One loss at all three stages; the task semantics live entirely in the TARGETS (`src/tasks.py`):

| target value | term |
|---|---|
| +1 | one-sided hinge `h(θ⁺ − κ)` — free above |
| −1 | one-sided hinge `h(κ + θ⁻)` — free below |
| 0 | pin `q(κ)` |
| NaN | free — *except* inside `nolick_window`, where free steps get `h(κ₁)` under `nolick_weight` |

`hinge_shape` picks `h`, i.e. **how force scales with the size of a violation** — and the pin
norm `q` follows it, so the loss uses one norm throughout:

| `hinge_shape` | `h(x)` / `q(p)` | force near the target | use |
|---|---|---|---|
| `relu2` (default) | `relu(x)²` / `p²` | `2x` → vanishes; 10× weaker than relu at x=0.05, parity at 0.5 | soft margin; tolerates straddling |
| `relu` | `relu(x)` / `\|p\|` | constant 1 up to the threshold | boundary constraints ("don't lick"); crisp satisfaction |
| `softplus` | `softplus(x)` / `p²` | `σ(x)`, never 0 | the only shape that rewards *depth* beyond the target — also inflates every amplitude; positive loss floor ⇒ `stop_loss` disabled |

`relu2` and `relu` share an equilibrium (zero gradient once satisfied), so **neither can place a
state strictly beyond a threshold — displace the threshold instead of changing the shape.**

Thresholds: `go_hinge_thresh` (θ⁺) and `nogo_hinge_thresh` (θ⁻) on the go/nogo terms,
`dpa_hinge_thresh` on pairing + memory. Groups: baseline `bl`, `gng_*` (pre-`pair_start`),
`rwd_go`/`rwd_nogo` (response window, carved out), `pair_*` (post-`pair_start`), `mem_*`
(non-decision channels), `nolick`.

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
