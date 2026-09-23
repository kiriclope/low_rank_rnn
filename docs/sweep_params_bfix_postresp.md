# Parameters and arguments of the two last sweeps (2026-09-22)

Generated from the `config.json` on disk of seed 0 of each arm (all seeds of an arm share the config apart from `seed`/`run_id`).
Reference = the free fixed-code recipe `sweep_lif_recipe7_bfix` (8 seeds). `sweep_lif_symdpa_bfix`: 2 arms × 4 seeds; `sweep_lif_postresp`: 5 arms × 4 seeds.

## Launch commands

```bash
python sweep.py --out_dir results/dual/sweep_lif_symdpa_bfix --nonlinearity lif --run_filter symdpa_pair  --n_gpus 2 --per_run_screen
python sweep.py --out_dir results/dual/sweep_lif_symdpa_bfix --nonlinearity lif --run_filter symdpa_klein --n_gpus 2 --per_run_screen
python sweep.py --out_dir results/dual/sweep_lif_postresp    --nonlinearity lif --run_filter postresp_<pair|inv|test|klein|free> --n_gpus 2 --per_run_screen   # three waves of ≤ 8 runs
# then: plot_sweep.py --sweep_dir <sweep> --out_root results/figures; scratchpad/publish_gallery.sh <sweep>
```

Each run = one screen `sweep_<run_id>` teeing to `<run_dir>/train.log`; seeds 0–3 (bfix free: 0–7); stages DPA → GNG → Dual with the checkpoints `dpa_*.pth`, `naive_*.pth` (after GNG), `expert_*.pth` (after Dual).

## What differs between the arms

| field | free (recipe7_bfix) | σ₁ (symdpa_bfix pair) | V (symdpa_bfix klein) | postresp free | postresp σ₁ | postresp σ₂ | postresp σ₃ | postresp V |
|---|---|---|---|---|---|---|---|---|
| `symmetry` |  | pair | klein |  | pair | inv | test | klein |
| `symmetry_stages` | dpa, gng, dual | dpa | dpa | [] | dpa | dpa | dpa | dpa |
| `response_in_cue` | True | True | True | False | False | False | False | False |
| `dpa_post_response_window` | — | — | — | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |

`symmetry_stages` only matters when `symmetry` is set (the free arms carry the default or an empty list). Everything else is identical across the 28 runs and listed below. Two changes are code, not config: the symmetry tie covers `wi.bias` (block-averaged with the tie, so the DPA checkpoint is exactly symmetric) and the Dual-stage input freeze freezes `wi.bias` (both fixed 2026-09-21; verified on the checkpoints 2026-09-22: DPA bias asymmetry 0, bias unchanged from the GNG to the Dual checkpoint).

## All fields (shared value; RunConfig order, with the inline comment from sweep.py)

| section | field | value | meaning (sweep.py comment) |
|---|---|---|---|
| Network architecture | `hidden_size` | 1024 |  |
| Network architecture | `rank` | 2 |  |
| Network architecture | `gain` | 1.0 |  |
| Network architecture | `input_size` | 6 | post-__post_init__: 6 = sample A/B, test C/D, Go (+cue), NoGo |
| Network architecture | `target_rank` | 2 |  |
| Dynamics | `tau` | 0.2 |  |
| Dynamics | `dt_base` | 0.02 | dt = dt_base * tau_rec_frac |
| Dynamics | `tau_rec_frac` | 0.75 | scales both dt and tau_rec = tau * tau_rec_frac |
| Dynamics | `noise` | 1.0 | input noise prefactor; sigma = noise * sqrt(1 - exp(-alpha)^2) |
| Dynamics | `model_noise` | 0.0 | recurrent noise prefactor (same sigma formula) |
| Nonlinearity | `nonlinearity` | lif | "tanh" / "relu" / "softplus" / "erf" / "elu" / "lif" / "lif_sc" / "tanh_asym" |
| Nonlinearity | `nl_gamma` | 0.0 | asymmetry strength for "tanh_asym" (φ = tanh + γ·tanh²) |
| Model type | `model_type` | lowrank | "lowrank" / "ei"  (EI = Dale backbone + low-rank E→E) |
| Model type | `n_inh` | 128 | inhibitory units (EI only; E units = hidden_size) |
| Model type | `static_radius` | 1.5 | spectral radius of frozen Dale backbone (EI only) |
| Model type | `low_rank_scale` | 0.3 | init scale of trained low-rank E→E (EI only) |
| Model type | `low_rank_full` | False | EI: train low-rank on whole N×N graph (E↔I) vs E→E block |
| Model type | `use_stp` | False | EI: short-term plasticity (Tsodyks–Markram) on E presynapse |
| Model type | `stp_tau_f` | 1.5 | STP facilitation time constant (s) |
| Model type | `stp_tau_d` | 0.3 | STP depression time constant (s) |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `n_neuron` | 2000 | total units (E = frac·N) |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `j_stp` | 1.0 | E→E STP weight scale |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `eistp_lr_scale` | N | low-rank divisor (n@mᵀ)/lr_scale: "N" (=N_E) or "sqrtK" |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `eistp_r_max` | — | rate cap (anti-runaway); None = uncapped relu |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `eistp_lr_ueqv` | True | True: m init = n (critical g_mem); False: independent random init |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `eistp_lr_additive` | False | E→E low-rank: False=C·(1+lr) multiplicative; True=C+lr additive/dense |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `eistp_dense_cee` | False | E→E backbone C_EE: False=sparse binary/√K; True=dense ones/N_E |
| EISTPModel (model_type="eistp") — NeuroFlame dual-EI port | `eistp_init_noise` | 1.0 | init recurrent kick rates₀=relu(ff₀+init_noise·randn); 0 = deterministic/frozen |
| Initialisation | `init_style` | structured | "structured" / "random" |
| Initialisation | `memory_lambda` | 7.0 | κ0 sample-memory init eigenvalue |
| Initialisation | `gng_lambda` | 0.8 | κ1 gng-memory init eigenvalue (rank-3 only) |
| Initialisation | `decision_lambda` | 7.0 | κ1 (rank-2) / κ2 (rank-3) action-mode init eigenvalue |
| Initialisation | `target_mn_corr` | 1.0 |  |
| Initialisation | `target_out_mn_corr` | 1.0 |  |
| Initialisation | `readout_scale` | 2.6457513110645907 | σ(n₁) of the decision mode (init.py default 1.0). ISOTROPY: the memory mode has σ(m₀)=σ(n₀)=√(λ₀/ρ) while the decision mode has σ(n₁)=readout_scale and σ(m₁)=λ₁/(ρ·readout_scale) — set readout_scale=√(λ₁/ρ) to make the two modes exchangeable (§33: the ring needs equal factors, not just equal overlaps) |
| Initialisation | `mirror_tying` | False | A↔B reflection symmetry: build the init as two mirrored halves (memory mode sign-flipped, decision mode copied, A↔B / C↔D inputs swapped) and RE-IMPOSE it after every optimiser step. The two memory attractors are then forced to share a κ₁ — they move down together or not at all (§35) |
| Initialisation | `symmetry` |  **(differs, see above)** | task symmetry to tie: "" / "pair" (σ₁ = A↔B & C↔D, κ↦(−κ₀,+κ₁)) / "test" (σ₃ = C↔D, κ↦(+κ₀,−κ₁)) / "klein" (the full group). §36 |
| Initialisation | `symmetry_stages` | dpa, gng, dual **(differs, see above)** | stages where the symmetry is enforced; release it later and the next stage is free to break it |
| Initialisation | `sample_scale` | 1.0 |  |
| Initialisation | `test_scale` | 1.0 |  |
| Initialisation | `mix_strength` | 0.0 |  |
| Initialisation | `decision_readout_mean` | 0.0 | DC mean ⟨n₁⟩ of the decision readout (structured init). With a non-negative saturating φ (lif), resting κ₁=½⟨n₁⟩ → negative value seats the memory wells below the no-lick line (the clean saturating well-push). §20. |
| Initialisation | `rwd_input_scale` | 1.0 | scale of reward input alignment with u_read (structured init only) |
| Initialisation | `rwd_align_weight` | 0.0 | weight of reward-input ↔ n1 cosine alignment loss during DPA |
| Initialisation | `freeze_rank0_dual` | False | also freeze rank-0 of m/n during the Dual stage |
| Initialisation | `rule_timing` | gng | timing of the rule stage: "gng" (stim 2-3, cue 4-4.5, 6 s) or "2afc" (stim 2-3, cue 6-7, 8 s; §39) |
| Initialisation | `afc_response_to_end` | False | 2AFC (§39): score the response from cue-off to trial end (Leon's 2AFC), not the 0.5 s window |
| Initialisation | `freeze_rank0_gng` | True | freeze rank-0 of m/n during GNG (the curriculum default). False = a stand-alone rule task from scratch with both modes free (rulesym validation, §38) |
| Initialisation | `project_go_on_n1` | False | project go input column onto n₁ direction before GNG |
| Initialisation | `project_gng_orth_n0` | False | project go+nogo input columns orthogonal to n₀ before GNG |
| Initialisation | `use_fixed_weights` | False | add frozen random W_fixed to recurrent dynamics |
| Initialisation | `fixed_weight_scale` | 0.8 | g/sqrt(N) scale of W_fixed; use g>>1 for strong backbone |
| Initialisation | `fixed_weight_orthogonalize` | True | project W_fixed ⊥ m,n (False = backbone shapes κ-plane) |
| Initialisation | `fixed_weight_sparsity` | 1.0 | keep-prob p of W_fixed entries (1.0 = dense; rescales 1/√p) |
| Initialisation | `use_unit_bias` | False | per-unit bias inside φ; breaks κ-field odd symmetry |
| Initialisation | `unit_bias_trainable` | True | train the unit bias (False = frozen random) |
| Initialisation | `unit_bias_scale` | 0.2 | init scale of the random per-unit bias |
| Initialisation | `use_rec_scale` | False | trainable per-mode recurrent scale (decouples recurrence from readout) |
| Initialisation | `rwd_gng` | False | teacher-forced reward during GNG stage (False = disable) |
| Training (shared across all stages) | `learning_rate` | 0.01 |  |
| Training (shared across all stages) | `weight_decay` | 0.01 |  |
| Training (shared across all stages) | `batch_size` | 64 |  |
| Training (shared across all stages) | `grad_clip_norm` | — | None = disabled |
| Training (shared across all stages) | `n_batch` | 516 | trials per generated dataset |
| Training (shared across all stages) | `integrate` | both | which variables carry a time constant: "both" (legacy two-filter cascade: rec_inputs at tau_rec THEN rates at tau — the NeuroFlame / Wang-2002 lineage), "rates" (single filter on the rates, recurrent current instantaneous: tau r' = -r + phi(g(I+Wr))), or "rec" (single filter on the current, rates instantaneous: tau_rec x' = -x + W phi(g(I+x)) — the standard current-based rate RNN of Mante 2013, Song/Yang/Wang, Yang 2019, Mastrogiuseppe & Ostojic 2018, Dubreuil 2022). NOTE alpha_rec = dt_base/tau independent of tau_rec_frac, so the synaptic filter cannot be removed by tuning tau_rec_frac — hence this flag. |
| Training (shared across all stages) | `stop_loss` | 0.1 | early-stop threshold, ALL stages (Leon 2026-09-14: "just use 0.1 as a stop loss at each stage"). Fires when train AND val are both below it (src/train.py:358). Measured effect at the current epoch counts: DPA and Dual are unaffected (Dual ends at val 0.14-0.18 at 300 epochs, still falling), GNG stops at ~epoch 35 of 100 (train 0.092 / val 0.094 there, vs 0.037 at 100). ⚠ `after_gng/dpa` is measured right after GNG, so it now reflects a THIRD of the GNG training and is NOT comparable with pre-2026-09-14 retention numbers (foundation 0.996-1.000, this thread 0.670-0.999) which all used 100 GNG epochs. |
| Per-stage epoch budgets | `epochs_dpa` | 250 |  |
| Per-stage epoch budgets | `epochs_gng` | 100 |  |
| Per-stage epoch budgets | `epochs_dual` | 150 |  |
| saved as the "naive" checkpoint (replacing GNG's). | `dual_paired_stage` | False |  |
| saved as the "naive" checkpoint (replacing GNG's). | `epochs_dual_paired` | 100 |  |
| Task variant | `cue_on_go_input` | True |  |
| Task variant | `go_on_rwd_input` | False | route go stim + cue through reward channel; sets input_size=6 |
| Task variant | `cue_scale` | 2.0 | amplitude of the GNG cue signal |
| Task variant | `nogo_target` | — | target value for nogo response window (-1 or 0). None → nogo response FREE (no target at all): use with nolick_late_delay, whose don't-lick window then covers the whole after-cue span — "nogo = don't lick", one mechanism (eval boundary falls back to nogo_hinge_thresh) |
| Task variant | `go_target` | 1.0 | target value for go response window |
| Task variant | `input_scale` | 1.0 | global multiplier on all stimulus + cue input amplitudes |
| Task variant | `attention_input` | False | tonic attention/context input (last channel, =1 from first stim onset); input_size += 1 |
| Task variant | `attention_gated` | True | gate attention to the RETENTION+REWARD bracket: on from first-stim OFFSET through the reward (last-stim off + 1s), off during sample & final washout. False = original (from first-stim onset to end) |
| Task variant | `freeze_attention_input` | False | keep the attention channel's wi column FROZEN at init in ALL stages (untrained attention) |
| Task variant | `attention_scale` | 1.0 | amplitude multiplier on the tonic attention channel (× input_scale). >1 strengthens the readout-plane symmetry-breaking bias b_attn → pushes memory wells' κ₁ DOWN (attention-direct term). See ring_lowerplane_log §19. |
| Task variant | `rwd` | False | teacher-forced reward feedback |
| Task variant | `rwd_scale` | 1.0 | amplitude of the reward pulse (default +1) |
| (Attention is a DPA-learned tonic context input → frozen from GNG on by default.) | `freeze_input_stages` | dual |  |
| Has no effect on DPA learning (those channels are always zero during DPA). | `freeze_gng_input_during_dpa` | False |  |
| Has no effect on DPA learning (those channels are always zero during DPA). | `use_scheduler` | False | set False to use constant lr throughout |
| Has no effect on DPA learning (those channels are always zero during DPA). | `optimizer` | adam | "adamw" or "adam" (adam has no weight decay) |
| Has no effect on DPA learning (those channels are always zero during DPA). | `dpa_ckpt` | — | path to existing DPA checkpoint; skips DPA training if set |
| Has no effect on DPA learning (those channels are always zero during DPA). | `gng_ckpt` | — | path to existing GNG (naive) checkpoint; skips DPA+GNG if set |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dual_loss` | unified | only runnable value (2026-08-12); legacy names still VALIDATE |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `loss_thresh` | 0.5 | (legacy "threshold" loss — no longer runnable) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_weight` | 1.0 |  |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_weight` | 0.0 |  |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_decay_weight` | 1.0 | unified loss: weight on the gng decay-to-0 term (independent of gng_weight) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `pair_decay_weight` | 1.0 | unified loss: weight on the pairing decay-to-0 tail (independent of dpa/pair_weight) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_response` | True | windowed targets: re-add the 0.5 s response window after cue-off (go→go_target, nogo→nogo_target); scored by the unified rwd group |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dual_gng_memory` | True | DUAL stage: supervise the go/nogo pre-cue hold (the go/nogo working memory). False = not re-supervised in Dual (must survive on GNG-learned/frozen structure) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `rwd_go_weight` | 1.0 | unified loss: weight on the response-window go term (+1 hinge) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `rwd_nogo_weight` | 1.0 | unified loss: weight on the response-window nogo term (0→pin; allows go/nogo imbalance after cue) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `rwd_nogo_onesided` | False | unified loss: response window scores ONLY the nogo lick penalty relu(κ₁)² (no go +1 hinge, no nogo pin) — go response & no-lick value both free. False = go +1 hinge + nogo pin-to-0 |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `rwd_nogo_l1` | False | unified loss: nogo pin form — True = /κ₁/ (L1, NeuroFlame's 0.1·/overlap/, constant weak pull, permits a low autonomous well); False = κ₁² (L2, stiffer the deeper the well) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `rwd_keep_go_hinge` | False | unified loss: with rwd_nogo_onesided, KEEP the go +1 hinge (go MUST lick) → go-preserving one-sided. Under the shared response cue this forces the nogo well below the lick line (emergent well-push) instead of the never-lick collapse |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_rwd_onesided` | False | let rwd_nogo_onesided apply in the GNG STAGE too (default False = legacy: GNG always trains the nogo response two-sided/pinned). Safe only with rwd_keep_go_hinge (go supervision kept). Sign design: nogo response ≤0 free below in ALL stages |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_go_weight` | 1.0 | relative weight on go trials within gng_loss |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_nogo_weight` | 1.0 | relative weight on nogo trials within gng_loss |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `go_hinge_thresh` | 1.0 | if set, go response window uses relu(thresh-pred)² instead of MSE |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nogo_hinge_thresh` | -1.0 | hinge_gng no-lick threshold during the memory delay (0 after cue) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `ramping_gng` | False | cue-driven ramping decision (no delay memory-hold; nogo cancels the cue ramp) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `windowed_targets` | True | windowed transient decisions: 0.5 s pre-cue hold + short expression window (gng nogo not reset on cue); was `decay_decision` |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_decay_to_zero` | False | GNG-STAGE-ONLY decay override (Leon 2026-08-12): pin the lick back to 0 on BOTH trial types from the response end to TRIAL END in the GNG stage (even when decay_to_zero=False globally) — full return-to-rest, the softplus epochs cannot park inflated up OR down structure for Dual to inherit; GNG nolick becomes inert under the pin. Down-seating then happens purely in the Dual delay term |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `decay_to_zero` | False | within windowed_targets: add explicit decay-back-to-0 targets after the expression window (gng response & pairing). False = express then leave free. Decay zeros are PINNED (MSE-to-0) at all stages (pin_decay_zeros in the GNG/Dual losses; DPA ThresholdLoss pins via dpa_zero_thresh=0) — pre-sample baseline keeps its own separate term |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `decay_onesided` | False | decay window scored ONE-SIDED at thresh 0 (go-decay penalises κ₁>0, nogo-decay penalises κ₁<0) instead of pin-to-0 — each trace relaxes to rest from its own side (transient decision). Needs windowed_targets + decay_to_zero |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_post_response_window` | 0.5 **(differs, see above)** | DPA with response_in_cue=False: seconds of pairing target after test OFFSET (None = legacy 0.25 s); 0.5 = Leon's post-test choice (2026-09-22) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `response_in_cue` | True **(differs, see above)** | score the RESPONSE in the last 0.5 s of its triggering stimulus (gng response cue / DPA test) — cue ON — so the lick is input-DRIVEN, not held from memory. Removes the source of the go-rule "up copies". Needs windowed_targets |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_prelick_free` | True | DPA: pin the readout only PRE-SAMPLE; sample→test FREE (no delay no-lick supervision at all — wells placed by pairing training alone). Default False = legacy two-sided 0-pin to test-on |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `hinge_shape` | relu2 | unified loss hinge form — how force scales with violation size x: "relu2" (legacy, force 2x → vanishes near the threshold, so straddling is nearly free), "relu" (force 1 up to the threshold then 0 — the hinge/SVM form: every violation counts equally, crisp satisfaction; right shape for a BOUNDARY target like "don't lick"), or "softplus" (force σ(x) never vanishes on the correct side, depth keeps being rewarded; positive loss floor → stop_loss effectively disabled). relu2/relu share the same equilibrium (zero gradient once satisfied) — to sit strictly BEYOND a threshold, displace the threshold rather than change the shape |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_rwd_after_cue` | False | move the GNG RESPONSE window back to POST-cue (co, co+half) even when response_in_cue: the nogo pressure then acts on the RELAXING state near the well (a nolick-like push) instead of the cue-driven transient. Pairing stays per response_in_cue |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_hinge_thresh` | 1.0 | if set, DPA ±1 decision uses squared hinge toward ±thresh (DPA + dual stages) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_zero_thresh` | 0.0 | DPA ThresholdLoss dead-zone for ZERO targets (baselines); 0 ⇒ MSE-to-0 (pins baseline) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `hinge_squared` | True | DPA ThresholdLoss: True=relu(...)² (default), False=linear margin relu(...) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `aux_weight` | 1.0 | weight on the memory (non-decision) channels |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `bl_weight` | 1.0 | weight on the pre-sample baseline term |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `kappa1_reg_weight` | 0.0 | decision-subcriticality reg, ALL stages (DPA/GNG/Dual): weight*relu(gain·n_dec^T m_dec/N − 1)² — keeps decisions transient (no parking) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `kappa1_clamp` | — | HARD constraint: rescale m1,n1 so g·λ₁ ≤ this after each Dual step (vs. soft reg) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `kappa_gain_target` | — | CRITICALITY: pin ALL modes' g·λ to this value (two-sided) after each step, ALL stages |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `rate_reg_weight` | 0.0 | ACTIVITY L2: w·⟨rates²⟩ (all units/steps) added to the objective at every stage. For non-saturating φ (relu) the memory field is linear on each half-line and the one-sided hinge is free above θ, so nothing else prices a runaway (fdrelu: κ₀→30–40, rates→300). Bounds the activity; cannot by itself create a well (Leon 2026-09-08). |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `max_val_loss` | 100.0 | Optimization abort threshold on the val loss. Raise for relu + activity L2: at the unstable init ⟨r²⟩≈1e4 so the penalty alone exceeds 100 before the first step can shrink the rates (rr01 first launch died at epoch 1). |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nolick_weight` | 1.0 | one-sided no-lick penalty hinge(κ₁≤0, per hinge_shape) over free decision windows (Dual) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nolick_late_delay` | False | DON'T-LICK imposition (NeuroFlame train_dual.org port): restrict the nolick term to the LATE DELAY (cue-off → test-onset) of the Dual stage. Free-(NaN)-steps only, so the finite go/nogo response targets inside the span keep their own rwd terms; covers nogo + 'none' (pure-DPA) trials' late delay and the go post-response tail. 'none' trials sit ON the sample wells there → the term grades the wells' κ₁ directly (delay-time supervision, the §24f factorization route). Needs nolick_weight > 0. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nolick_full_delay` | True | extend the Dual don't-lick to the WHOLE delay (sample-off → test-onset) on the DPA TRIALS ONLY — the Dual-task trials with no go/nogo stimulus and no cue (sample→delay→test; the generator labels them gng="none", condition names "A_C"/"B_D", src/tasks.py:278). They sit ON the sample wells for that entire span, so the hinge grades the wells' κ₁ over ~3x more steps than nolick_late_delay. go/nogo trials keep the late window (a full-delay hinge would fight their +1 rule hold on the SAME κ₁ axis in rank-2). DPA trials are identified by having no finite decision target in the go/nogo span, so this REQUIRES dual_gng_memory=True (else go/nogo trials look like DPA trials); guarded below. Needs nolick_weight > 0. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nolick_nogo_in_cue` | True | NOGO rows: the don't-lick span starts at CUE ONSET (cue-on → test-on in Dual, cue-on → end in GNG) instead of cue-off — the cue IS the lick window and a nogo trial must not be in κ₁>0 during it. The rule Leon stated 2026-09-07: forbid κ₁>0 wherever a lick would be wrong (nogo from the cue on, DPA trials the whole delay, go after the cue) and let the network relocate the wells; the task's pushes (go stimulus, cue on both types) are never traded against. Rows are classified from their own hold target in the gng span, so it needs dual_gng_memory=True (guarded) and nolick_weight>0; combine with nolick_late_delay (go tail) + nolick_full_delay (DPA trials). |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `cue_duration` | 0.5 | DURATION of the go/nogo response cue in SECONDS (default 0.5 = the canonical make_timings value). Lengthens the cue window in the GNG and Dual timings only (cue ONSET is unchanged, so every loss window keyed to cue-on — nolick_nogo_in_cue, the pre-cue hold — is untouched; the offset moves, which matters only for windows keyed to cue-off: the gng response window and nolick_late_delay, both OFF in this line). Dual cue on 6.0 s, so 1.0 s ends at 7.0 s, still 1 s clear of test onset at 8.0 s; GNG cue on 4.0 s ends at 5.0 s inside the 6 s trial. The cue PUSH on kappa1 is an input property (§27d), so duration and amplitude are two separate doses of it. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_hold_window` | 0.5 | DPA-STAGE A/B memory supervision restricted to the last `dpa_hold_window` SECONDS ending at test onset (0.0 = legacy: sample ONSET → test onset, the whole delay + the sample itself). Mirrors the GNG identity hold, which is a 0.25 s window ending at cue onset (generate_gng_trials, hold_full_delay=False) — a short hold just before the readout instead of a clamp across the delay. The legacy span demands /κ₀/ ≥ θ already DURING the sample, so it prices the RISE TIME as well as the amplitude and self-amplification (λ⁺>1) is the cheapest way to meet it; a terminal window prices only "be there when read", which is what makes a subcritical λ⁺<1 solution affordable (§28c: λ⁺<1 is the necessary condition for a relu well). The Dual stage has NO κ₀ target at all (mem_* ≡ 0 there), so this flag touches the DPA stage only. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_hold_anchor` | sample | where dpa_hold_window sits: "test" = last hold_window s ending at test onset (default); "sample" = first hold_window s after sample offset, then FREE to decay across the delay; "none" = no A/B memory target at all (pairing at test is the only memory supervision) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_nolick_weight` | 0.0 | apply the same one-sided don't-lick over the DPA-STAGE delay (sample-off → test-onset). NOT the legacy two-sided pin (dpa_prelick_free=False), which clamps wells ON the line: one-sided leaves κ₁<0 free, so wells may seat at/below 0 but are never pulled back up. Intent: enter GNG with no up-structure to inherit. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nolick_thresh` | 0.0 | DISPLACE the no-lick hinge to κ₁ ≤ −thresh (all stages that use nolick). relu/relu² have zero gradient once satisfied, so a hinge at 0 seats wells AT the line no matter how wide the window — this is the only lever that buys DEPTH (§25e). |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nolick_shape` | — | shape of the no-lick term only (None = hinge_shape). "softplus" = logistic lick cost log(1+e^κ₁): keeps a tail below the line instead of switching off (Leon 2026-09-16, arm A) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `pair_pin` | True | TWO-SIDED ±1 pairing decision (DPA rwd group + Dual pair group): DPA becomes a bowl at κ₁=0 for any test kick (Leon 2026-09-17, design item 1) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dual_nolick_shape` | softplus | no-lick shape for the DUAL stage only (None = nolick_shape). design2: softplus tail in Dual, relu² in DPA/GNG (Leon 2026-09-17: the tail only where the bowl opposes it) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nolick_split_sample` | True | DUAL loss (GNG has no sample): no-lick hinge as two masked means (A-sample rows + B-sample rows) so both memories are pushed down equally; identity = sign of the κ₀ target → Dual needs dual_mem_targets=True (Leon 2026-09-16) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dual_mem_targets` | True | write the A/B memory hold (dpa_hold_window / dpa_hold_anchor) into the DUAL targets |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dual_mem_supervise` | False | … and USE it in the Dual memory loss (False = Dual memory supervised through pairing only; targets serve the nolick split) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `dpa_nolick_split` | False | DPA loss: split the pre-test κ₁ term (one-sided no-lick hinge, or the legacy κ₁=0 pin) into A-rows + B-rows means (Leon 2026-09-16) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_decouple_decision` | False | GNG stage: after each step project n[:,dec] ⟂ m[:,0], keeping the decision readout blind to the sample-memory direction. Targets the leakage that INVERTS the DPA memory during GNG (§26/§27): /κ₁(A)−κ₁(B)/ ≥ 0.85 → flip in 6/6, ≤ 0.27 → intact in 5/5. m[:,0] is frozen in GNG so n[:,dec] is the only party that can build the overlap. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `rwd_go_thresh` | — | RESPONSE-window go hinge threshold, decoupled from go_hinge_thresh (the hold). Set ABOVE the hold (e.g. 2.0 vs 1.0) so the parked +1 rule cannot satisfy the lick by itself — the cue must supply the difference in its 0.5 s, i.e. the cue gets a trained push (Leon 2026-09-04, §27g). Needs gng_response=True (guarded). |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_hold_pin` | False | score the go/nogo HOLD two-sided (pinned to ±θ) instead of one-sided. Companion to rwd_go_thresh: with a one-sided hold the net parks the go rule AT the lick threshold and the cue never has to push (cuego, 2026-09-04: hold +1.8, push unchanged). Pinned hold ⇒ the cue must carry lick−hold. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_hold_ceiling` | — | PREMATURE-LICK ceiling: one-sided hinge from ABOVE on the go HOLD (pre-cue steps, absolute κ₁), hinge(κ₁ − ceiling). The hold stays one-sided at θ (≥θ is a correct rule state; the two-sided pin was rejected — a lick is a threshold event) but is bounded above, so 'lick at the cue' = rwd_go_thresh > ceiling is well defined and the cue must supply the difference in its window (cuego parked the hold at +1.8 and let the cue idle, §27g). nogo untouched. Needs rwd_go_thresh > ceiling ≥ go hinge threshold (guarded). |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `attention_through_cue` | False | GNG task: gated attention stays ON through the cue (off at cue-OFF) instead of dropping at cue onset — the generic 'off at last-stim onset' rule made the GNG task inconsistent with Dual, where attention runs through the cue to test-on (Leon 2026-09-03). Affects GNG-task eval/traces; GNG-stage training is gradient-blind to the cue epoch unless a post-cue target exists. |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `gng_hold_full_delay` | True | windowed go/nogo hold spans stim-off → cue-on (GNG stage; go/nogo-stim-off → cue-on in Dual) instead of the 0.25 s pre-cue hold — symmetric with how the A/B memory is supervised across its whole delay. Eval trial generators don't carry it (targets unused in scoring). |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `hinge_gng` | True | unified one-sided decision hinge at κ₁=0 (go+nogo & match/nonmatch, all stages) |
| Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick. | `nogo_push_memory` | False | Dual: True FORCES the nogo memory to κ₁≤−1 (defeats emergent lowering); False (default) = gentle κ₁≤0, well location left to emerge |
| (set by __post_init__ / not a dataclass field) | `stp_U` | 0.2 | |
| (set by __post_init__ / not a dataclass field) | `eistp_K` | 250.0 | |

## Derived quantities (from `run_dt_alpha`, as sweep.run_single derives them)

| quantity | value | from |
|---|---|---|
| dt | 0.0150 s | `dt_base` × `tau_rec_frac` |
| α (rates) | 0.0750 | dt / τ |
| α_rec | 0.1000 | dt / (τ · `tau_rec_frac`) |
| η (state-noise s.d. of κ, the unit of every height) | noise · √(1 − e^(−2α)) = 0.373 | `noise` |
| trial timings (dual) | sample 2–3 s, Go/NoGo odor 4–5 s, cue 6–6.5 s, test 8–9 s, 11 s | `make_timings` |
| trial timings (DPA) | sample 2–3 s, test 8–9 s, 11 s | `make_timings` |
| trial timings (GNG) | odor 2–3 s, cue 4–4.5 s, 6 s | `make_timings` |
| targets (in-cue) | baseline pin κ₁ = 0 for 0–2 s; Go +1 from 5.8 s (hold) through the cue; NoGo −1 hold 5.8–6.0 s only; pairing ±1 in the last 0.5 s of the test | `response_in_cue=True` |
| targets (post) | as above but Go +1 at 5.8–6.0 s and 6.5–7.0 s; pairing ±1 at 9.0–9.5 s (DPA-only trials too) | `response_in_cue=False`, `dpa_post_response_window=0.5` |
| NoGo don't-lick hinge | softplus on κ₁ > 0 from cue onset to test onset, weight 1 — identical in both sweeps | `nolick_nogo_in_cue`, `dual_nolick_shape`, `nolick_weight` |
| sweep accuracy (results.jsonl) | window MEAN of κ₁ vs 0 (midpoint of `go_hinge_thresh` 1 and `nogo_hinge_thresh` −1); Go/NoGo window = last 0.5 s of the cue (in-cue) or 0.5 s after cue-off (post); pairing = last 0.5 s of the test or 1 s after test-off | `_dual_accuracy` |
| paper accuracy (Fig. 5, ED 17–18) | κ₁ > 0 at ANY step of the same windows (0.5 s after test-off for post) | `paper/*.py`, `LICK=any` |
| freezing | GNG: rank 0 (`freeze_rank0_gng`) + sample/test input columns; Dual: all input columns and the input bias (`freeze_input_stages=['dual']`) | `sweep.run_single` |
