"""
sweep.py — Sequential DPA → GNG → Dual training sweep.

Each run goes through three stages:
    1. DPA   : train on delayed paired association  → dpa_{run_id}.pth
    2. Naive : freeze rank-0 + DPA input dims, train on GNG → naive_{run_id}.pth
    3. Expert: freeze all input dims, train on dual task    → expert_{run_id}.pth

Completed run metrics are appended to {out_dir}/results.jsonl one line at a time.
If --wandb_project is given, each run is also logged to Weights & Biases with
per-epoch loss curves and per-stage accuracy summaries.

Usage
-----
    python sweep.py                                        # no W&B
    python sweep.py --wandb_project rnn-dual              # W&B on
    python sweep.py --n_gpus 1
    python sweep.py --out_dir ../results/dual/vanilla
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.optim as optim

from src.tasks import TaskTiming, make_timings, generate_dpa_trials, generate_gng_trials, generate_dual_trials
from src.models import LowRankModel, EILowRankModel, EISTPModel
from src.train  import Optimization, UnifiedLoss, train_val_split
from src.init   import init_dpa_internal_readout_prepost


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    run_id: str = "run_0"
    seed:   int = 0

    # Network architecture
    hidden_size:  int   = 512
    rank:         int   = 2
    gain:         float = 1.0
    input_size:   int   = 8       # 7 inputs + 1 reward channel
    target_rank:  int   = 2

    # Dynamics
    tau:          float = 0.3
    dt_base:      float = 0.03   # dt = dt_base * tau_rec_frac
    tau_rec_frac: float = 0.75   # scales both dt and tau_rec = tau * tau_rec_frac
    noise:        float = 0.5    # input noise prefactor; sigma = noise * sqrt(1 - exp(-alpha)^2)
    model_noise:  float = 0.0    # recurrent noise prefactor (same sigma formula)

    # Nonlinearity
    nonlinearity: str = "tanh"   # "tanh" | "relu" | "softplus" | "erf" | "elu" | "lif" | "lif_sc" | "tanh_asym"
    nl_gamma:     float = 0.0    # asymmetry strength for "tanh_asym" (φ = tanh + γ·tanh²)

    # Model type
    model_type:    str   = "lowrank"  # "lowrank" | "ei"  (EI = Dale backbone + low-rank E→E)
    n_inh:         int   = 128   # inhibitory units (EI only; E units = hidden_size)
    static_radius: float = 1.5   # spectral radius of frozen Dale backbone (EI only)
    low_rank_scale: float = 0.3  # init scale of trained low-rank E→E (EI only)
    low_rank_full: bool = False  # EI: train low-rank on whole N×N graph (E↔I) vs E→E block
    use_stp:       bool = False  # EI: short-term plasticity (Tsodyks–Markram) on E presynapse
    stp_U:         float = 0.2   # STP baseline utilisation
    stp_tau_f:     float = 1.5   # STP facilitation time constant (s)
    stp_tau_d:     float = 0.3   # STP depression time constant (s)
    # EISTPModel (model_type="eistp") — NeuroFlame dual-EI port
    n_neuron:      int   = 2000  # total units (E = frac·N)
    eistp_K:       float = 250.0 # avg presynaptic inputs (balanced 1/√K)
    j_stp:         float = 1.0   # E→E STP weight scale
    eistp_lr_scale: str  = "N"   # low-rank divisor (n@mᵀ)/lr_scale: "N" (=N_E) or "sqrtK"
    eistp_r_max:   float | None = None  # rate cap (anti-runaway); None = uncapped relu
    eistp_lr_ueqv: bool  = True  # True: m init = n (critical g_mem); False: independent random init
    eistp_lr_additive: bool = False  # E→E low-rank: False=C·(1+lr) multiplicative; True=C+lr additive/dense
    eistp_dense_cee: bool = False  # E→E backbone C_EE: False=sparse binary/√K; True=dense ones/N_E
    eistp_init_noise: float = 1.0  # init recurrent kick rates₀=relu(ff₀+init_noise·randn); 0 = deterministic/frozen

    # Initialisation
    init_style:         str   = "random"        # "structured" | "random"
    memory_lambda:      float = 0.8              # κ0 sample-memory init eigenvalue
    gng_lambda:         float = 0.8              # κ1 gng-memory init eigenvalue (rank-3 only)
    decision_lambda:    float = 0.5              # κ1 (rank-2) / κ2 (rank-3) action-mode init eigenvalue
    target_mn_corr:     float = 0.8
    target_out_mn_corr: float = 0.8
    sample_scale:       float = 1.0
    test_scale:         float = 1.0
    mix_strength:       float = 0.0
    decision_readout_mean: float = 0.0   # DC mean ⟨n₁⟩ of the decision readout (structured init). With a non-negative saturating φ (lif), resting κ₁=½⟨n₁⟩ → negative value seats the memory wells below the no-lick line (the clean saturating well-push). §20.
    rwd_input_scale:    float = 1.0   # scale of reward input alignment with u_read (structured init only)
    rwd_align_weight:   float = 0.0   # weight of reward-input ↔ n1 cosine alignment loss during DPA
    freeze_rank0_dual:  bool  = False  # also freeze rank-0 of m/n during the Dual stage
    project_go_on_n1:    bool  = False  # project go input column onto n₁ direction before GNG
    project_gng_orth_n0: bool  = False  # project go+nogo input columns orthogonal to n₀ before GNG
    use_fixed_weights:          bool  = False  # add frozen random W_fixed to recurrent dynamics
    fixed_weight_scale:         float = 0.8   # g/sqrt(N) scale of W_fixed; use g>>1 for strong backbone
    fixed_weight_orthogonalize: bool  = True   # project W_fixed ⊥ m,n (False = backbone shapes κ-plane)
    fixed_weight_sparsity:      float = 1.0   # keep-prob p of W_fixed entries (1.0 = dense; rescales 1/√p)
    use_unit_bias:              bool  = False  # per-unit bias inside φ; breaks κ-field odd symmetry
    unit_bias_trainable:        bool  = True   # train the unit bias (False = frozen random)
    unit_bias_scale:            float = 0.2    # init scale of the random per-unit bias
    use_rec_scale:              bool  = False  # trainable per-mode recurrent scale (decouples recurrence from readout)
    rwd_gng:             bool  = True   # teacher-forced reward during GNG stage (False = disable)

    # Training (shared across all stages)
    learning_rate:  float = 0.01
    weight_decay:   float = 0.01
    batch_size:     int   = 64
    grad_clip_norm: float | None = None   # None = disabled
    n_batch:        int   = 516   # trials per generated dataset
    stop_loss:      float = 0.005 # early-stop threshold (all stages)

    # Per-stage epoch budgets
    epochs_dpa:    int = 100
    epochs_gng:    int = 100
    epochs_dual:   int = 100
    # Curriculum: insert a Dual-paired (MATCH-only) stage between GNG and full Dual,
    # saved as the "naive" checkpoint (replacing GNG's).
    dual_paired_stage:  bool = False
    epochs_dual_paired: int  = 100

    # Task variant
    cue_on_go_input:  bool  = False
    go_on_rwd_input:  bool  = False  # route go stim + cue through reward channel; sets input_size=6
    cue_scale:        float = 1.0   # amplitude of the GNG cue signal
    nogo_target: float | None = 0.0 # target value for nogo response window (-1 or 0). None → nogo response FREE (no target at all): use with nolick_late_delay, whose don't-lick window then covers the whole after-cue span — "nogo = don't lick", one mechanism (eval boundary falls back to nogo_hinge_thresh)
    go_target:        float = 1.0   # target value for go response window
    input_scale:      float = 1.0   # global multiplier on all stimulus + cue input amplitudes
    attention_input:  bool  = False # tonic attention/context input (last channel, =1 from first stim onset); input_size += 1
    attention_gated:  bool  = False # gate attention to the RETENTION+REWARD bracket: on from first-stim OFFSET through the reward (last-stim off + 1s), off during sample & final washout. False = original (from first-stim onset to end)
    freeze_attention_input: bool = False # keep the attention channel's wi column FROZEN at init in ALL stages (untrained attention)
    attention_scale:  float = 1.0   # amplitude multiplier on the tonic attention channel (× input_scale). >1 strengthens the readout-plane symmetry-breaking bias b_attn → pushes memory wells' κ₁ DOWN (attention-direct term). See ring_lowerplane_log §19.
    rwd:              bool  = False  # teacher-forced reward feedback
    rwd_scale:        float = 1.0   # amplitude of the reward pulse (default +1)
    # Which stages freeze ALL input dims. Subset of ['dpa', 'gng', 'dual'].
    # GNG always freezes DPA + rwd + attention dims regardless; 'gng' extends that to all channels.
    # (Attention is a DPA-learned tonic context input → frozen from GNG on by default.)
    freeze_input_stages: list = field(default_factory=lambda: ["dual"])
    # Freeze GNG input dims (go/nogo/cue = channels 4..input_size-2) during DPA.
    # Prevents AdamW weight decay from zeroing them before GNG training starts.
    # Has no effect on DPA learning (those channels are always zero during DPA).
    freeze_gng_input_during_dpa: bool = False
    use_scheduler: bool = True  # set False to use constant lr throughout
    optimizer: str = "adamw"    # "adamw" or "adam" (adam has no weight decay)
    dpa_ckpt: str | None = None  # path to existing DPA checkpoint; skips DPA training if set
    gng_ckpt: str | None = None  # path to existing GNG (naive) checkpoint; skips DPA+GNG if set

    # Loss selection — "unified" is the ONLY loss sweep.py still runs (2026-08-12; the legacy
    # multi/separated/threshold paths were removed — run_one raises on them. The values remain
    # accepted by __post_init__ so historical arm dicts in make_configs still construct).
    #   "unified"    → UnifiedLoss at ALL THREE stages: value-based classes (+1/−1 one-sided
    #                  hinges beyond ±thresh, 0 → pin MSE-to-0, NaN free). Baseline is its own
    #                  pinned term (separate from decay); the decision channel splits into
    #                  SEPARATE gng (go/nogo) and pair (match/nonmatch) terms at the test onset,
    #                  and the two decay terms are separately weighted (gng/pair_decay_weight).
    #                  Weights map: gng_weight, dpa_weight→pair, aux_weight→mem, bl_weight, nolick.
    dual_loss:  str   = "unified"   # only runnable value (2026-08-12); legacy names still VALIDATE
                                    # so stored configs load, but run_single rejects them
    loss_thresh: float = 0.5    # (legacy "threshold" loss — no longer runnable)
    dpa_weight:      float = 1.0
    gng_weight:      float = 1.0
    gng_decay_weight:  float = 1.0      # unified loss: weight on the gng decay-to-0 term (independent of gng_weight)
    pair_decay_weight: float = 1.0      # unified loss: weight on the pairing decay-to-0 tail (independent of dpa/pair_weight)
    gng_response:      bool  = False    # windowed targets: re-add the 0.5 s response window after cue-off (go→go_target, nogo→nogo_target); scored by the unified rwd group
    dual_gng_memory:   bool  = True     # DUAL stage: supervise the go/nogo pre-cue hold (the go/nogo working memory). False = not re-supervised in Dual (must survive on GNG-learned/frozen structure)
    rwd_go_weight:     float = 1.0      # unified loss: weight on the response-window go term (+1 hinge)
    rwd_nogo_weight:   float = 1.0      # unified loss: weight on the response-window nogo term (0→pin; allows go/nogo imbalance after cue)
    rwd_nogo_onesided: bool  = False    # unified loss: response window scores ONLY the nogo lick penalty relu(κ₁)² (no go +1 hinge, no nogo pin) — go response & no-lick value both free. False = go +1 hinge + nogo pin-to-0
    rwd_nogo_l1:       bool  = False    # unified loss: nogo pin form — True = |κ₁| (L1, NeuroFlame's 0.1·|overlap|, constant weak pull, permits a low autonomous well); False = κ₁² (L2, stiffer the deeper the well)
    rwd_keep_go_hinge: bool  = False    # unified loss: with rwd_nogo_onesided, KEEP the go +1 hinge (go MUST lick) → go-preserving one-sided. Under the shared response cue this forces the nogo well below the lick line (emergent well-push) instead of the never-lick collapse
    gng_rwd_onesided:  bool  = False    # let rwd_nogo_onesided apply in the GNG STAGE too (default False = legacy: GNG always trains the nogo response two-sided/pinned). Safe only with rwd_keep_go_hinge (go supervision kept). Sign design: nogo response ≤0 free below in ALL stages
    gng_go_weight:   float = 1.0        # relative weight on go trials within gng_loss
    gng_nogo_weight: float = 1.0        # relative weight on nogo trials within gng_loss
    go_hinge_thresh: float | None = None  # if set, go response window uses relu(thresh-pred)² instead of MSE
    nogo_hinge_thresh: float = -1.0       # hinge_gng no-lick threshold during the memory delay (0 after cue)
    ramping_gng: bool = False             # cue-driven ramping decision (no delay memory-hold; nogo cancels the cue ramp)
    windowed_targets: bool = False        # windowed transient decisions: 0.5 s pre-cue hold + short expression window (gng nogo not reset on cue); was `decay_decision`
    gng_decay_to_zero: bool = False       # GNG-STAGE-ONLY decay override (Leon 2026-08-12): pin the lick back to 0 on BOTH trial types from the response end to TRIAL END in the GNG stage (even when decay_to_zero=False globally) — full return-to-rest, the softplus epochs cannot park inflated up OR down structure for Dual to inherit; GNG nolick becomes inert under the pin. Down-seating then happens purely in the Dual delay term
    decay_to_zero: bool = True            # within windowed_targets: add explicit decay-back-to-0 targets after the expression window (gng response & pairing). False = express then leave free. Decay zeros are PINNED (MSE-to-0) at all stages (pin_decay_zeros in the GNG/Dual losses; DPA ThresholdLoss pins via dpa_zero_thresh=0) — pre-sample baseline keeps its own separate term
    decay_onesided: bool = False          # decay window scored ONE-SIDED at thresh 0 (go-decay penalises κ₁>0, nogo-decay penalises κ₁<0) instead of pin-to-0 — each trace relaxes to rest from its own side (transient decision). Needs windowed_targets + decay_to_zero
    response_in_cue: bool = False         # score the RESPONSE in the last 0.5 s of its triggering stimulus (gng response cue / DPA test) — cue ON — so the lick is input-DRIVEN, not held from memory. Removes the source of the go-rule "up copies". Needs windowed_targets
    dpa_prelick_free: bool = False        # DPA: pin the readout only PRE-SAMPLE; sample→test FREE (no delay no-lick supervision at all — wells placed by pairing training alone). Default False = legacy two-sided 0-pin to test-on
    hinge_shape: str = "relu2"            # unified loss hinge form — how force scales with violation size x: "relu2" (legacy, force 2x → vanishes near the threshold, so straddling is nearly free), "relu" (force 1 up to the threshold then 0 — the hinge/SVM form: every violation counts equally, crisp satisfaction; right shape for a BOUNDARY target like "don't lick"), or "softplus" (force σ(x) never vanishes on the correct side, depth keeps being rewarded; positive loss floor → stop_loss effectively disabled). relu2/relu share the same equilibrium (zero gradient once satisfied) — to sit strictly BEYOND a threshold, displace the threshold rather than change the shape
    gng_rwd_after_cue: bool = False       # move the GNG RESPONSE window back to POST-cue (co, co+half) even when response_in_cue: the nogo pressure then acts on the RELAXING state near the well (a nolick-like push) instead of the cue-driven transient. Pairing stays per response_in_cue
    dpa_hinge_thresh: float | None = None # if set, DPA ±1 decision uses squared hinge toward ±thresh (DPA + dual stages)
    dpa_zero_thresh: float = 0.0   # DPA ThresholdLoss dead-zone for ZERO targets (baselines); 0 ⇒ MSE-to-0 (pins baseline)
    hinge_squared:   bool = True   # DPA ThresholdLoss: True=relu(...)² (default), False=linear margin relu(...)
    aux_weight:      float = 1.0   # weight on the memory (non-decision) channels
    bl_weight:       float = 1.0   # weight on the pre-sample baseline term
    kappa1_reg_weight: float = 0.0  # decision-subcriticality reg, ALL stages (DPA/GNG/Dual): weight*relu(gain·n_dec^T m_dec/N − 1)² — keeps decisions transient (no parking)
    kappa1_clamp:    float | None = None  # HARD constraint: rescale m1,n1 so g·λ₁ ≤ this after each Dual step (vs. soft reg)
    kappa_gain_target: float | None = None  # CRITICALITY: pin ALL modes' g·λ to this value (two-sided) after each step, ALL stages
    rate_reg_weight: float = 0.0   # ACTIVITY L2: w·⟨rates²⟩ (all units/steps) added to the objective at every stage. For non-saturating φ (relu) the memory field is linear on each half-line and the one-sided hinge is free above θ, so nothing else prices a runaway (fdrelu: κ₀→30–40, rates→300). Bounds the activity; cannot by itself create a well (Leon 2026-09-08).
    max_val_loss:   float = 100.0  # Optimization abort threshold on the val loss. Raise for relu + activity L2: at the unstable init ⟨r²⟩≈1e4 so the penalty alone exceeds 100 before the first step can shrink the rates (rr01 first launch died at epoch 1).
    nolick_weight:   float = 0.0   # one-sided no-lick penalty hinge(κ₁≤0, per hinge_shape) over free decision windows (Dual)
    nolick_late_delay: bool = False # DON'T-LICK imposition (NeuroFlame train_dual.org port): restrict the nolick term to the LATE DELAY (cue-off → test-onset) of the Dual stage. Free-(NaN)-steps only, so the finite go/nogo response targets inside the span keep their own rwd terms; covers nogo + 'none' (pure-DPA) trials' late delay and the go post-response tail. 'none' trials sit ON the sample wells there → the term grades the wells' κ₁ directly (delay-time supervision, the §24f factorization route). Needs nolick_weight > 0.
    nolick_full_delay: bool = False # extend the Dual don't-lick to the WHOLE delay (sample-off → test-onset) on the DPA TRIALS ONLY — the Dual-task trials with no go/nogo stimulus and no cue (sample→delay→test; the generator labels them gng="none", condition names "A_C"/"B_D", src/tasks.py:278). They sit ON the sample wells for that entire span, so the hinge grades the wells' κ₁ over ~3x more steps than nolick_late_delay. go/nogo trials keep the late window (a full-delay hinge would fight their +1 rule hold on the SAME κ₁ axis in rank-2). DPA trials are identified by having no finite decision target in the go/nogo span, so this REQUIRES dual_gng_memory=True (else go/nogo trials look like DPA trials); guarded below. Needs nolick_weight > 0.
    nolick_nogo_in_cue: bool = False # NOGO rows: the don't-lick span starts at CUE ONSET (cue-on → test-on in Dual, cue-on → end in GNG) instead of cue-off — the cue IS the lick window and a nogo trial must not be in κ₁>0 during it. The rule Leon stated 2026-09-07: forbid κ₁>0 wherever a lick would be wrong (nogo from the cue on, DPA trials the whole delay, go after the cue) and let the network relocate the wells; the task's pushes (go stimulus, cue on both types) are never traded against. Rows are classified from their own hold target in the gng span, so it needs dual_gng_memory=True (guarded) and nolick_weight>0; combine with nolick_late_delay (go tail) + nolick_full_delay (DPA trials).
    cue_duration:    float = 0.5   # DURATION of the go/nogo response cue in SECONDS (default 0.5 = the canonical make_timings value). Lengthens the cue window in the GNG and Dual timings only (cue ONSET is unchanged, so every loss window keyed to cue-on — nolick_nogo_in_cue, the pre-cue hold — is untouched; the offset moves, which matters only for windows keyed to cue-off: the gng response window and nolick_late_delay, both OFF in this line). Dual cue on 6.0 s, so 1.0 s ends at 7.0 s, still 1 s clear of test onset at 8.0 s; GNG cue on 4.0 s ends at 5.0 s inside the 6 s trial. The cue PUSH on kappa1 is an input property (§27d), so duration and amplitude are two separate doses of it.
    dpa_hold_window: float = 0.0   # DPA-STAGE A/B memory supervision restricted to the last `dpa_hold_window` SECONDS ending at test onset (0.0 = legacy: sample ONSET → test onset, the whole delay + the sample itself). Mirrors the GNG identity hold, which is a 0.25 s window ending at cue onset (generate_gng_trials, hold_full_delay=False) — a short hold just before the readout instead of a clamp across the delay. The legacy span demands |κ₀| ≥ θ already DURING the sample, so it prices the RISE TIME as well as the amplitude and self-amplification (λ⁺>1) is the cheapest way to meet it; a terminal window prices only "be there when read", which is what makes a subcritical λ⁺<1 solution affordable (§28c: λ⁺<1 is the necessary condition for a relu well). The Dual stage has NO κ₀ target at all (mem_* ≡ 0 there), so this flag touches the DPA stage only.
    dpa_nolick_weight: float = 0.0  # apply the same one-sided don't-lick over the DPA-STAGE delay (sample-off → test-onset). NOT the legacy two-sided pin (dpa_prelick_free=False), which clamps wells ON the line: one-sided leaves κ₁<0 free, so wells may seat at/below 0 but are never pulled back up. Intent: enter GNG with no up-structure to inherit.
    nolick_thresh:   float = 0.0   # DISPLACE the no-lick hinge to κ₁ ≤ −thresh (all stages that use nolick). relu/relu² have zero gradient once satisfied, so a hinge at 0 seats wells AT the line no matter how wide the window — this is the only lever that buys DEPTH (§25e).
    gng_decouple_decision: bool = False # GNG stage: after each step project n[:,dec] ⟂ m[:,0], keeping the decision readout blind to the sample-memory direction. Targets the leakage that INVERTS the DPA memory during GNG (§26/§27): |κ₁(A)−κ₁(B)| ≥ 0.85 → flip in 6/6, ≤ 0.27 → intact in 5/5. m[:,0] is frozen in GNG so n[:,dec] is the only party that can build the overlap.
    rwd_go_thresh: float | None = None # RESPONSE-window go hinge threshold, decoupled from go_hinge_thresh (the hold). Set ABOVE the hold (e.g. 2.0 vs 1.0) so the parked +1 rule cannot satisfy the lick by itself — the cue must supply the difference in its 0.5 s, i.e. the cue gets a trained push (Leon 2026-09-04, §27g). Needs gng_response=True (guarded).
    gng_hold_pin: bool = False    # score the go/nogo HOLD two-sided (pinned to ±θ) instead of one-sided. Companion to rwd_go_thresh: with a one-sided hold the net parks the go rule AT the lick threshold and the cue never has to push (cuego, 2026-09-04: hold +1.8, push unchanged). Pinned hold ⇒ the cue must carry lick−hold.
    gng_hold_ceiling: float | None = None # PREMATURE-LICK ceiling: one-sided hinge from ABOVE on the go HOLD (pre-cue steps, absolute κ₁), hinge(κ₁ − ceiling). The hold stays one-sided at θ (≥θ is a correct rule state; the two-sided pin was rejected — a lick is a threshold event) but is bounded above, so 'lick at the cue' = rwd_go_thresh > ceiling is well defined and the cue must supply the difference in its window (cuego parked the hold at +1.8 and let the cue idle, §27g). nogo untouched. Needs rwd_go_thresh > ceiling ≥ go hinge threshold (guarded).
    attention_through_cue: bool = False # GNG task: gated attention stays ON through the cue (off at cue-OFF) instead of dropping at cue onset — the generic 'off at last-stim onset' rule made the GNG task inconsistent with Dual, where attention runs through the cue to test-on (Leon 2026-09-03). Affects GNG-task eval/traces; GNG-stage training is gradient-blind to the cue epoch unless a post-cue target exists.
    gng_hold_full_delay: bool = False # windowed go/nogo hold spans stim-off → cue-on (GNG stage; go/nogo-stim-off → cue-on in Dual) instead of the 0.25 s pre-cue hold — symmetric with how the A/B memory is supervised across its whole delay. Eval trial generators don't carry it (targets unused in scoring).
    hinge_gng:       bool  = False # unified one-sided decision hinge at κ₁=0 (go+nogo & match/nonmatch, all stages)
    nogo_push_memory: bool = False # Dual: True FORCES the nogo memory to κ₁≤−1 (defeats emergent lowering); False (default) = gentle κ₁≤0, well location left to emerge

    # Output
    out_dir: str = "../results/dual/vanilla"

    def __post_init__(self):
        if self.go_on_rwd_input:
            self.input_size = 8 - 2 - int(not self.rwd)   # go+cue merged into rwd channel
        else:
            self.input_size = 8 - int(self.cue_on_go_input) - int(not self.rwd)
        if self.attention_input:
            # tonic attention occupies an appended LAST channel (=1 from first stim onset).
            # The reward feedback (models.py rwd_channel=-1) also writes the LAST channel,
            # so it must be OFF or it corrupts the attention signal (during GNG, rwd_gng).
            if self.rwd or self.go_on_rwd_input:
                raise ValueError("attention_input needs the last channel; set rwd=False, go_on_rwd_input=False.")
            self.rwd_gng = False   # prevent GNG reward-feedback writing onto the attention channel
            self.input_size += 1
        if self.dual_loss not in ("multi", "separated", "threshold", "unified"):
            raise ValueError(f"dual_loss must be 'multi', 'separated', 'threshold', or 'unified', got {self.dual_loss!r}")


# ---------------------------------------------------------------------------
# Accuracy helpers  (defined here so they don't live in the general modules)
# ---------------------------------------------------------------------------

def _dpa_score(pred, y, timing, response_in_cue=False):
    """Score DPA match/nonmatch from the SUPERVISED decision window — the timesteps where the ±1
    pairing target is actually set. Robust to windowed_targets/decay_to_zero: those express the
    decision as a ±1 plateau that then decays to 0, so the old `y[:, -1, -1]` (last-timestep target)
    reads 0/NaN and mislabels every trial → spurious 0.50 + pair=nan. Here we read pred and target
    over exactly the supervised steps instead. response_in_cue moves that window into the last 0.5 s
    of the TEST (test-off − 0.5 s → test-off) so the window starts half a second earlier."""
    half       = int(round(0.5 / timing.dt))
    decision_t = int(timing.n_stim_off[1]) - (half if response_in_cue else 0)
    tgt   = torch.nan_to_num(y[..., -1], nan=0.0)          # (B,T): ±1 in the decision window, 0 else
    dmask = torch.zeros_like(tgt, dtype=torch.bool)
    dmask[:, decision_t:] = tgt[:, decision_t:] != 0        # only post-test supervised steps
    pred_dec    = (pred * dmask).sum(1) / dmask.sum(1).clamp_min(1)   # mean readout over the window
    target_sign = tgt.masked_fill(~dmask, 0.0).sum(1)       # >0 match, <0 nonmatch
    correct     = (pred_dec > 0) == (target_sign > 0)
    valid       = dmask.any(1)
    pair_mask   = valid & (target_sign > 0)
    unpair_mask = valid & (target_sign < 0)
    return {
        "overall": correct[valid].float().mean().item()       if valid.any()       else float("nan"),
        "pair":    correct[pair_mask].float().mean().item()   if pair_mask.any()   else float("nan"),
        "unpair":  correct[unpair_mask].float().mean().item() if unpair_mask.any() else float("nan"),
    }


@torch.no_grad()
def _dpa_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1, input_scale=1.0, attention_input=False, attention_gated=False, attention_scale=1.0, windowed_targets=False, decay_to_zero=True):
    model.eval()
    X, y = generate_dpa_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, input_scale=input_scale,
                                attention_input=attention_input, attention_gated=attention_gated, attention_scale=attention_scale,
                                windowed_targets=windowed_targets, decay_to_zero=decay_to_zero)
    pred = model(X.to(device), y.to(device))[..., -1].cpu()
    return _dpa_score(pred, y, timing)["overall"]


@torch.no_grad()
def _dpa_accuracy_by_type(model, timing, input_size, noise, device, n_trials=1024, target_rank=1, input_scale=1.0, attention_input=False, attention_gated=False, attention_scale=1.0, windowed_targets=False, decay_to_zero=True, response_in_cue=False):
    model.eval()
    X, y = generate_dpa_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, input_scale=input_scale,
                                attention_input=attention_input, attention_gated=attention_gated, attention_scale=attention_scale,
                                windowed_targets=windowed_targets, decay_to_zero=decay_to_zero, response_in_cue=response_in_cue)
    pred = model(X.to(device), y.to(device))[..., -1].cpu()
    return _dpa_score(pred, y, timing, response_in_cue=response_in_cue)


@torch.no_grad()
def _gng_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                  cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False, input_scale=1.0, attention_input=False, attention_gated=False, attention_scale=1.0,
                  go_hinge_thresh=None, nogo_hinge_thresh=-1.0, gng_rwd_after_cue=False, attention_through_cue=False):
    model.eval()
    X, y = generate_gng_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, cue_on_go_input=cue_on_go_input,
                                cue_scale=cue_scale, nogo_target=nogo_target,
                                go_on_rwd_input=go_on_rwd_input, input_scale=input_scale,
                                attention_input=attention_input, attention_gated=attention_gated, attention_scale=attention_scale,
                                gng_rwd_after_cue=gng_rwd_after_cue, attention_through_cue=attention_through_cue)
    pred        = model(X.to(device), y.to(device))[..., -1].cpu()
    stim_epoch  = slice(int(timing.n_stim_on[0]), int(timing.n_stim_off[0]))
    go_ch       = input_size - 1 if go_on_rwd_input else 4
    ngo_ch      = 4              if go_on_rwd_input else 5
    is_go       = X[:, stim_epoch, go_ch].mean(1) > X[:, stim_epoch, ngo_ch].mean(1)
    decision_t  = int(timing.n_stim_off[1])
    pred_final  = pred[:, decision_t:].mean(1)
    # decision boundary follows the LOSS: go hinge at go_hinge_thresh (1.0 legacy). With sign-based
    # hinges (go_hinge_thresh=0) the boundary is 0 — a σ-scaled go response must not be mis-scored.
    th_go       = go_hinge_thresh if go_hinge_thresh is not None else 1.0
    nogo_ref    = (nogo_hinge_thresh if nogo_target is None   # None = nogo response FREE (nolick covers it)
                   else max(nogo_target, nogo_hinge_thresh))   # hinge boundary wins over target amplitude
    thresh      = (th_go + nogo_ref) / 2.0
    return ((pred_final > thresh) == is_go).float().mean().item()


@torch.no_grad()
def _gng_accuracy_by_type(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                           cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False, input_scale=1.0, attention_input=False, attention_gated=False, attention_scale=1.0, response_in_cue=False,
                           go_hinge_thresh=None, nogo_hinge_thresh=-1.0, gng_rwd_after_cue=False, attention_through_cue=False):
    model.eval()
    X, y = generate_gng_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, cue_on_go_input=cue_on_go_input,
                                cue_scale=cue_scale, nogo_target=nogo_target,
                                go_on_rwd_input=go_on_rwd_input, input_scale=input_scale,
                                attention_input=attention_input, attention_gated=attention_gated, attention_scale=attention_scale, response_in_cue=response_in_cue,
                                gng_rwd_after_cue=gng_rwd_after_cue, attention_through_cue=attention_through_cue)
    pred        = model(X.to(device), y.to(device))[..., -1].cpu()
    stim_epoch  = slice(int(timing.n_stim_on[0]), int(timing.n_stim_off[0]))
    go_ch       = input_size - 1 if go_on_rwd_input else 4
    ngo_ch      = 4              if go_on_rwd_input else 5
    is_go       = X[:, stim_epoch, go_ch].mean(1) > X[:, stim_epoch, ngo_ch].mean(1)
    half        = int(round(0.5 / timing.dt))
    # response_in_cue: read the lick in the last 0.5 s of the response cue (cue ON, before cue-off);
    # else the legacy window from cue-off to trial end.
    co          = int(timing.n_stim_off[1])
    _in_cue     = response_in_cue and not gng_rwd_after_cue
    pred_final  = (pred[:, co - half:co] if _in_cue else pred[:, co:]).mean(1)
    # decision boundary follows the LOSS: go hinge at go_hinge_thresh (1.0 legacy). With sign-based
    # hinges (go_hinge_thresh=0) the boundary is 0 — a σ-scaled go response must not be mis-scored.
    th_go       = go_hinge_thresh if go_hinge_thresh is not None else 1.0
    nogo_ref    = (nogo_hinge_thresh if nogo_target is None   # None = nogo response FREE (nolick covers it)
                   else max(nogo_target, nogo_hinge_thresh))   # hinge boundary wins over target amplitude
    thresh      = (th_go + nogo_ref) / 2.0
    correct     = (pred_final > thresh) == is_go
    return {
        "overall": correct.float().mean().item(),
        "go":      correct[is_go].float().mean().item()  if is_go.any()  else float("nan"),
        "nogo":    correct[~is_go].float().mean().item() if (~is_go).any() else float("nan"),
    }


@torch.no_grad()
def _dual_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                   cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False, input_scale=1.0, attention_input=False, attention_gated=False, attention_scale=1.0,
                   go_target=1.0, response_in_cue=False, go_hinge_thresh=None, nogo_hinge_thresh=-1.0, gng_rwd_after_cue=False):
    model.eval()
    X, y, _, condition_names = generate_dual_trials(
        n_trials, timing=timing, input_size=input_size, noise=noise, target_rank=target_rank,
        cue_on_go_input=cue_on_go_input, cue_scale=cue_scale, nogo_target=nogo_target,
        go_on_rwd_input=go_on_rwd_input, input_scale=input_scale,
        attention_input=attention_input, attention_gated=attention_gated, attention_scale=attention_scale,
        response_in_cue=response_in_cue, gng_rwd_after_cue=gng_rwd_after_cue,
    )
    pred  = model(X.to(device), y.to(device))[..., -1].cpu()
    names = np.asarray(condition_names).astype(str)

    # pairing expression window: the 1 s right after test-off ([n_off[3], n_off[3]+1s]). Averaging to
    # END instead diluted the signal across the decay-to-0 / free tail (windowed_targets) → understated
    # (esp. the decay arm, where match is pulled back toward 0). For non-windowed targets the decision is
    # held past test-off so this window still captures it.
    half      = int(round(0.5 / timing.dt))
    # response_in_cue: pairing decision read in the last 0.5 s of the TEST (test ON, before test-off);
    # else the legacy 1 s window right after test-off.
    if response_in_cue:
        pred_dpa = pred[:, int(timing.n_stim_off[3]) - half:int(timing.n_stim_off[3])].mean(1)
    else:
        dpa_start = int(timing.n_stim_off[3])
        pred_dpa  = pred[:, dpa_start:dpa_start + 2*half].mean(1)
    # pairing label from the ground-truth condition (match = A→C or B→D), NOT y[:, -1, -1]: with the
    # ramp-style pairing target the last timestep is NaN → NaN>0 marks every trial "nonmatch" → 0.5.
    samp      = np.array([n[0]  for n in names])
    tst       = np.array([n[-1] for n in names])
    is_pair   = torch.as_tensor(((samp == "A") & (tst == "C")) | ((samp == "B") & (tst == "D")))
    dpa_acc   = ((pred_dpa > 0) == is_pair).float().mean().item()

    # go/nogo: evaluate κ₁ in the AFTER-CUE target window (where the response target actually
    # lives, [n_off[2], n_off[2]+½·(test−cue2)]), and score each side by whether it goes to its
    # target — go reaches the go side, nogo reaches ≤ its target — past the go/nogo midpoint.
    # response_in_cue: read in the last 0.5 s of the response cue (cue ON, before cue-off).
    if response_in_cue and not gng_rwd_after_cue:
        rwd_start = int(timing.n_stim_off[2]) - half
        rwd_stop  = int(timing.n_stim_off[2])
    else:
        rwd_start = int(timing.n_stim_off[2])
        rwd_stop  = int(timing.n_stim_off[2] + (timing.n_stim_off[3] - timing.n_stim_on[3]) / 2)
    pred_gng  = pred[:, rwd_start:rwd_stop].mean(1)
    is_go     = torch.as_tensor(["_go_"   in n for n in names])
    is_ng     = torch.as_tensor(["_nogo_" in n for n in names])
    # decision boundary = the go/nogo MIDPOINT (matches _gng_accuracy). "nogo correct" means "not
    # licking" = κ₁ below the midpoint (go sits at go_target, nogo at ~nogo_target). Thresholding at
    # the exact nogo_target instead mis-scored a correct nogo whose brief cue-overshoot tipped the
    # windowed mean just above it (e.g. emergent nolick=0 runs read ~0.2 while behaving correctly).
    # go hinge (go_hinge_thresh, sign-based when 0) overrides go_target as the go-side reference.
    th_go     = go_hinge_thresh if go_hinge_thresh is not None else go_target
    nogo_ref  = (nogo_hinge_thresh if nogo_target is None     # None = nogo response FREE (nolick covers it)
                 else max(nogo_target, nogo_hinge_thresh))    # hinge boundary wins over target amplitude
    thresh    = (th_go + nogo_ref) / 2.0
    go_acc    = (pred_gng[is_go] >  thresh).float().mean().item() if is_go.any() else float("nan")
    nogo_acc  = (pred_gng[is_ng] <= thresh).float().mean().item() if is_ng.any() else float("nan")
    gng_acc   = float(np.nanmean([go_acc, nogo_acc]))
    return dpa_acc, gng_acc, go_acc, nogo_acc


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _kappa1_regularizer(config: "RunConfig", model):
    """Soft decision-subcriticality reg: w·relu(g·λ_dec − 1)², λ_dec = n_decᵀm_dec/N (last low-rank
    column = κ₁ rank-2 / κ₂ rank-3). Applied at ALL stages (DPA/GNG/Dual) so the decision mode never
    trains supercritical — a supercritical decision self-gain is what lets κ₁ PARK at a decision
    (persistent lick attractor) instead of expressing transiently and relaxing back to 0."""
    if config.kappa1_reg_weight <= 0.0:
        return None
    _w    = config.kappa1_reg_weight
    _gain = float(model.gain) if torch.is_tensor(model.gain) else float(model.gain)
    _dec  = config.rank - 1
    def reg(m, _w=_w, _gain=_gain, _dec=_dec):
        N     = m.m.shape[0]
        lam_d = _gain * (m.n[:, _dec] @ m.m[:, _dec]) / N   # gain * n_dec^T m_dec / N
        return _w * torch.relu(lam_d - 1.0) ** 2
    return reg


def run_single(config: RunConfig, device: str, models_dir: str | None = None,
               wandb_project: str | None = None) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    DT        = config.dt_base * config.tau_rec_frac
    alpha     = DT / config.tau
    alpha_rec = DT / (config.tau * config.tau_rec_frac)
    noise            = float(config.noise       * torch.sqrt(1.0 - torch.exp(torch.tensor(-alpha)) ** 2))
    model_noise_sigma = float(config.model_noise * torch.sqrt(1.0 - torch.exp(torch.tensor(-alpha)) ** 2))

    # Task timings live in src/tasks.make_timings (single source shared with plot_sweep.py)
    _timings    = make_timings(DT)
    dpa_timing  = _timings["dpa"]
    gng_timing  = _timings["gng"]
    dual_timing = _timings["dual"]
    if config.cue_duration != 0.5:      # widen the cue window (onset fixed, offset moves)
        def _recue(t: TaskTiming, idx: int) -> TaskTiming:
            off = list(t.stim_off); off[idx] = t.stim_on[idx] + config.cue_duration
            # in the GNG task the cue IS the last stimulus, so bound by trial end there
            nxt = t.stim_on[idx + 1] if idx + 1 < len(t.stim_on) else t.t_steps
            assert off[idx] <= nxt + 1e-9, (
                f"cue_duration={config.cue_duration} runs the cue (on {t.stim_on[idx]}s) past "
                f"{nxt}s")
            return dataclasses.replace(t, stim_off=off)
        gng_timing  = _recue(gng_timing, 1)    # GNG: cue is stimulus index 1
        dual_timing = _recue(dual_timing, 2)   # Dual: cue is stimulus index 2
        print(f"[{config.run_id}]  cue_duration={config.cue_duration}s -> gng cue {gng_timing.stim_on[1]}-{gng_timing.stim_off[1]}s, "
              f"dual cue {dual_timing.stim_on[2]}-{dual_timing.stim_off[2]}s", flush=True)


    if models_dir is None:
        models_dir = os.path.join(config.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    rid    = config.run_id
    t_run  = time.time()
    SEP    = f"[{rid}] " + "─" * 60

    def _log_params(label: str):
        m    = model.m.detach().cpu().numpy()
        n    = model.n.detach().cpu().numpy()
        N    = m.shape[0]
        rank = m.shape[1]
        gain = float(model.gain) if torch.is_tensor(model.gain) else float(model.gain)

        m_norms = np.linalg.norm(m, axis=0)
        n_norms = np.linalg.norm(n, axis=0)
        corrs   = [float(np.dot(m[:, r], n[:, r]) / (m_norms[r] * n_norms[r] + 1e-12))
                   for r in range(rank)]

        # κ-space effective recurrent Jacobian: gain * n^T m / N  (rank × rank)
        J_kappa = gain * (n.T @ m) / N
        eigvals = np.linalg.eigvals(J_kappa)

        wi_w    = model.wi.weight.detach().cpu().numpy()
        wi_fro  = float(np.linalg.norm(wi_w, 'fro'))
        wi_col  = np.linalg.norm(wi_w, axis=0)  # per-input-channel norm

        p = f"[{rid}]"
        print(f"{p}  ┌── params: {label}", flush=True)
        for r in range(rank):
            print(f"{p}  │  rank {r}: ||m||={m_norms[r]:.3f}  ||n||={n_norms[r]:.3f}"
                  f"  corr(m,n)={corrs[r]:+.3f}", flush=True)
        eig_str = "  ".join(f"{e.real:+.4f}" + (f"{e.imag:+.4f}j" if abs(e.imag) > 1e-6 else "")
                            for e in eigvals)
        # For EISTPModel the low-rank modulates the STP E→E weight (1 + n mᵀ/N), so this is
        # the low-rank overlap, NOT the full effective Jacobian (which is STP-dependent).
        jac_label = ("low-rank overlap n^Tm/N (modulates STP E→E)"
                     if model.__class__.__name__ == "EISTPModel"
                     else "κ-Jacobian eigvals (gain·n^Tm/N)")
        print(f"{p}  │  {jac_label}: {eig_str}", flush=True)
        print(f"{p}  │  Wi: ||·||_F={wi_fro:.3f}  per-channel={' '.join(f'{v:.2f}' for v in wi_col)}",
              flush=True)
        print(f"{p}  └{'─'*50}", flush=True)

    def _stage_header(name: str, epochs: int, freeze_lr: list, freeze_cols: list):
        p = f"[{rid}]"
        print(SEP, flush=True)
        print(f"{p}  STAGE: {name}   device={device}   epochs={epochs}", flush=True)
        if freeze_cols:
            print(f"{p}  freeze recurrent cols: {freeze_cols}", flush=True)
        if freeze_lr:
            print(f"{p}  freeze input dims:     {freeze_lr}", flush=True)
        print(SEP, flush=True)

    def _stage_summary(name: str, train_l: list, val_l: list,
                       acc: dict, t0: float):
        elapsed = time.time() - t0
        p = f"[{rid}]"
        loss_str = f"  final train={train_l[-1]:.4f}  val={val_l[-1]:.4f}" if train_l else "  (checkpoint)"
        print(f"{p}  {name} done in {elapsed:.1f}s{loss_str}"
              f"  dpa={acc['dpa']:.3f}  gng={acc['gng']:.3f}",
              flush=True)

    if config.model_type == "eistp":
        model = EISTPModel(
            n_neuron=config.n_neuron, K=config.eistp_K, rank=config.rank, gain=config.gain,
            dt=DT, input_size=config.input_size,
            stp_use=config.stp_U, stp_tau_fac=config.stp_tau_f, stp_tau_rec=config.stp_tau_d,
            j_stp=config.j_stp, lr_ini=config.low_rank_scale, lr_scale=config.eistp_lr_scale,
            lr_ueqv=config.eistp_lr_ueqv, lr_additive=config.eistp_lr_additive,
            dense_cee=config.eistp_dense_cee, r_max=config.eistp_r_max,
            init_noise=config.eistp_init_noise,
            train_inputs=False, nonlinearity=config.nonlinearity,
            device=device, seed=config.seed,
        )
    elif config.model_type == "ei":
        model = EILowRankModel(
            input_size=config.input_size, output_size=0, rank=config.rank,
            n_exc=config.hidden_size, n_inh=config.n_inh, gain=config.gain,
            alpha=alpha, alpha_rec=alpha_rec, noise=0.0,
            static_radius=config.static_radius, low_rank_scale=config.low_rank_scale,
            low_rank_full=config.low_rank_full,
            use_stp=config.use_stp, stp_U=config.stp_U, stp_tau_f=config.stp_tau_f,
            stp_tau_d=config.stp_tau_d, stp_dt=DT,
            rwd=config.rwd, rwd_scale=config.rwd_scale,
            nonlinearity=config.nonlinearity, device=device, seed=config.seed,
        )
    else:
        model = LowRankModel(
            input_size=config.input_size, hidden_size=config.hidden_size,
            output_size=0, rank=config.rank, gain=config.gain,
            alpha=alpha, alpha_rec=alpha_rec, noise=0.0,
            rwd=config.rwd, rwd_scale=config.rwd_scale,
            use_fixed_weights=config.use_fixed_weights,
            fixed_weight_scale=config.fixed_weight_scale,
            fixed_weight_orthogonalize=config.fixed_weight_orthogonalize,
            fixed_weight_sparsity=config.fixed_weight_sparsity,
            nonlinearity=config.nonlinearity,
            nl_gamma=config.nl_gamma,
            use_unit_bias=config.use_unit_bias,
            unit_bias_trainable=config.unit_bias_trainable,
            unit_bias_scale=config.unit_bias_scale,
            use_rec_scale=config.use_rec_scale,
            device=device,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p = f"[{rid}]"
    print(f"{p} {'═'*60}", flush=True)
    print(f"{p}  RUN START  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{p}  run_id={rid}  seed={config.seed}  device={device}", flush=True)
    if config.model_type == "eistp":
        print(f"{p}  arch:  N={config.n_neuron} (E={model.n_exc} I={model.n_inh})"
              f"  K={config.eistp_K:g} (prob {config.eistp_K/model.n_exc:.3f})"
              f"  rank={config.rank}  gain={config.gain}  input_size={config.input_size}"
              f"  params={n_params:,}", flush=True)
    else:
        print(f"{p}  arch:  hidden={config.hidden_size}  rank={config.rank}  gain={config.gain}"
              f"  input_size={config.input_size}  params={n_params:,}", flush=True)
    print(f"{p}  task:  cue_on_go={config.cue_on_go_input}  rwd={config.rwd}"
          f"  rwd_scale={config.rwd_scale}  freeze_input_stages={config.freeze_input_stages}  init={config.init_style}", flush=True)
    if config.model_type == "eistp":
        print(f"{p}  dynamics: dt={DT:.4f}  tau={model.tau}s  tau_syn={model.tau_syn}s"
              f"  STP(use={config.stp_U} tau_fac={model.stp_tau_fac}s tau_rec={model.stp_tau_rec}s)"
              f"  j_stp={config.j_stp}", flush=True)
    else:
        print(f"{p}  dynamics: alpha={alpha:.4f}  alpha_rec={alpha_rec:.4f}  dt={DT:.4f}"
              f"  tau={config.tau}  tau_rec_frac={config.tau_rec_frac}", flush=True)
    print(f"{p}  noise:  input_sigma={noise:.4f} (×{config.noise})"
          f"  model_sigma={model_noise_sigma:.4f} (×{config.model_noise})", flush=True)
    print(f"{p}  optim:  lr={config.learning_rate}  wd={config.weight_decay}"
          f"  batch={config.batch_size}  n_batch={config.n_batch}  clip={config.grad_clip_norm}", flush=True)
    print(f"{p}  epochs: dpa={config.epochs_dpa}  gng={config.epochs_gng}"
          f"  dual={config.epochs_dual}  loss={config.dual_loss}", flush=True)
    print(f"{p}  shapes: m{list(model.m.shape)}  n{list(model.n.shape)}"
          f"  Wi{list(model.wi.weight.shape)}", flush=True)

    if config.init_style == "structured" and config.model_type not in ("ei", "eistp"):
        # rank-3 role split: κ0 sample-memory (mem), κ1 gng-memory (gng), κ2 action/decision (out).
        rank3 = config.rank >= 3
        init_dpa_internal_readout_prepost(
            model, mem=0, out=(2 if rank3 else 1),
            gng=(1 if rank3 else None), gng_lambda=config.gng_lambda,
            memory_lambda=config.memory_lambda,
            decision_lambda=config.decision_lambda,
            target_mn_corr=config.target_mn_corr,
            target_out_mn_corr=config.target_out_mn_corr,
            sample_scale=config.sample_scale,
            test_scale=config.test_scale,
            decision_readout_mean=config.decision_readout_mean,
            mix_strength=config.mix_strength,
            noise_scale_mn=1.0, noise_scale_in=1.0,
            rwd_input_scale=config.rwd_input_scale,
            seed=config.seed, verbose=True,
        )
    # "random" → keep default LowRankModel init
    _log_params("init")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    wb_run = None
    if wandb_project is not None:
        try:
            import wandb
            wb_run = wandb.init(
                project=wandb_project,
                name=config.run_id,
                config=dataclasses.asdict(config),
                reinit=True,
            )
        except Exception as e:
            print(f"[{rid}] W&B init failed ({e}); continuing without logging.", flush=True)

    # UNIFIED-ONLY (2026-08-12): the legacy loss paths (multi = MaskedMultiTargetLoss,
    # separated = MaskedMultiTargetDualLoss, threshold = ThresholdLoss, MaskedGNGLoss for the
    # GNG stage) were removed from sweep.py — UnifiedLoss covers all three stages. The classes
    # remain in src/train.py for old scripts; historical arm dicts in make_configs still
    # construct but raise here if actually launched.
    if config.dual_loss != "unified":
        raise ValueError(f"[{rid}] dual_loss={config.dual_loss!r}: legacy losses removed from "
                         f"sweep.py (2026-08-12) — only 'unified' is runnable")
    # UnifiedLoss params (used at all three stage sites).
    # _uth (go_hinge_thresh) = GNG-side hinge threshold (pre-cue holds + rwd response hinges);
    # 0 = sign-based go/nogo. _uth_pair (dpa_hinge_thresh) = pair + memory-hold threshold —
    # kept separate so sign-based gng does NOT silently relax the pairing/κ₀ amplitude supervision.
    _uth      = config.go_hinge_thresh  if config.go_hinge_thresh  is not None else 1.0
    _uth_pair = config.dpa_hinge_thresh if config.dpa_hinge_thresh is not None else 1.0
    # NOGO/neg-side gng hinge magnitude from the SIGNED nogo_hinge_thresh field: −1 (legacy) → 1.0
    # (matches the old symmetric behaviour), 0 → 0 (nogo ≤ 0, free below — asymmetric sign design).
    _uth_neg  = -config.nogo_hinge_thresh
    _uw  = dict(gng_weight=config.gng_weight, pair_weight=config.dpa_weight,
                gng_decay_weight=config.gng_decay_weight,
                pair_decay_weight=config.pair_decay_weight,
                rwd_go_weight=config.rwd_go_weight,
                rwd_nogo_weight=config.rwd_nogo_weight,
                rwd_nogo_onesided=config.rwd_nogo_onesided,
                rwd_nogo_l1=config.rwd_nogo_l1,
                rwd_keep_go_hinge=config.rwd_keep_go_hinge,
                decay_onesided=config.decay_onesided,
                mem_weight=config.aux_weight, bl_weight=config.bl_weight)
    # The one-sided nogo scoring (rwd_nogo_onesided) drops the go +1 hinge in the response window; that
    # is fine in DUAL (frees the nogo value to settle low) but MUST NOT apply in the GNG stage, where it
    # would remove the go supervision and the go/nogo working memory never forms (go→0). So GNG always
    # trains the go/nogo memory TWO-SIDED; one-sided is a Dual-only relaxation.
    _uw_gng = ({**_uw} if config.gng_rwd_onesided
               else {**_uw, "rwd_nogo_onesided": False})
    _half_steps = int(round(0.5 / gng_timing.dt))   # response window length in steps
    # ONE value-based loss at all three stages (targets carry the semantics). Baseline is its
    # own pinned term; the decision channel splits into separate gng/pair terms at test onset.
    # dpa_nolick_weight: one-sided don't-lick over the DPA-stage delay (sample-off → test-onset).
    # There is no go/nogo in DPA, so every row gets it and no trial-scoping is needed.
    dpa_criterion = UnifiedLoss(dpa_timing, thresh=_uth_pair, gng_thresh=_uth, gng_neg_thresh=_uth_neg, hinge_shape=config.hinge_shape,
                                pair_start=int(dpa_timing.n_stim_on[1]),
                                nolick_weight=config.dpa_nolick_weight,
                                nolick_window=(int(dpa_timing.n_stim_off[0]), int(dpa_timing.n_stim_on[1])),
                                nolick_thresh=config.nolick_thresh, **_uw)
    # rwd_window follows response_in_cue: targets sit IN-cue (co-half:co) when set, else the
    # legacy post-cue window. (Fixed 2026-08-10: the window was never shifted, so in-cue nogo
    # 0-targets fell into the gng group's two-sided PIN and rwd_nogo_weight was inert.)
    _co_g = int(gng_timing.n_stim_off[1])
    _ric_g = config.response_in_cue and not config.gng_rwd_after_cue
    _rw_g = (_co_g - _half_steps, _co_g) if _ric_g else (_co_g, _co_g + _half_steps)
    # nolick_late_delay: restrict the nolick term to the after-cue span — the late-delay
    # DON'T-LICK imposition. Dual: (cue-off, test-onset). GNG stage (no test): (cue-off, trial
    # end) — needed once nogo_target=None drops the nogo response target, so "nogo = don't lick
    # after the cue" stays supervised in the GNG stage too. Without the flag the GNG stage keeps
    # no nolick (legacy Dual-only behaviour).
    _nlw_d = ((int(dual_timing.n_stim_off[2]), int(dual_timing.n_stim_on[3]))
              if config.nolick_late_delay else None)
    _nlw_g = ((int(gng_timing.n_stim_off[1]), int(gng_timing.n_steps))
              if config.nolick_late_delay else None)
    # nolick_full_delay: the whole delay (sample-off → test-onset) on the 'none' rows only. Those
    # rows are identified inside the loss by having no finite decision target in the go/nogo span,
    # which only distinguishes them from go/nogo rows if those rows carry the pre-cue hold.
    if config.nolick_full_delay and not config.dual_gng_memory:
        raise ValueError("nolick_full_delay needs dual_gng_memory=True: without the pre-cue hold, "
                         "go/nogo rows have no finite target in the gng span either, so they would "
                         "be treated as 'none' and the full-delay hinge would fight the go rule.")
    if config.nolick_nogo_in_cue and not (config.dual_gng_memory and config.nolick_weight > 0):
        raise ValueError("nolick_nogo_in_cue needs dual_gng_memory=True (nogo rows are identified by their "
                         "hold target) and nolick_weight > 0 (else the term is silently inert).")
    if config.nolick_full_delay and config.target_rank != 2:
        raise ValueError("nolick_full_delay: DPA-trial detection reads the DECISION channel, and the "
                         "pre-cue hold lands there only for target_rank=2 (rank-3 holds live on ch1, "
                         "decision on ch2 — go/nogo rows would be misread as DPA trials).")
    if config.rwd_go_thresh is not None and not config.gng_response:
        raise ValueError("rwd_go_thresh needs gng_response=True: without a response window the go "
                         "response hinge never fires (silently inert).")
    if config.gng_hold_ceiling is not None:
        if config.gng_hold_ceiling < _uth:
            raise ValueError(f"gng_hold_ceiling={config.gng_hold_ceiling} < go hold threshold {_uth}: "
                             "the ceiling and the hold hinge contradict (no satisfiable go state).")
        if config.rwd_go_thresh is None or config.rwd_go_thresh <= config.gng_hold_ceiling:
            raise ValueError(f"gng_hold_ceiling={config.gng_hold_ceiling} needs rwd_go_thresh ABOVE it "
                             f"(got {config.rwd_go_thresh}): otherwise the held rule still satisfies the "
                             "lick and the cue has no job — the cuego loophole (§27g).")
    if config.dpa_nolick_weight > 0 and not config.dpa_prelick_free:
        raise ValueError("dpa_nolick_weight needs dpa_prelick_free=True: the legacy two-sided 0-pin "
                         "fills the DPA delay with finite targets, leaving the one-sided term no free "
                         "steps — it would be silently inert (the rwd_window trap, 2026-08-10).")
    _nlf_d = ((int(dual_timing.n_stim_off[0]), int(dual_timing.n_stim_on[3]))
              if config.nolick_full_delay else None)
    _nlg_d = (int(dual_timing.n_stim_on[1]), int(dual_timing.n_stim_off[2]))
    # nolick_nogo_in_cue: nogo rows from cue ONSET. Dual: (cue-on, test-on); GNG: (cue-on, end).
    # The GNG stage also needs the gng span for the row classification (go/nogo stim-on → cue-off).
    _nlg_g = (int(gng_timing.n_stim_on[0]), int(gng_timing.n_stim_off[1]))
    _nln_d = ((int(dual_timing.n_stim_on[2]), int(dual_timing.n_stim_on[3])) if config.nolick_nogo_in_cue else None)
    _nln_g = ((int(gng_timing.n_stim_on[1]), int(gng_timing.n_steps)) if config.nolick_nogo_in_cue else None)
    gng_criterion = UnifiedLoss(gng_timing, thresh=_uth_pair, gng_thresh=_uth, gng_neg_thresh=_uth_neg, hinge_shape=config.hinge_shape, pair_start=None,
                                rwd_window=_rw_g,
                                nolick_weight=(config.nolick_weight if (config.nolick_late_delay or config.nolick_nogo_in_cue) else 0.0),
                                nolick_window=_nlw_g, nolick_gng_span=_nlg_g, nolick_nogo_window=_nln_g, nolick_thresh=config.nolick_thresh, rwd_go_thresh=config.rwd_go_thresh, hold_pin=config.gng_hold_pin, hold_ceiling=config.gng_hold_ceiling, **_uw_gng)
    print(f"[{rid}]  loss=unified (ALL stages): ±1→one-sided hinge(gng th={_uth}, pair/mem th={_uth_pair}),"
          f" 0→pin, NaN→free  [bl | gng | pair split @ test-on]"
          f"{f'  DPA-stage nolick w={config.dpa_nolick_weight} over the delay' if config.dpa_nolick_weight else ''}",
          flush=True)
    losses    = {}
    _global_step = [0]   # mutable so the nested helper can increment it

    def _wb_log_losses(stage: str, train_l: list, val_l: list):
        if wb_run is None:
            return
        for tl, vl in zip(train_l, val_l):
            wb_run.log({f"{stage}/train_loss": tl, f"{stage}/val_loss": vl},
                       step=_global_step[0])
            _global_step[0] += 1

    def _opt_and_sched():
        if config.optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            opt = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-5) \
                if config.use_scheduler else None
        return opt, sched

    def _eval(label):
        model.noise = 0.0
        dpa = _dpa_accuracy_by_type(model, dpa_timing, config.input_size, noise=noise, device=device,
                                    target_rank=config.target_rank, input_scale=config.input_scale, attention_input=config.attention_input, attention_gated=config.attention_gated, attention_scale=config.attention_scale,
                                    windowed_targets=config.windowed_targets, decay_to_zero=config.decay_to_zero, response_in_cue=config.response_in_cue)
        gng = _gng_accuracy_by_type(model, gng_timing, config.input_size, noise=noise, device=device,
                                    target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                    cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                    go_on_rwd_input=config.go_on_rwd_input, input_scale=config.input_scale, attention_input=config.attention_input, attention_gated=config.attention_gated, attention_scale=config.attention_scale, response_in_cue=config.response_in_cue,
                                    go_hinge_thresh=config.go_hinge_thresh, nogo_hinge_thresh=config.nogo_hinge_thresh,
                                    gng_rwd_after_cue=config.gng_rwd_after_cue,
                                    attention_through_cue=config.attention_through_cue)
        print(f"[{rid}]   {label}: "
              f"dpa={dpa['overall']:.3f} (pair={dpa['pair']:.3f} unpair={dpa['unpair']:.3f})  "
              f"gng={gng['overall']:.3f} (go={gng['go']:.3f} nogo={gng['nogo']:.3f})", flush=True)
        model.noise = 0.0
        return {"dpa": dpa["overall"], "gng": gng["overall"]}

    # ------------------------------------------------------------------
    # Stage 1 — DPA
    # ------------------------------------------------------------------
    if config.dpa_ckpt is not None:
        print(f"[{rid}]  DPA: loading checkpoint from {config.dpa_ckpt}", flush=True)
        sd = torch.load(config.dpa_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if [k for k in missing if k != "gain"] or unexpected:
            raise RuntimeError(f"DPA ckpt mismatch: missing={missing}, unexpected={unexpected}")
        losses["dpa"] = {}
        train_l, val_l, t0 = [], [], time.time()
        torch.save(model.state_dict(), os.path.join(models_dir, f"dpa_{rid}.pth"))
    else:
        dpa_freeze_input = list(range(config.input_size)) if "dpa" in config.freeze_input_stages else []
        if config.freeze_gng_input_during_dpa:
            gng_dims = list(range(4, config.input_size - 1))  # go/nogo/cue; excludes reward (last)
            dpa_freeze_input = sorted(set(dpa_freeze_input) | set(gng_dims))
        if config.freeze_attention_input and config.attention_input:
            dpa_freeze_input = sorted(set(dpa_freeze_input) | {config.input_size - 1})
        # rank-3: freeze κ₁ (the init'd gng-rule mode) during DPA — DPA has no go/nogo, so keep the
        # self-sustaining mode intact until GNG trains it (else DPA's free training drifts its m,n).
        dpa_freeze_cols = [1] if config.rank >= 3 else None
        _stage_header("DPA", config.epochs_dpa, dpa_freeze_input, dpa_freeze_cols or [])
        t0 = time.time()
        X, y   = generate_dpa_trials(config.n_batch, dpa_timing, config.input_size, noise=noise, target_rank=config.target_rank, input_scale=config.input_scale, attention_input=config.attention_input, attention_gated=config.attention_gated, attention_scale=config.attention_scale, windowed_targets=config.windowed_targets, decay_to_zero=config.decay_to_zero, decay_onesided=config.decay_onesided, response_in_cue=config.response_in_cue, prelick_free=config.dpa_prelick_free, hold_window=config.dpa_hold_window)
        print(f"[{rid}]  data: {list(X.shape)} → {list(y.shape)}", flush=True)
        tl, vl     = train_val_split(X.to(device), y.to(device), config.batch_size)
        opt, sched = _opt_and_sched()
        model.noise = model_noise_sigma
        dpa_regularizer = _kappa1_regularizer(config, model)
        if dpa_regularizer is not None:
            print(f"[{rid}]  decision-subcriticality reg (DPA): w={config.kappa1_reg_weight}·relu(g·λ{config.rank-1}−1)²", flush=True)
        if config.rwd and config.rwd_align_weight > 0.0:
            _w = config.rwd_align_weight
            _k1reg = dpa_regularizer
            def dpa_regularizer(m, _w=_w, _k1reg=_k1reg):
                wi_rwd = m.wi.weight[:, -1]
                n1     = m.n[:, 1]
                cos    = torch.dot(wi_rwd, n1) / (wi_rwd.norm() * n1.norm()).clamp_min(1e-8)
                align  = _w * (1.0 - cos)
                return align + (_k1reg(m) if _k1reg is not None else 0.0)

        trainer    = Optimization(model, tl, vl, dpa_criterion, opt, sched,
                                  config.grad_clip_norm, num_epochs=config.epochs_dpa,
                                  freeze_input_dims=dpa_freeze_input,
                                  freeze_low_rank_cols=dpa_freeze_cols,
                                  regularizer=dpa_regularizer,
                                  stop_loss=config.stop_loss,
                                  kappa_gain_target=config.kappa_gain_target,
                                  rate_reg=config.rate_reg_weight,
                                  max_val_loss=config.max_val_loss,
                                  verbose=True)
        train_l, val_l, _ = trainer.fit()
        losses["dpa"] = {"train": train_l, "val": val_l}
        torch.save(model.state_dict(), os.path.join(models_dir, f"dpa_{rid}.pth"))

    # Option A: project go input column onto n₁ (decision readout direction) before GNG
    if config.project_go_on_n1:
        with torch.no_grad():
            go_ch = config.input_size - 1 if config.go_on_rwd_input else 4
            n1 = model.n[:, 1]
            n1_hat = n1 / (n1.norm().clamp_min(1e-12))
            w_go = model.wi.weight.data[:, go_ch]
            model.wi.weight.data[:, go_ch] = n1_hat * w_go.norm()
        print(f"[{rid}]  projected go ch={go_ch} onto n₁ unit vector", flush=True)

    # Orthogonalise go+nogo input columns to n₀ (memory direction) before GNG
    if config.project_gng_orth_n0:
        with torch.no_grad():
            go_ch   = config.input_size - 1 if config.go_on_rwd_input else 4
            nogo_ch = 4                      if config.go_on_rwd_input else 5
            n0      = model.n[:, 0]
            n0_hat  = n0 / n0.norm().clamp_min(1e-12)
            for ch in [go_ch, nogo_ch]:
                w      = model.wi.weight.data[:, ch]
                w_orth = w - (w @ n0_hat) * n0_hat          # remove n₀ component
                w_orth = w_orth / w_orth.norm().clamp_min(1e-12) * w.norm()  # preserve norm
                model.wi.weight.data[:, ch] = w_orth
        print(f"[{rid}]  orthogonalised go ch={go_ch}, nogo ch={nogo_ch} to n₀", flush=True)

    acc_after_dpa = _eval("after DPA")
    _stage_summary("DPA", train_l, val_l, acc_after_dpa, t0)
    _log_params("after DPA")

    _wb_log_losses("dpa", train_l, val_l)
    if wb_run is not None:
        wb_run.log({"after_dpa/acc_dpa": acc_after_dpa["dpa"],
                    "after_dpa/acc_gng": acc_after_dpa["gng"]})

    # ------------------------------------------------------------------
    # Stage 2 — GNG  (freeze rank-0 of m,n and DPA input dims; also freeze reward dim if rwd=True)
    # ------------------------------------------------------------------
    if config.gng_ckpt is not None:
        print(f"[{rid}]  GNG: loading checkpoint from {config.gng_ckpt}", flush=True)
        sd = torch.load(config.gng_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if [k for k in missing if k != "gain"] or unexpected:
            raise RuntimeError(f"GNG ckpt mismatch: missing={missing}, unexpected={unexpected}")
        losses["gng"] = {}
        train_l, val_l, t0 = [], [], time.time()
        torch.save(model.state_dict(), os.path.join(models_dir, f"naive_{rid}.pth"))
        acc_after_gng = _eval("after GNG")
        _stage_summary("GNG", train_l, val_l, acc_after_gng, t0)
        _log_params("after GNG")
    else:
        # Default GNG freeze: DPA sample/test dims [0,1,2,3], plus the LAST channel whenever it carries
        # reward OR attention. Attention is a tonic CONTEXT input learned in DPA — freeze it from GNG
        # onward (train in DPA, frozen GNG+Dual, like the DPA dims) so GNG can't rescale/rotate the
        # learned attention bias away. (freeze_attention_input additionally freezes it in DPA too.)
        gng_freeze_input = (list(range(config.input_size)) if "gng" in config.freeze_input_stages
                            else [0, 1, 2, 3]
                                 + ([config.input_size - 1] if (config.rwd or config.attention_input) else []))
        if config.freeze_attention_input and config.attention_input:
            gng_freeze_input = sorted(set(gng_freeze_input) | {config.input_size - 1})
        _stage_header("GNG", config.epochs_gng, gng_freeze_input, [0])
        t0 = time.time()
        X, y   = generate_gng_trials(config.n_batch, gng_timing, config.input_size, noise=noise, target_rank=config.target_rank,
                                      cue_on_go_input=config.cue_on_go_input, cue_scale=config.cue_scale,
                                      nogo_target=config.nogo_target, go_target=config.go_target, go_on_rwd_input=config.go_on_rwd_input,
                                      input_scale=config.input_scale, attention_input=config.attention_input, attention_gated=config.attention_gated, attention_scale=config.attention_scale,
                                      ramping_gng=config.ramping_gng, windowed_targets=config.windowed_targets, decay_to_zero=config.decay_to_zero or config.gng_decay_to_zero, decay_to_end=config.gng_decay_to_zero, gng_response=config.gng_response, decay_onesided=config.decay_onesided, response_in_cue=config.response_in_cue, gng_rwd_after_cue=config.gng_rwd_after_cue, hold_full_delay=config.gng_hold_full_delay, attention_through_cue=config.attention_through_cue)
        print(f"[{rid}]  data: {list(X.shape)} → {list(y.shape)}", flush=True)
        tl, vl     = train_val_split(X.to(device), y.to(device), config.batch_size)
        opt, sched = _opt_and_sched()
        model.noise = model_noise_sigma
        model.rwd   = config.rwd_gng   # optionally disable reward during GNG training
        gng_regularizer = _kappa1_regularizer(config, model)
        if gng_regularizer is not None:
            print(f"[{rid}]  decision-subcriticality reg (GNG): w={config.kappa1_reg_weight}·relu(g·λ{config.rank-1}−1)²", flush=True)
        # gng_decouple_decision: keep the decision readout blind to the sample-memory direction —
        # the leakage that INVERTS the DPA memory during GNG (§26/§27). rank-0 is frozen here, so
        # n[:,dec] is the only party that can build the overlap.
        _orth = (config.rank - 1, 0) if config.gng_decouple_decision else None
        if _orth is not None:
            print(f"[{rid}]  decoupling (GNG): project n[:,{_orth[0]}] ⟂ m[:,0] after each step", flush=True)
        trainer    = Optimization(model, tl, vl, gng_criterion, opt, sched,
                                  config.grad_clip_norm, num_epochs=config.epochs_gng,
                                  freeze_low_rank_cols=[0],
                                  freeze_input_dims=gng_freeze_input,
                                  stop_loss=config.stop_loss,
                                  regularizer=gng_regularizer,
                                  kappa_gain_target=config.kappa_gain_target,
                                  rate_reg=config.rate_reg_weight,
                                  max_val_loss=config.max_val_loss,
                                  orthogonalize_cols=_orth,
                                  verbose=True)
        train_l, val_l, _ = trainer.fit()
        model.rwd = config.rwd         # restore reward for eval and subsequent stages
        losses["gng"] = {"train": train_l, "val": val_l}
        torch.save(model.state_dict(), os.path.join(models_dir, f"naive_{rid}.pth"))
        acc_after_gng = _eval("after GNG")
        _stage_summary("GNG", train_l, val_l, acc_after_gng, t0)
        _log_params("after GNG")

    _wb_log_losses("gng", train_l, val_l)
    if wb_run is not None:
        wb_run.log({"after_gng/acc_dpa": acc_after_gng["dpa"],
                    "after_gng/acc_gng": acc_after_gng["gng"]})

    # rank-3 progressive freeze: the Dual stages lock BOTH memories (κ0 sample + κ1 gng) and train
    # only κ2 (action). rank-2 keeps the old behaviour (optionally freeze rank-0).
    dual_mem_freeze = [0, 1] if config.rank >= 3 else ([0] if config.freeze_rank0_dual else None)

    # ------------------------------------------------------------------
    # Stage 2.5 — Dual-paired (MATCH trials only) → overwrites "naive" (curriculum bridge)
    # ------------------------------------------------------------------
    if config.dual_paired_stage:
        paired_freeze_input = list(range(config.input_size)) if "dual" in config.freeze_input_stages else []
        _stage_header("Dual-paired", config.epochs_dual_paired, paired_freeze_input,
                      dual_mem_freeze or [])
        t0 = time.time()
        Xp, yp, _, _ = generate_dual_trials(config.n_batch, dual_timing, config.input_size, noise=noise,
                                            target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                            cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                            go_target=config.go_target, go_on_rwd_input=config.go_on_rwd_input,
                                            input_scale=config.input_scale, attention_input=config.attention_input, attention_gated=config.attention_gated, attention_scale=config.attention_scale,
                                            paired_only=True, ramping_gng=config.ramping_gng, windowed_targets=config.windowed_targets, decay_to_zero=config.decay_to_zero, gng_response=config.gng_response, gng_memory=config.dual_gng_memory, decay_onesided=config.decay_onesided, response_in_cue=config.response_in_cue, gng_rwd_after_cue=config.gng_rwd_after_cue, hold_full_delay=config.gng_hold_full_delay)
        print(f"[{rid}]  data(paired): {list(Xp.shape)} → {list(yp.shape)}", flush=True)
        tlp, vlp     = train_val_split(Xp.to(device), yp.to(device), config.batch_size)
        optp, schedp = _opt_and_sched()
        model.noise  = model_noise_sigma
        _co_d = int(dual_timing.n_stim_off[2])
        _ric_d = config.response_in_cue and not config.gng_rwd_after_cue
        _rw_d = (_co_d - _half_steps, _co_d) if _ric_d else (_co_d, _co_d + _half_steps)
        paired_criterion = UnifiedLoss(dual_timing, thresh=_uth_pair, gng_thresh=_uth, gng_neg_thresh=_uth_neg, hinge_shape=config.hinge_shape,
                                       pair_start=int(dual_timing.n_stim_on[3]),
                                       rwd_window=_rw_d,
                                       nolick_weight=config.nolick_weight,
                                       nolick_window=_nlw_d, nolick_full_window=_nlf_d,
                                       nolick_gng_span=_nlg_d, nolick_nogo_window=_nln_d, nolick_thresh=config.nolick_thresh, rwd_go_thresh=config.rwd_go_thresh, hold_pin=config.gng_hold_pin, hold_ceiling=config.gng_hold_ceiling, **_uw)
        trainer = Optimization(model, tlp, vlp, paired_criterion, optp, schedp,
                               config.grad_clip_norm, num_epochs=config.epochs_dual_paired,
                               freeze_low_rank_cols=dual_mem_freeze,
                               freeze_input_dims=paired_freeze_input,
                               stop_loss=config.stop_loss,
                               regularizer=_kappa1_regularizer(config, model),
                               kappa1_clamp=config.kappa1_clamp,
                               kappa_gain_target=config.kappa_gain_target,
                                  rate_reg=config.rate_reg_weight,
                                  max_val_loss=config.max_val_loss,
                               verbose=True)
        tl, vl, _ = trainer.fit()
        losses["dual_paired"] = {"train": tl, "val": vl}
        torch.save(model.state_dict(), os.path.join(models_dir, f"naive_{rid}.pth"))  # naive = Dual-paired
        acc_after_gng = _eval("after Dual-paired")
        _stage_summary("Dual-paired", tl, vl, acc_after_gng, t0)
        _log_params("after Dual-paired")

    # ------------------------------------------------------------------
    # Stage 3 — Dual  (freeze all input dims)
    # ------------------------------------------------------------------
    dual_freeze_input = list(range(config.input_size)) if "dual" in config.freeze_input_stages else []
    _stage_header("Dual", config.epochs_dual, dual_freeze_input, dual_mem_freeze or [])
    t0 = time.time()
    X, y, _, _ = generate_dual_trials(config.n_batch, dual_timing, config.input_size, noise=noise, target_rank=config.target_rank,
                                       cue_on_go_input=config.cue_on_go_input, cue_scale=config.cue_scale,
                                       nogo_target=config.nogo_target, go_target=config.go_target, go_on_rwd_input=config.go_on_rwd_input,
                                       input_scale=config.input_scale, attention_input=config.attention_input, attention_gated=config.attention_gated, attention_scale=config.attention_scale,
                                       ramping_gng=config.ramping_gng, windowed_targets=config.windowed_targets, decay_to_zero=config.decay_to_zero, gng_response=config.gng_response, gng_memory=config.dual_gng_memory, decay_onesided=config.decay_onesided, response_in_cue=config.response_in_cue, gng_rwd_after_cue=config.gng_rwd_after_cue, hold_full_delay=config.gng_hold_full_delay)
    print(f"[{rid}]  data: {list(X.shape)} → {list(y.shape)}", flush=True)
    tl, vl     = train_val_split(X.to(device), y.to(device), config.batch_size)
    opt, sched = _opt_and_sched()
    model.noise = model_noise_sigma

    _co_d = int(dual_timing.n_stim_off[2])
    _ric_d = config.response_in_cue and not config.gng_rwd_after_cue
    _rw_d = (_co_d - _half_steps, _co_d) if _ric_d else (_co_d, _co_d + _half_steps)
    dual_criterion = UnifiedLoss(dual_timing, thresh=_uth_pair, gng_thresh=_uth, gng_neg_thresh=_uth_neg, hinge_shape=config.hinge_shape,
                                 pair_start=int(dual_timing.n_stim_on[3]),
                                 rwd_window=_rw_d,
                                 nolick_weight=config.nolick_weight,
                                 nolick_window=_nlw_d, nolick_full_window=_nlf_d,
                                 nolick_gng_span=_nlg_d, nolick_nogo_window=_nln_d, nolick_thresh=config.nolick_thresh, rwd_go_thresh=config.rwd_go_thresh, hold_pin=config.gng_hold_pin, hold_ceiling=config.gng_hold_ceiling, **_uw)
    print(f"[{rid}]  loss=unified  gng_w={config.gng_weight}  pair_w={config.dpa_weight}"
          f"  nolick_w={config.nolick_weight}"
          f"{f'  nolick_window={_nlw_d} (late delay)' if _nlw_d else ''}"
          f"{f'  nolick_FULL={_nlf_d} on none-rows (gng span {_nlg_d})' if _nlf_d else ''}"
          f"{f'  nolick_NOGO={_nln_d} on nogo-rows from cue-on' if _nln_d else ''}"
          f"{f'  nolick_thresh=−{config.nolick_thresh}' if config.nolick_thresh else ''}", flush=True)

    dual_freeze_rank0 = dual_mem_freeze

    dual_regularizer = _kappa1_regularizer(config, model)
    if dual_regularizer is not None:
        print(f"[{rid}]  decision-subcriticality reg (Dual): w={config.kappa1_reg_weight}·relu(g·λ{config.rank-1}−1)²", flush=True)

    trainer    = Optimization(model, tl, vl, dual_criterion, opt, sched,
                              config.grad_clip_norm, num_epochs=config.epochs_dual,
                              freeze_low_rank_cols=dual_freeze_rank0,
                              freeze_input_dims=dual_freeze_input,
                              stop_loss=config.stop_loss,
                              regularizer=dual_regularizer,
                              kappa1_clamp=config.kappa1_clamp,
                              kappa_gain_target=config.kappa_gain_target,
                                  rate_reg=config.rate_reg_weight,
                                  max_val_loss=config.max_val_loss,
                              verbose=True)
    if config.kappa1_clamp is not None:
        print(f"[{rid}]  κ₁ hard clamp: g·λ₁ ≤ {config.kappa1_clamp} after each Dual step", flush=True)

    train_l, val_l, _ = trainer.fit()
    dual_loss_components = dict(dual_criterion.last_components)
    if dual_loss_components:
        print(f"[{rid}]  loss components: "
              f"{ {k: round(v, 4) for k, v in dual_loss_components.items()} }", flush=True)
    losses["dual"] = {"train": train_l, "val": val_l}
    torch.save(model.state_dict(), os.path.join(models_dir, f"expert_{rid}.pth"))
    acc_after_dual = _eval("after Dual")
    dual_dpa, dual_gng, dual_go, dual_nogo = _dual_accuracy(model, dual_timing, config.input_size, noise=noise, device=device,
                                         target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                         cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                         go_on_rwd_input=config.go_on_rwd_input, input_scale=config.input_scale, attention_input=config.attention_input, attention_gated=config.attention_gated, attention_scale=config.attention_scale,
                                         go_target=config.go_target, response_in_cue=config.response_in_cue,
                                         go_hinge_thresh=config.go_hinge_thresh, nogo_hinge_thresh=config.nogo_hinge_thresh,
                                         gng_rwd_after_cue=config.gng_rwd_after_cue)
    _stage_summary("Dual", train_l, val_l, acc_after_dual, t0)
    _log_params("after Dual")

    acc = {
        "after_dpa":  acc_after_dpa,
        "after_gng":  acc_after_gng,
        "after_dual": {**acc_after_dual, "dual_dpa": dual_dpa, "dual_gng": dual_gng,
                        "dual_go": dual_go, "dual_nogo": dual_nogo},
    }

    _wb_log_losses("dual", train_l, val_l)
    if wb_run is not None:
        wb_run.log({
            "after_dual/acc_dpa":  acc_after_dual["dpa"],
            "after_dual/acc_gng":  acc_after_dual["gng"],
            "after_dual/dual_dpa": dual_dpa,
            "after_dual/dual_gng": dual_gng,
        })
        if dual_loss_components:
            wb_run.log({f"dual_loss/{k}": v for k, v in dual_loss_components.items()})
        wb_run.summary.update({
            "final/dual_dpa": dual_dpa,
            "final/dual_gng": dual_gng,
            "final/acc_dpa":  acc_after_dual["dpa"],
            "final/acc_gng":  acc_after_dual["gng"],
        })
        wb_run.finish()

    t_total = time.time() - t_run
    p = f"[{rid}]"
    print(f"{p} {'═'*60}", flush=True)
    print(f"{p}  RUN COMPLETE  {time.strftime('%Y-%m-%d %H:%M:%S')}  total={t_total:.1f}s", flush=True)
    print(f"{p}  ACCURACY SUMMARY", flush=True)
    print(f"{p}    after DPA : dpa={acc_after_dpa['dpa']:.3f}  gng={acc_after_dpa['gng']:.3f}", flush=True)
    print(f"{p}    after GNG : dpa={acc_after_gng['dpa']:.3f}  gng={acc_after_gng['gng']:.3f}", flush=True)
    print(f"{p}    dual go/nogo: go={dual_go:.3f}  nogo={dual_nogo:.3f}  (gng={dual_gng:.3f})", flush=True)
    print(f"{p}    after Dual: dpa={acc_after_dual['dpa']:.3f}  gng={acc_after_dual['gng']:.3f}"
          f"  dual_dpa={dual_dpa:.3f}  dual_gng={dual_gng:.3f}", flush=True)
    print(f"{p} {'═'*60}", flush=True)

    return {
        "run_id": rid,
        "status": "ok",
        "config": dataclasses.asdict(config),
        "accuracy": acc,
        "final_train_loss": {stage: v["train"][-1] for stage, v in losses.items() if v.get("train")},
        "final_val_loss":   {stage: v["val"][-1]   for stage, v in losses.items() if v.get("val")},
        "loss_curves": {stage: {"train": v["train"], "val": v["val"]}
                        for stage, v in losses.items() if v.get("train")},
        "dual_loss_components": dual_loss_components,
    }


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker(worker_id: int, n_gpus: int, job_queue: mp.Queue, result_queue: mp.Queue,
            out_dir: str, wandb_project: str | None = None):
    device = f"cuda:{worker_id % n_gpus}" if torch.cuda.is_available() else "cpu"
    while True:
        config = job_queue.get()
        if config is None:      # sentinel → done
            break

        run_dir  = os.path.join(out_dir, config.run_id)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")

        with open(log_path, "w", buffering=1) as log_f:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = log_f
            try:
                result = run_single(config, device, models_dir=run_dir,
                                    wandb_project=wandb_project)
            except Exception:
                tb = traceback.format_exc()
                result = {
                    "run_id": config.run_id,
                    "status": "error",
                    "config": dataclasses.asdict(config),
                    "traceback": tb,
                }
                print(f"[{config.run_id}] ERROR:\n{tb}", flush=True)
            finally:
                sys.stdout = old_out
                sys.stderr = old_err

        result_queue.put(result)


# ---------------------------------------------------------------------------
# Sweep definition  ← edit this to change what gets run
# ---------------------------------------------------------------------------

def make_configs(out_dir: str, nonlinearity: str = "relu", cue_on_go_input: bool = True,
                 nogo_target: float | None = None, hinge_squared: bool | None = None,
                 lr_additive: bool | None = None, dense_cee: bool | None = None,
                 hinge_gng: bool | None = None) -> list[RunConfig]:
    """
    Return the list of runs to execute.  Edit freely.

    Tips
    ----
    - run_id must be unique across all configs (it names the checkpoint files).
    - NAME ARMS READABLY (Leon 2026-09-10). The arm tag lands in every run_id, checkpoint filename,
      log line and results.jsonl row — so use the same tokens as the gallery title, short form:
      `<phi>_<init>_<what this arm adds>`, e.g. `pin_cue_nolick`, `sub_cue`, `relu_sub`. NOT
      initialisms like k1zcnl / sclnl / w5rl2. Write the gallery title to
      results/dual/<sweep>/TITLE at launch; publish with scratchpad/publish_gallery.sh.
    - Add / remove loops to vary more or fewer axes.
    - Use dataclasses.replace(base_cfg, seed=s, ...) to share defaults cleanly.
    """
    configs = []

    # --- Vanilla rank-2: TWO isolated LOW memory wells (kill ring + lower) ----
    # Target = two disconnected A/B memory wells, both at κ₂<0 (no-lick), no 270° arc.
    # Two orthogonal ingredients:
    #   (1) ISOLATE — hold the decision self-gain g·λ₁ near critical so there is NO
    #       autonomous decision bistability → the 4-well/270°-U ring collapses to the
    #       two memory wells.  Lever: reduced decision_lambda at init (starts subcritical)
    #       + kappa1_reg_weight penalising g·λ₁>1 during Dual (SWEPT here).
    #   (2) LOWER — directional break pushes those two wells to κ₂<0.  Fixed lever:
    #       tanh_asym (γ=0.3, saturating→spiral-free) + one-sided no-lick hinge (nolick).
    # See docs/ring_lowerplane_log.md, theory_landscape.md §4/§8, scratchpad test_subcritical.py.
    shared = dict(
        model_type="lowrank",
        # rank / target_rank / hidden_size set PER-ARM below
        # gain / memory_lambda / decision_lambda are set PER-ARM below (gain scan).
        init_style="structured",
        nl_gamma=0.0,                  # nonlinearity set PER-ARM below
        attention_input=True,          # tonic attention bias (breaks κ-field odd symmetry, replaces tanh_asym)
        # nolick_weight set PER-ARM below
        hinge_gng=True,                # hinges for ALL stages (DPA ThresholdLoss + GNG/Dual asymmetric one-sided)
        rwd_gng=False,                 # no reward-feedback onto the last channel (clean; avoids the rwd/attention collision)
        cue_on_go_input=True,          # cue rides on go channel (attention arm → input_size=7, else 6)
        # cue_scale set PER-ARM below
        input_scale=1.0,
        model_noise=0.0,               # `noise` (input noise) is set PER-ARM below
        nogo_target=0.0,               # one-sided nogo (consistent no-lick philosophy)
        go_target=1.0,
        go_hinge_thresh=1.0,
        dpa_hinge_thresh=1.0,
        dual_loss="unified",
        freeze_input_stages=["dual"],
        freeze_rank0_dual=False,
        optimizer="adam",              # no weight decay
        learning_rate=0.01,
        use_scheduler=False,           # FIXED lr (these nets learn better with constant lr; no scheduler)
        stop_loss=0.1,
        batch_size=64, n_batch=516,
        grad_clip_norm=None,
        epochs_dpa=100, epochs_gng=100, epochs_dual=100,
        out_dir=out_dir,
    )
    if nogo_target is not None:        # CLI override if desired
        shared["nogo_target"] = nogo_target
    if hinge_gng is not None:          # CLI override: False = uncorrected two-sided MSE-to-±1 holds
        shared["hinge_gng"] = hinge_gng

    # SUBCRITICAL vs SUPERCRITICAL memory mode (rank-2), same everything else. Tests the baseline
    # hypothesis: with g·λ0<1 the pre-sample baseline κ=0 is a stable FP (flat, no drift); with g·λ0>1
    # it's an unstable saddle and the net drifts into a well before any stimulus. Additive attention
    # (last channel, 0 at baseline / 1 after stim) gates the bistability. Launch into SEPARATE dirs with
    # --run_filter _sub / _sup.
    base = dict(rank=2, target_rank=2, tau=0.30, hidden_size=1024, gain=1.0, noise=0.25,
                nonlinearity="tanh", nolick_weight=0.5, decision_lambda=0.5,
                kappa1_reg_weight=0.0, nogo_hinge_thresh=-1.0, cue_scale=2.0)

    # EMERGENT recipe: the no-lick memory wells must EMERGE from the asymmetric-cue nogo-error
    # pressure alone — NO explicit κ₁ penalty. So nolick_weight=0 AND nogo_push_memory=False
    # (gentle nogo κ₁≤0; the well location is left free). attention ON = the symmetry break (not a
    # directional κ₁ penalty). memory_lambda=0.8 (subcritical), 100/100/100. See ring_lowerplane_log §16.
    emergent = {**base, "nolick_weight": 0.0}
    # UNIFIED-LOSS feature-isolation ladder (2026-07-31). ONE value-based loss at all 3 stages
    # (dual_loss="unified"); windowed targets; all weights 1; NO reg. Three arms differ only in the
    # response window (gng_response) and the decay-to-0 targets (decay_to_zero):
    #   _base  : gng_response=F, decay=F  — pre-cue hold + pairing only (minimal)
    #   _rwd   : gng_response=T, decay=F  — + the 0.5 s response window (rwd_go / rwd_nogo terms)
    #   _decay : gng_response=F, decay=T  — + decay-to-0 (the lower-ring lever), NO rwd
    # Feature isolation: rwd effect = _rwd vs _base, decay effect = _decay vs _base. Launch each arm
    # into its OWN dir with --run_filter _base / _rwd / _decay.
    shared_win = {**shared, "freeze_rank0_dual": True, "epochs_dual": 300, "dual_loss": "unified"}
    arms = [("base", False, False), ("rwd", True, False), ("decay", False, True)]
    for tag, rwd, decay in arms:
        for seed in range(4):
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed, memory_lambda=0.8,
                                     nogo_push_memory=False, ramping_gng=True,
                                     windowed_targets=True, decay_to_zero=decay,
                                     gng_response=rwd,
                                     **emergent, **shared_win))

    # NO-GNG-MEMORY-in-Dual ladder (2026-07-31): dual_gng_memory=False → the go/nogo pre-cue hold is
    # NOT re-supervised in Dual (κ₁ go/nogo memory must ride the GNG-learned structure; freeze_rank0
    # only protects κ₀). All unified, all weights 1, no reg, decay OFF. Vary the response window:
    #   _nmnr  : no rwd              (gng_response=F) — go/nogo fully emergent in Dual (only pairing on κ₁)
    #   _nmrwd : rwd, pin nogo→0     (gng_response=T, rwd_nogo_onesided=F)
    #   _nm1s  : rwd, nogo lick-only (gng_response=T, rwd_nogo_onesided=T)
    # Launch each into its own dir with --run_filter _nmnr / _nmrwd / _nm1s.
    nm_arms = [("nmnr", False, False), ("nmrwd", True, False), ("nm1s", True, True)]
    for tag, rwd, onesided in nm_arms:
        for seed in range(4):
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed, memory_lambda=0.8,
                                     nogo_push_memory=False, ramping_gng=True,
                                     windowed_targets=True, decay_to_zero=False,
                                     dual_gng_memory=False,
                                     gng_response=rwd, rwd_nogo_onesided=onesided,
                                     **emergent, **shared_win))

    # ★ First HONEST well-lowering attempt (2026-07-31): the clean nmrwd base (no pre-cue memory, pin
    # response, 4/4) + a one-sided no-lick penalty `nolick_weight·relu(κ₁)²` over the FREE decision
    # windows (Dual stages; sample+baseline excluded). Penalise lick only, κ₁<0 free → lowering must
    # EMERGE (no painted value), vs the decay artifact (§17f) and the reg re-routing (§17b). --run_filter _nlk.
    emergent_nolick = {**emergent, "nolick_weight": 0.5}
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_nlk", seed=seed, memory_lambda=0.8,
                                 nogo_push_memory=False, ramping_gng=True,
                                 windowed_targets=True, decay_to_zero=False,
                                 dual_gng_memory=False,
                                 gng_response=True, rwd_nogo_onesided=False,
                                 **emergent_nolick, **shared_win))

    # ★ UNFREEZE κ₀ in Dual (2026-07-31): freeze_rank0_dual=False so the memory mode m0,n0 is trainable
    # during Dual → the lowering pressure can finally RELOCATE the well (its κ₁ is set by the frozen-
    # until-now memory coupling). Safe now that the ≤−1 tail bug is gone; the pairing (must read the
    # sample 6 s later) is the implicit memory-retention pressure. Completes the freeze×pressure 2×2 vs
    # the frozen controls (nmrwd done, nlk running). nmrwd base. --run_filter _ufnr / _ufnl.
    shared_unfrozen = {**shared_win, "freeze_rank0_dual": False}
    for seed in range(4):   # unfreeze, NO nolick — does memory survive on the pairing alone?
        configs.append(RunConfig(run_id=f"s{seed}_ufnr", seed=seed, memory_lambda=0.8,
                                 nogo_push_memory=False, ramping_gng=True,
                                 windowed_targets=True, decay_to_zero=False,
                                 dual_gng_memory=False,
                                 gng_response=True, rwd_nogo_onesided=False,
                                 **emergent, **shared_unfrozen))
    for seed in range(4):   # unfreeze + nolick — the lowering candidate (DOF × directional pressure)
        configs.append(RunConfig(run_id=f"s{seed}_ufnl", seed=seed, memory_lambda=0.8,
                                 nogo_push_memory=False, ramping_gng=True,
                                 windowed_targets=True, decay_to_zero=False,
                                 dual_gng_memory=False,
                                 gng_response=True, rwd_nogo_onesided=False,
                                 **emergent_nolick, **shared_unfrozen))

    # ★ RANDOM INIT (2026-07-31, 8 seeds): the clean unfrozen/nmrwd base + delay-only attention_gated,
    # but init_style="random" (NO hand-installed structured eigenmodes) — tests whether the recipe /
    # geometry depend on the structured init. Unified, no nolick, no lever. --run_filter _rnd.
    shared_rnd = {**shared_unfrozen, "init_style": "random"}
    for seed in range(8):
        configs.append(RunConfig(run_id=f"s{seed}_rnd", seed=seed, memory_lambda=0.8,
                                 nogo_push_memory=False, ramping_gng=True,
                                 windowed_targets=True, decay_to_zero=False,
                                 dual_gng_memory=False,
                                 gng_response=True, rwd_nogo_onesided=False,
                                 attention_gated=True,
                                 **emergent, **shared_rnd))

    # ★ NOGO-PIN sweep (2026-08-03): the NeuroFlame convergence. Our rwd nogo term IS point-2
    # (go +1 hinge, nogo pin-to-0); NeuroFlame runs that pin at weight 0.1·|overlap| (L1), we ran
    # it at 1.0·κ₁² (L2). A stiff pin clamps κ₁≈0 and holds the autonomous well up (+0.02..+0.09);
    # a weak pin should let the field relax the well BELOW 0. Base = the clean unfrozen +
    # attention_gated + nmrwd recipe (structured init), vary ONLY the nogo pin. 4 arms:
    # Form fixed at L1 (|κ₁|, NeuroFlame's form); vary ONLY the weight → isolates the weight as the
    # lever within the L1 pin:
    #   _pinng10L1: rwd_nogo_weight=1.0, L1 (|κ₁|) — L1 form at our current weight (stiff L1)
    #   _pinng01L1: rwd_nogo_weight=0.1, L1 (|κ₁|) — NeuroFlame weight AND form (exact match)
    # Prediction: autonomous well κ₁ drops below 0 at weight 0.1 (pinng01L1) but not at 1.0 (pinng10L1);
    # task accuracy stays ~1 (go still hard-pinned up). Both share the token "pin" → one launch:
    # --run_filter pin (both into ONE dir for a clean same-init comparison).
    PIN_SEEDS = 4
    shared_pin = {**shared_unfrozen}          # structured init, unfrozen κ₀, epochs 100/100/300
    pin_arms = [("pinng10L1", 1.0, True), ("pinng01L1", 0.1, True)]
    for tag, w, l1 in pin_arms:
        for seed in range(PIN_SEEDS):
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed, memory_lambda=0.8,
                                     nogo_push_memory=False, ramping_gng=True,
                                     windowed_targets=True, decay_to_zero=False,
                                     dual_gng_memory=False,
                                     gng_response=True, rwd_nogo_onesided=False,
                                     rwd_nogo_weight=w, rwd_nogo_l1=l1,
                                     attention_gated=True,
                                     **emergent, **shared_pin))

    # ★ SOFTPLUS capacity probe (2026-08-03): swap tanh→softplus (rectifying, non-saturating; an
    # intrinsic even-curvature symmetry break in φ itself — mechanistically distinct from tanh_asym).
    # Same clean unfrozen + attention_gated + nmrwd base, default pin (w=1.0, L2). Questions: does the
    # even curvature bend the AUTONOMOUS wells below 0, and do two isolated A/B wells survive
    # rectification (softplus is not odd/saturating → the ring/bistability may not hold)?
    # --run_filter _sp.  Plot with --xlim -5 5 (κ runs wider than tanh's ±1.5).
    SP_SEEDS = 4
    emergent_sp = {**emergent, "nonlinearity": "softplus"}   # override tanh; everything else identical
    for seed in range(SP_SEEDS):
        configs.append(RunConfig(run_id=f"s{seed}_sp", seed=seed, memory_lambda=0.8,
                                 nogo_push_memory=False, ramping_gng=True,
                                 windowed_targets=True, decay_to_zero=False,
                                 dual_gng_memory=False,
                                 gng_response=True, rwd_nogo_onesided=False,
                                 attention_gated=True,
                                 **emergent_sp, **shared_unfrozen))

    # ★ RELU capacity probe (2026-08-03): same clean base as the softplus probe but a HARD rectifier
    # (relu). Even more extreme non-saturation than softplus — tests whether hard rectification bends
    # the autonomous wells below 0 and whether isolated A/B wells survive (softplus gave marginal,
    # κ₁≈0 wells; relu is the sharper case). --run_filter _relu.  Plot with --auto_xlim (κ runs wide).
    RELU_SEEDS = 4
    emergent_relu = {**emergent, "nonlinearity": "relu"}     # override tanh; everything else identical
    for seed in range(RELU_SEEDS):
        configs.append(RunConfig(run_id=f"s{seed}_relu", seed=seed, memory_lambda=0.8,
                                 nogo_push_memory=False, ramping_gng=True,
                                 windowed_targets=True, decay_to_zero=False,
                                 dual_gng_memory=False,
                                 gng_response=True, rwd_nogo_onesided=False,
                                 attention_gated=True,
                                 **emergent_relu, **shared_unfrozen))

    # ★★ WELL-PUSH via the ATTENTION symmetry-breaker (2026-08-03) — §19. The odd-φ "forbidden"
    # framing assumes b⊥n; but attention is clamped ON in the autonomous field, so b_attn breaks the
    # odd symmetry (theory §8 route 1). Measured: the well κ₁ = attention-direct term ⟨n₁,φ(g·b_attn)⟩/N
    # (DOWN, −0.13 for tanh) vs memory-modulated even coupling ⟨n₁·φ''(g·b_attn)·m₀²⟩ (UP for tanh).
    # De-risked at fixed weights: scaling attention flips net-even negative at scale≈2.5–3 → tanh wells
    # go BELOW 0 without leaving tanh (clean bounded, no relu spiral). Arms (share token "wp"):
    #   Route A (tanh, sweep attention amplitude): wp_attn1 / wp_attn2 / wp_attn3  (scale 1/2/3)
    #   Route B (rectification comparison):        wp_lifsc (bounded saturating rectifier),
    #                                              wp_reludeep (relu + deep supercritical memory λ₀)
    # --run_filter wp (all into ONE dir for a same-base comparison). Base = the relu/sp arm base.
    WP_SEEDS = 4
    emergent_lifsc = {**emergent, "nonlinearity": "lif_sc"}
    emergent_reludeep = {**emergent, "nonlinearity": "relu"}
    wp_arms = [   # (tag, nonlinearity-dict, attention_scale, memory_lambda)
        ("wp_attn1",    emergent,          1.0, 0.8),
        ("wp_attn2",    emergent,          2.0, 0.8),
        ("wp_attn3",    emergent,          3.0, 0.8),
        ("wp_lifsc",    emergent_lifsc,    1.0, 0.8),
        ("wp_reludeep", emergent_reludeep, 1.0, 3.0),
    ]
    for tag, nl_dict, att_scale, mem_lam in wp_arms:
        for seed in range(WP_SEEDS):
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed, memory_lambda=mem_lam,
                                     nogo_push_memory=False, ramping_gng=True,
                                     windowed_targets=True, decay_to_zero=False,
                                     dual_gng_memory=False,
                                     gng_response=True, rwd_nogo_onesided=False,
                                     attention_gated=True, attention_scale=att_scale,
                                     **nl_dict, **shared_unfrozen))

    # ★★★ LIF (Gaussian CDF) + decision-readout DC — the CLEAN saturating well-push (2026-08-04) §20.
    # φ = ½[1+erf(x/√2)] is non-negative, bounded, SATURATING → compact bistable wells, no relu spiral/
    # blowup. Being non-negative, resting κ₁ = φ(0)·⟨n₁⟩ = ½⟨n₁⟩, so a net-inhibitory decision readout
    # (`decision_readout_mean`<0) seats BOTH A/B memory wells below the no-lick line — cleanly. Toy
    # confirmed: mean well κ₁ tracks ½⟨n₁⟩ (amplified by feedback): ⟨n₁⟩ 0/−0.3/−0.6/−1.0 → κ₁
    # +0.02/−0.24/−0.47/−0.73, compact κ₀≈±1, bistable. lif needs supercritical memory (g·λ₀·φ'(0)>1,
    # φ'(0)≈0.4) → gain=2, memory_lambda=3 (toy regime). Sweep the DC: does the task stay intact and do
    # the wells drop as predicted (and does training KEEP the negative ⟨n₁⟩ vs the baseline-pin dragging
    # it to 0)? --run_filter lifdc.
    LIFDC_SEEDS = 4
    emergent_lif = {**emergent, "nonlinearity": "lif", "gain": 2.0}
    lifdc_arms = [("lifdc0", 0.0), ("lifdc03", -0.3), ("lifdc06", -0.6), ("lifdc10", -1.0)]
    for tag, drm in lifdc_arms:
        for seed in range(LIFDC_SEEDS):
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed, memory_lambda=3.0,
                                     decision_readout_mean=drm,
                                     nogo_push_memory=False, ramping_gng=True,
                                     windowed_targets=True, decay_to_zero=False,
                                     dual_gng_memory=False,
                                     gng_response=True, rwd_nogo_onesided=False,
                                     attention_gated=True,
                                     **emergent_lif, **shared_unfrozen))

    # ★★★ KILL THE UP COPIES (2026-08-04) §21. sweep_lifdc gave 4 wells (2 up + 2 DOWN) because the
    # DECISION axis is deeply SUPERCRITICAL (measured g·λ₁≈8–9, threshold for lif = 1/φ'(0)≈2.5) → a
    # decision double-well → each memory splits into an up and a down attractor. Kill the up copy by
    # making the decision SUBCRITICAL (kappa1_clamp caps g·λ₁), so κ₁ is monostable at the DC-set value
    # (½⟨n₁⟩<0 = down) → ONE down well per memory. The clamp rescales n₁ (shrinks the DC ~√(clamp/g·λ₁)),
    # so use a STRONGER decision_readout_mean to keep ⟨n₁⟩ negative after the rescale. Base = lifdc.
    # Predict: control keeps 4 wells; clamped arms → 2 DOWN wells only (GOAL all-down 4/4), task intact
    # (decision becomes input-driven/transient, the memory stays the persistent attractor). --run_filter lifup.
    LIFUP_SEEDS = 4
    lifup_arms = [   # (tag, decision_readout_mean, kappa1_clamp)
        ("lifup_ctl",  -1.0, None),    # control: supercritical decision → 4 wells
        ("lifup_c20",  -1.5, 2.0),     # subcritical (g·λ₁≤2.0, eff 0.80) + stronger DC
        ("lifup_c15",  -2.0, 1.5),     # more subcritical (eff 0.60) + strongest DC
    ]
    for tag, drm, clamp in lifup_arms:
        for seed in range(LIFUP_SEEDS):
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed, memory_lambda=3.0,
                                     decision_readout_mean=drm, kappa1_clamp=clamp,
                                     nogo_push_memory=False, ramping_gng=True,
                                     windowed_targets=True, decay_to_zero=False,
                                     dual_gng_memory=False,
                                     gng_response=True, rwd_nogo_onesided=False,
                                     attention_gated=True,
                                     **emergent_lif, **shared_unfrozen))

    # ★★★ TASK-FLAG conditions for the UP wells to vanish EMERGENTLY (2026-08-04) §21. Keep lif + the
    # DC (decision_readout_mean — a transfer-function shift, KEPT) but NO clamp. The up copies come from
    # a SUPERCRITICAL decision axis (g·λ₁≈8); ask which TASK STRUCTURE keeps the decision transient
    # (subcritical) on its own → only the DOWN wells survive. Feature-isolation on the lif+DC(−1.0)
    # base (each arm toggles ONE task flag). --run_filter tf_
    tf_common = dict(memory_lambda=3.0, decision_readout_mean=-1.0,
                     nogo_push_memory=False, ramping_gng=True,
                     windowed_targets=True, decay_to_zero=False, dual_gng_memory=False,
                     gng_response=True, rwd_nogo_onesided=False, attention_gated=True)
    tf_arms = [
        ("tf_base",  {}),                           # reference: 4 wells (2 up + 2 down)
        ("tf_decay", {"decay_to_zero": True}),      # decision + pairing decay to 0 → transient
        ("tf_1s",    {"rwd_nogo_onesided": True}),  # nogo scored by lick penalty only (no pin/hinge)
        ("tf_ngt",   {"nogo_target": -1.0}),        # nogo response target −1 (vs 0)
        ("tf_norwd", {"gng_response": False}),      # no 0.5 s response window (go/nogo emergent on κ₁)
        ("tf_gm",    {"dual_gng_memory": True}),    # supervise the go/nogo pre-cue hold in Dual
        # reruns (2026-08-04): fuller DPA convergence (epochs_dpa 100→250, stop_loss 0.1→0.02) on the
        # tf_1s one-sided base. tf1sep = attention ON (the rerun); tf1sna = attention OFF (does the
        # well-lowering survive with NO attention symmetry-breaker?). --run_filter tf1sep / tf1sna.
        ("tf1sep",   {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.02}),
        ("tf1sna",   {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.02,
                      "attention_input": False}),
        # attention-only: NO DC (decision_readout_mean 0), attention ON — does the attention
        # symmetry-breaker alone lower the wells without the readout DC shift? --run_filter tf1sndc
        ("tf1sndc",  {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.02,
                      "decision_readout_mean": 0.0}),
        # control cell of the 2×2 (well-lowering levers): NO DC and NO attention → neither
        # symmetry-breaker. --run_filter tf1snn
        ("tf1snn",   {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.02,
                      "decision_readout_mean": 0.0, "attention_input": False}),
        # (a) NOISE sweep on the DC=0, attention-on, one-sided base (attention frozen in GNG by
        # default). Hypothesis: emergent well depth = noise-robustness margin, so raising the input
        # noise (0.25→0.5→0.75) should deepen the nogo wells below 0. --run_filter nzA
        # stop_loss=0.05 (not 0.02): Dual solves the task by ~loss 0.05; pushing to 0.02 just overtrains
        # (300 Dual epochs) and can distort the wells. DPA already converges by 0.05 (fixed eval).
        ("nzA5",     {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.05,
                      "decision_readout_mean": 0.0, "noise": 0.5}),
        ("nzA7",     {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.05,
                      "decision_readout_mean": 0.0, "noise": 0.75}),
        # higher-noise plain one-sided (= sweep_noise / sweep_nodc_afrz condition, control for nzB1).
        # --run_filter nzA1
        ("nzA10",    {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.05,
                      "decision_readout_mean": 0.0, "noise": 1.0}),
        ("nzA15",    {"rwd_nogo_onesided": True, "epochs_dpa": 250, "stop_loss": 0.05,
                      "decision_readout_mean": 0.0, "noise": 1.5}),
        # (a)+(b) go-preserving one-sided (KEEP the go +1 hinge) at elevated noise → under the shared
        # response cue the net must seat nogo below the lick line (margin binds). --run_filter nzB
        ("nzB5",     {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "epochs_dpa": 250,
                      "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 0.5}),
        ("nzB7",     {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "epochs_dpa": 250,
                      "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 0.75}),
        # HIGHER-noise rerun of the go-preserving winner (nogo wells deepened −0.75→−0.93 over noise
        # 0.5→0.75). Does depth keep growing, and does the task survive? --run_filter nzB1
        ("nzB10",    {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "epochs_dpa": 250,
                      "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 1.0}),
        ("nzB15",    {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "epochs_dpa": 250,
                      "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 1.5}),
        # transient/decaying decision (decay_to_zero): κ₁ is driven back to rest AFTER the response so
        # the go-up is a decaying pulse, not a latched state — does it collapse the go-memory UP wells
        # while keeping go correct? go-preserving base at the 0.75/1.0 sweet spot. --run_filter nzBd
        ("nzBd7",    {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "decay_to_zero": True,
                      "epochs_dpa": 250, "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 0.75}),
        ("nzBd10",   {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "decay_to_zero": True,
                      "epochs_dpa": 250, "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 1.0}),
        # ONE-SIDED decay (decay_onesided): instead of pinning the decay window to 0, penalise κ₁>0 on
        # go and κ₁<0 on nogo — each trace relaxes to rest from its own side (transient decision). Does
        # it collapse the go-memory UP wells while keeping go correct? --run_filter nzBo
        ("nzBo7",    {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "decay_to_zero": True,
                      "decay_onesided": True, "epochs_dpa": 250, "stop_loss": 0.05,
                      "decision_readout_mean": 0.0, "noise": 0.75}),
        ("nzBo10",   {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "decay_to_zero": True,
                      "decay_onesided": True, "epochs_dpa": 250, "stop_loss": 0.05,
                      "decision_readout_mean": 0.0, "noise": 1.0}),
        # walk the tightrope: WEAKER one-sided decay (gng_decay_weight 0.5 / 0.25 vs nzBo's 1.0) at
        # noise 0.75 — enough to un-latch the go up-state, gentle enough to keep sample memory (DPA).
        # --run_filter nzBw
        ("nzBw5",    {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "decay_to_zero": True,
                      "decay_onesided": True, "gng_decay_weight": 0.5, "epochs_dpa": 250,
                      "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 0.75}),
        ("nzBw25",   {"rwd_nogo_onesided": True, "rwd_keep_go_hinge": True, "decay_to_zero": True,
                      "decay_onesided": True, "gng_decay_weight": 0.25, "epochs_dpa": 250,
                      "stop_loss": 0.05, "decision_readout_mean": 0.0, "noise": 0.75}),
        # RANK-3: rule on κ₁ (bistable, gng_lambda), lick on κ₂ (subcritical, decision_lambda). The
        # one-sided decay + noise now act on κ₂ (against κ₁), so the transient lick can't spiral into
        # κ₀ — sample wells should stay clean nodes AND lose the up copies. go-preserving base, DC=0.
        # init out=2/gng=1 + freeze κ₀,κ₁ in Dual are already rank-aware. --run_filter r3o
        # RANK-2 baseline (= sweep_noise_go: go-preserving, NO decay), matched noise 0.75/1.0, new 0.25s
        # windows — the up-copy reference to compare the rank-3 run against. --run_filter r2go
        ("r2go7",    {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "decision_readout_mean": 0.0, "epochs_dpa": 250, "stop_loss": 0.05, "noise": 0.75}),
        ("r2go10",   {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "decision_readout_mean": 0.0, "epochs_dpa": 250, "stop_loss": 0.05, "noise": 1.0}),
        # RANK-3, NO decay, noise 1.0. Transient lick relies on κ₂ being SUBCRITICAL (decision_lambda
        # 0.5 → g·λ=1.0 < 2.5) so it relaxes on its own — no decay target needed. The go/nogo RULE must
        # self-sustain on κ₁ (not re-supervised in Dual), so gng_lambda=1.5 (g·λ=3.0 > 2.5) makes κ₁
        # BISTABLE — else the rule decays through the delay. --run_filter r3o
        ("r3o10",    {"rank": 3, "target_rank": 3, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "decision_readout_mean": 0.0, "gng_lambda": 1.5, "decision_lambda": 0.5,
                      "epochs_dpa": 250, "stop_loss": 0.05, "noise": 1.0}),
        # CUE-DRIVEN response (response_in_cue): score the go/pairing response in the LAST 0.5 s of its
        # triggering stimulus (cue/test ON) so the lick is input-DRIVEN, not held from memory — removes
        # the SOURCE of the go-rule "up copies". NO decay target (emergent): subcritical κ₂ relaxes after
        # the stimulus on its own. r3cue = rank-3 (= r3o10 + response_in_cue); r2cue = rank-2 control to
        # see if the root-cause fix alone clears the up-copies without decay. --run_filter cue
        ("r3cue",    {"rank": 3, "target_rank": 3, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "decision_readout_mean": 0.0, "gng_lambda": 1.5, "decision_lambda": 0.5,
                      "response_in_cue": True, "decay_to_zero": False,
                      "epochs_dpa": 250, "stop_loss": 0.05, "noise": 1.0}),
        ("r2cue",    {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "epochs_dpa": 250, "stop_loss": 0.05, "noise": 1.0}),
        # SIGN-BASED decision (hinge-at-0): a lick is a SIGN event, not an amplitude. go_hinge_thresh
        # AND nogo_hinge_thresh both 0 ⇒ no κ₁ supervision demands amplitude anywhere (the ±1 pre-cue
        # holds become pure sign constraints) ⇒ nothing recruits autonomous/supercritical κ₁ structure.
        # The rule bridges the 1 s gap as a SUBCRITICAL decaying trace — removes the amplify(reach ±1)
        # vs contract(transient) conflict whose complex-pair compromise was the spiraling. All margins/
        # depths/lick amplitude must emerge at the NOISE scale (σ_eff≈0.37 → predict lick ≈ +0.4…0.7,
        # wells ≈ −0.4…−1; the κ₁≡0 degenerate optimum is broken by noise alone). Goal portrait:
        # exactly TWO autonomous wells (A/B sample memory), both κ₁<0. = r2cue + hinges at 0.
        # See ring_lowerplane_log §22g→§23. --run_filter sign
        ("sign",     {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 0.0, "nogo_hinge_thresh": 0.0,
                      "epochs_dpa": 250, "stop_loss": 0.05, "noise": 1.0}),
        # SIGN v2 (Leon's strict-sign calibration; margin on the HOLD, sign on the RESPONSE):
        #   HOLDS (pre-cue rule memory): go ≥ +0.25 / nogo ≤ −0.25 (go_hinge_thresh /
        #     nogo_hinge_thresh=−0.25) — the rule trace must carry a real ±ε separation through the
        #     1 s gap, killing the κ₁≡0 degenerate optimum without ±1 amplitude parking.
        #   RESPONSE (in-cue): go ≥ +0.25 strict (a lick must clear threshold); nogo ≤ 0 FREE below
        #     (one-sided relu(κ₁)², rwd_nogo_onesided + gng_rwd_onesided so it applies in the GNG
        #     stage too — no two-sided pin anywhere; response-side depth fully emergent).
        #   dpa_prelick_free=True — NO supervision on the DPA delay readout at all (Leon: "don't
        #     penalize"); wells placed by pairing training alone (legacy two-sided 0-pin removed).
        #   stop_loss=0.005 — sign hinges shrink the loss scale ~25×; 0.05 early-stopped GNG/Dual at
        #     birth (sweep_r2sign: GNG 10 s, nogo 0.38–0.64). 0.005 ≈ 1.5–2σ margins.
        # Eval boundary auto: (0.25 + max(0, −0.25))/2 = 0.125. --run_filter sign2
        ("sign2",    {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 0.25, "nogo_hinge_thresh": -0.25, "nogo_target": 0.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ SYNTHESIS ARM (Leon 2026-08-11): sign2 substrate + the FIXED rwd_window (nogo response
        # truly one-sided, free below — first time for the ε=0.25 base) + rwd_nogo_weight=5 (the
        # down-force that pushed th1w's nogo to −0.2…−0.3). From sign2's per-seed DPA ckpts (DPA
        # config unchanged). Prediction: 2 transient wells (no parking to bend into a U), nogo
        # response sinks, and the cue-coupled pressure pulls the WELLS below the line.
        # --run_filter sign2w5
        ("sign2w5",  {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 5.0,
                      "dpa_ckpt": "results/dual/sweep_r2sign2/s{seed}_sign2/dpa_s{seed}_sign2.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 0.25, "nogo_hinge_thresh": -0.25, "nogo_target": 0.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ SOFTPLUS + AFTER-CUE NOGO (Leon 2026-08-11): every hinge becomes softplus (BCE-with-
        # logits form — gradient σ(x) never dies on the correct side, so DEPTH is finally rewarded
        # by the loss shape itself, §21's missing incentive) AND the gng response window moves back
        # POST-cue (gng_rwd_after_cue): the nogo pressure lands on the RELAXING state near the well —
        # a nolick-like push on the wells, not the cue transient (the sign2w5 lesson: response-window
        # pressure gets absorbed by the input-driven excursion). Pairing stays IN-test (keeps the
        # no-parking benefit). ε=0.25 substrate. Two doses (softplus's persistent gradient makes the
        # weight bite harder — w5 may basement-shove, w1 is the guard). Full runs from DPA: the
        # DPA-stage pairing/memory hinges are softplus too, so DPA ckpts don't transfer.
        # NOTE softplus loss floor >0 ⇒ stop_loss never fires ⇒ full epoch budgets. --run_filter spw
        ("spw1",     {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "softplus", "gng_rwd_after_cue": True,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0, "nogo_target": 0.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        ("spw5",     {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 5.0,
                      "hinge_shape": "softplus", "gng_rwd_after_cue": True,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0, "nogo_target": 0.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # THRESHOLD CONTROL (= sign2 with ALL hinge thresholds at 1): isolates the threshold
        # variable causally — same free DPA delay, same one-sided nogo response (≤0), same
        # response_in_cue, but amplitude-1 demands on holds/go (±1) and pairing (already 1).
        # If the autonomous rule wells RETURN here, the small-ε thresholds are what dissolved
        # them in sign2; if they stay gone, it was the freed delay + one-sided structure.
        # --run_filter th1
        ("th1",      {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0, "nogo_target": 0.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # NOGO-WEIGHT dose on the th1 base (Leon 2026-08-10): both arms found nogo in-cue means
        # slightly ABOVE 0 at the honest boundary (sign2 ≈+0.05, th1 ≈+0.3, nogo(<=0) 0.37–0.52)
        # — the one-sided relu(κ₁)² gradient at +0.05 is ~0.1, no match for the go-side cue
        # demand; the optimizer parks just above the line. rwd_nogo_weight multiplies that
        # down-force (applies in GNG too via gng_rwd_onesided). Fresh full runs from DPA.
        # Score nogo at boundary 0. --run_filter th1w
        ("th1w5",    {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 5.0,
                      "dpa_ckpt": "results/dual/sweep_r2th1/s{seed}_th1/dpa_s{seed}_th1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0, "nogo_target": 0.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        ("th1w10",   {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 10.0,
                      "dpa_ckpt": "results/dual/sweep_r2th1/s{seed}_th1/dpa_s{seed}_th1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0, "nogo_target": 0.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ SOFTPLUS + LATE-DELAY DON'T-LICK (Leon 2026-08-12, the NeuroFlame train_dual.org
        # port): its Dual loss hinges the CHOICE axis DOWN during the DELAY on every no-lick
        # trial — class 0 = NoGo AND the cue-less DPA trials, which sit ON the sample wells
        # there, so the hinge grades the wells' κ₁ directly (delay-time supervision — the §24f
        # factorization route; its response window pushes nothing down). Here: the nolick term
        # restricted to the after-cue span — Dual (cue-off 6.5 s → test-onset 8 s), GNG stage
        # (cue-off 4.5 s → trial end) — softplus(κ₁) ⇒ κ₁≤0 on all free steps. nogo_target=None
        # (Leon): the nogo response target is REDUNDANT with the nolick window (same one-sided
        # hinge(κ₁≤0)) → dropped; "nogo = hold −1 pre-cue, then just don't lick after the cue",
        # one mechanism covering nogo + 'none' (pure-DPA) trials and the go post-response tail
        # (the go +1 response hinge 6.5–7 keeps its own rwd term; eval boundary → (1+(−1))/2 = 0,
        # the honest boundary). spw substrate (all-softplus, post-cue response, th ±1) so the ONLY
        # new force is the delay term; from spw1's per-seed DPA ckpts (DPA config identical — the
        # term doesn't touch the DPA stage). Two doses. --run_filter spnl
        ("spnl1",    {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "softplus", "gng_rwd_after_cue": True,
                      "nolick_late_delay": True, "nolick_weight": 1.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2sp/s{seed}_spw1/dpa_s{seed}_spw1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        ("spnl5",    {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "softplus", "gng_rwd_after_cue": True,
                      "nolick_late_delay": True, "nolick_weight": 5.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2sp/s{seed}_spw1/dpa_s{seed}_spw1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ IN-CUE GO variant (Leon 2026-08-12): = spnl minus gng_rwd_after_cue. The go +1 hinge
        # moves back IN-cue (6.0-6.5, input-driven) — the only Dual-stage demand for autonomous
        # κ₁>0 disappears; the nolick window (6.5-8.0, unchanged, no overlap) then actively erodes
        # inherited up structure. Causal test: do the (κ₀≈0, κ₁≈+3.5) up wells die with the
        # post-cue go demand? (If they persist → GNG-inherited via the ±1 rule hold.) Softplus
        # kept (Leon) — the deep down extras may remain at w5. dual_gng_memory stays False (no
        # rule holds in Dual). Same spw1 DPA ckpts. --run_filter spnlc
        ("spnlc1",   {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "softplus",
                      "nolick_late_delay": True, "nolick_weight": 1.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2sp/s{seed}_spw1/dpa_s{seed}_spw1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        ("spnlc5",   {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "softplus",
                      "nolick_late_delay": True, "nolick_weight": 5.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2sp/s{seed}_spw1/dpa_s{seed}_spw1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ GNG-DECAY variant (Leon 2026-08-12): = spnlc1 + gng_decay_to_zero. The GNG stage pins
        # BOTH trial types back to 0 from the response end to TRIAL END (4.5-6.0 s) — the
        # trajectories showed the rule/lick overlaps softplus-inflated through the delay; full
        # return-to-rest removes parked structure (up AND down) at its source, so Dual inherits
        # transient decisions only. GNG nolick inert under the pin; the down-seating happens
        # purely in Dual's delay term. Dual decay unchanged (off). Same spw1 DPA ckpts.
        # --run_filter spnld
        ("spnld1",   {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "softplus", "gng_decay_to_zero": True,
                      "nolick_late_delay": True, "nolick_weight": 1.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2sp/s{seed}_spw1/dpa_s{seed}_spw1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ RELU² control of spnld1 (Leon 2026-08-12): identical design, hinge_shape back to
        # relu2 — the causal test of the softplus depth-reward. relu² nolick has ZERO gradient at
        # κ₁≤0 (no depth reward, no basement risk): if the wells still seat below the line, the
        # decay-to-rest + delay-window STRUCTURE does the work; if they hover at 0, softplus's
        # never-dying push was load-bearing. stop_loss 0.005 is ACTIVE again (no softplus floor).
        # DPA from th1's ckpts (relu², th ±1, prelick-free, in-test pairing — the exact DPA this
        # arm would train; spw1's are softplus-trained). --run_filter renld
        ("renld1",   {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "gng_decay_to_zero": True,
                      "nolick_late_delay": True, "nolick_weight": 1.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2th1/s{seed}_th1/dpa_s{seed}_th1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ NOLICK DOSE LADDER on the relu² substrate (Leon 2026-08-12): = renld1 with
        # nolick_weight 0 and 5. renldw0 is the CONTROL that isolates the delay term — note it
        # leaves the Dual nogo trials with NO κ₁ supervision at all (no hold: dual_gng_memory
        # False; no response target: nogo_target None; no decay: decay_to_zero False), so nogo
        # must survive purely on the GNG-learned structure and the wells sit wherever the
        # decay-to-rest + pairing put them. renldw5 doses the same term ×5 — under relu² the push
        # SATURATES at κ₁≤0 (no depth reward, unlike softplus w5 which collapsed pairs), so this
        # tests whether more weight buys seating without the inflation/collapse. Filter "renldw"
        # matches both and NOT renld1. --run_filter renldw
        ("renldw0",  {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "gng_decay_to_zero": True,
                      "nolick_late_delay": True, "nolick_weight": 0.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2th1/s{seed}_th1/dpa_s{seed}_th1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        ("renldw5",  {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "gng_decay_to_zero": True,
                      "nolick_late_delay": True, "nolick_weight": 5.0, "nogo_target": None,
                      "dpa_ckpt": "results/dual/sweep_r2th1/s{seed}_th1/dpa_s{seed}_th1.pth",
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ LINEAR-HINGE (L1) version of renld1 (Leon 2026-08-13): hinge_shape="relu" — every
        # hinge becomes relu(x) AND every 0-target pin becomes |p| (the norm follows the hinge).
        # Rationale: relu²'s force 2x vanishes near the target, so the states we care about
        # (nogo straddling 0 at ±0.05, the GNG go trace parked at +0.4) are priced at ~0 and left
        # there; L1 pulls with CONSTANT force right up to the target, so satisfaction is crisp
        # instead of asymptotic. Same equilibrium as relu² (zero gradient once satisfied) — this
        # buys decisiveness, not depth. DPA trained FRESH (relu²-trained th1 ckpts are a different
        # norm — the DPA hinges/pins are L1 here too). NOTE L1 loss values sit ~10x higher near
        # convergence, so stop_loss=0.005 is a much stricter test and likely never fires.
        # --run_filter linld
        ("linld1",   {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "relu", "gng_decay_to_zero": True,
                      "nolick_late_delay": True, "nolick_weight": 1.0, "nogo_target": None,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
        # ★ + HINGED NOGO RESPONSE (Leon 2026-08-13): = linld1 with nogo_target=0.0 instead of
        # None. With rwd_nogo_onesided the 0-target becomes a ONE-SIDED hinge h(κ₁) in the cue
        # response window (≤0, free below) — NOT a two-sided pin. Closes the hole measured on
        # 2026-08-13: with nogo_target=None the nogo trials carry 0/5632 finite target steps in
        # the cue window (5.98-6.48 s) and the nolick window only starts at cue-OFF, so nogo was
        # unsupervised at exactly the moment of decision and crossed into the lick region there
        # (+0.17 expert, +0.75 naive). Now: hinge in the response, hinge through the late delay —
        # continuous "don't lick" from cue onset to test. Same in the GNG stage (gng_rwd_onesided).
        # CAVEAT for reading results.jsonl: the eval boundary is (th_go + max(nogo_target,
        # nogo_hinge_thresh))/2 = (1+0)/2 = 0.5 here vs 0.0 for linld1 — compare arms with
        # flow_verdict.py, which always scores at the fixed boundary 0. --run_filter linldh
        ("linldh1",  {"rank": 2, "target_rank": 2, "rwd_nogo_onesided": True, "rwd_keep_go_hinge": True,
                      "gng_rwd_onesided": True, "rwd_nogo_weight": 1.0,
                      "hinge_shape": "relu", "gng_decay_to_zero": True,
                      "nolick_late_delay": True, "nolick_weight": 1.0, "nogo_target": 0.0,
                      "decision_readout_mean": 0.0, "response_in_cue": True, "decay_to_zero": False,
                      "go_hinge_thresh": 1.0, "nogo_hinge_thresh": -1.0,
                      "dpa_prelick_free": True,
                      "epochs_dpa": 250, "stop_loss": 0.005, "noise": 1.0}),
    ]
    # (rwd_gng needs the last channel → would require attention OFF, which we DON'T want — attention
    # stays on in every arm. So no rwd_gng arm here.) Run these ≤8 at a time (4/GPU) via --run_filter.
    for tag, over in tf_arms:
        kw = {**emergent_lif, **shared_unfrozen, **tf_common, **over}
        for seed in range(4):
            kw_seed = dict(kw)
            # per-seed checkpoint template: "{seed}" in dpa_ckpt/gng_ckpt resolves to this run's seed
            for _ck in ("dpa_ckpt", "gng_ckpt"):
                if kw_seed.get(_ck):
                    kw_seed[_ck] = kw_seed[_ck].format(seed=seed)
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed, **kw_seed))

    # ═══ RESET (Leon 2026-09-02): the simplest two-memory task, NO CUE, gain 1.0. ═══════════════
    # Strip every lever — no cue push, no response window, no nolick, no decay pins, no decoupling.
    # Supervision = the two memories + pairing only: A/B on κ₀ (whole delay, as always), go/nogo on
    # κ₁ with gng_hold_full_delay (stim-off → phantom-cue, symmetric with the A/B supervision;
    # re-supervised in Dual via dual_gng_memory), pairing 0.25 s at test-off, free tails everywhere
    # else. cue_scale=0.0 with cue_on_go_input keeps input_size/timings IDENTICAL to the cue
    # version, so sweep 2 (add the push-up) is a one-scalar delta (cue_scale 0→2). Question: what
    # geometry do two orthogonal sequential memories converge to BEFORE any asymmetry exists?
    # (ring_lowerplane_log §27 gap analysis → reset.) --run_filter nocue
    nocue_common = dict(nonlinearity="lif", gain=1.0, noise=1.0, cue_scale=0.0,
                        memory_lambda=3.0, decision_readout_mean=0.0,
                        nogo_push_memory=False, ramping_gng=False,
                        windowed_targets=True, gng_hold_full_delay=True,
                        dual_gng_memory=True, gng_response=False, nogo_target=None,
                        response_in_cue=False, decay_to_zero=False,
                        attention_gated=True, dpa_prelick_free=True,
                        epochs_dpa=250, epochs_gng=100, epochs_dual=300, stop_loss=0.005)
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_nocue", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common}))

    # fdrelu (Leon 2026-09-08, "compare the foundation model to the same parameters but using relu"):
    # = nocue with nonlinearity="relu" — the ONE-field delta. Φ (lif) is bounded (0,1) with φ′→0 at
    # both ends (the gain-steal mechanism, resting rate 0.5, slope 0.4); relu is unbounded above
    # with φ′=1 wherever active (no saturation, no gain steal, rest at 0). Same gain 1, noise 1,
    # structured init (memory_lambda 3 — with relu's ⟨φ′⟩≈½ the origin is still unstable at init),
    # same loss/epochs/freezing. DPA retrained (φ changes every stage). Plot with --xlim -5 5.
    # Readouts vs nocue: after_gng/dpa, mem HELD/FLIP, leak, well positions + κ scale, couplings
    # n₀ᵀm₁, whether pushdown behaves differently without saturation. --run_filter fdrelu
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_fdrelu", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common, "nonlinearity": "relu"}))

    # rr01 / rr1 / rrb01 (Leon 2026-09-08, "run relu on the dpa stage but with a regularization so
    # that activity does not diverge"): DPA STAGE ONLY (epochs_gng=0, epochs_dual=0) on the fdrelu
    # config. rr01/rr1 = activity L2 w·⟨r²⟩ at w=0.01/0.1 — removes the incentive to grow; with relu
    # the memory field is linear on each half-line, so the best it can do is tune λ⁺→1 (a marginal
    # line attractor, no zero crossing of F₀/κ₀). rrb01 = rr01 + use_unit_bias (random trainable
    # thresholds inside φ): the active set can now SHRINK with amplitude (λ_full > 1 > λ⁺ ⇒ a genuine
    # bounded fixed point, the threshold-linear bump mechanism) — the well test is F₀/κ₀ crossing 0
    # at finite κ₀. Readouts: ⟨r²⟩ per epoch, κ₀ sample-off→test-on, F₀/κ₀ vs κ₀, λ⁺/λ_full, dpa acc.
    # --run_filter rr01 / rr1 / rrb01 (rr1 does not match rr01).
    _dpa_only = dict(nonlinearity="relu", epochs_gng=0, epochs_dual=0, max_val_loss=1e6)
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_rr01", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common, **_dpa_only, "rate_reg_weight": 0.01}))
        configs.append(RunConfig(run_id=f"s{seed}_rr1", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common, **_dpa_only, "rate_reg_weight": 0.1}))
        configs.append(RunConfig(run_id=f"s{seed}_rrb01", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common, **_dpa_only, "rate_reg_weight": 0.01,
                                    "use_unit_bias": True, "unit_bias_trainable": True}))

    # cue1 (Leon 2026-09-02): = nocue + cue_scale=1.0 — the cue INPUT returns, NO new targets
    # (nothing during or after the cue; supervision bit-identical to nocue). DPA stage REUSED from
    # sweep_r2nocue via dpa_ckpt, so GNG/Dual start from the identical memory solution — any change
    # vs nocue is attributable to the cue push alone. --run_filter cue1
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_cue1", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common, "cue_scale": 1.0}))

    # cue2 (Leon 2026-09-03): the cue dose ladder continues — identical to cue1 but cue_scale=2.0.
    # Same nocue DPA ckpts, same everything else. --run_filter cue2
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_cue2", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common, "cue_scale": 2.0}))

    # g2cue1 (Leon 2026-09-03): the foundation at GAIN 2.0 with cue_scale=1.0 — separates the two
    # kinds of interference (§27f): does saturation bring back WEIGHT-level restructuring of the κ₁
    # mode (pure-DPA margin/d′ erosion, the flip) on top of the state-level co-activation cost?
    # DPA must be retrained (gain changes the DPA stage) — no ckpt reuse. --run_filter g2cue1
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_g2cue1", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "gain": 2.0, "cue_scale": 1.0}))

    # atc1 (Leon 2026-09-03): = cue1 with attention_through_cue=True (GNG-task attention stays on
    # to cue-OFF instead of dropping at cue onset). GNG stage ONLY (epochs_dual=0 → the "expert"
    # ckpt is just the naive weights), from the nocue DPA ckpts — the question is purely what the
    # attention context does to the cue PUSH at naive. Prediction on record: GNG training is
    # gradient-blind to the cue epoch, so weights ≈ cue1's and the push ≈ the factorial's
    # "GNG+attn ON" column (+0.31 vs +0.25 nogo). --run_filter atc1
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_atc1", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "cue_scale": 1.0, "attention_through_cue": True, "epochs_dual": 0}))

    # cuego (Leon 2026-09-04): give the cue a JOB. = atc1 + a go RESPONSE window in the cue
    # (gng_response, response_in_cue) whose hinge threshold rwd_go_thresh=2.0 sits ABOVE the ±1 hold —
    # the parked go rule cannot satisfy the lick, the cue must add ≥1 in 0.5 s. nogo response FREE
    # (nogo_target=None, no nolick): no suppression demand yet — we watch the trained push land on
    # nogo. GNG stage only, nocue DPA ckpts. Readouts: go response ≥2 at cue-off, nogo push vs
    # +0.3 (atc1), the direct-drive decomposition at rest/go/nogo holds, hold amplitudes (does the
    # net shrink the holds to keep the cue's gain?), Wi[:,4] growth. --run_filter cuego
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_cuego", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "cue_scale": 1.0, "attention_through_cue": True, "epochs_dual": 0,
                                    "gng_response": True, "response_in_cue": True, "rwd_go_thresh": 2.0,
                                    "nogo_target": None}))

    # cuegop (2026-09-04): = cuego + gng_hold_pin — the hold is CAPPED at ±1 (two-sided), the in-cue
    # lick hinge stays at 2 → the cue must add ≈+1 in 0.5 s. --run_filter cuegop
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_cuegop", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "cue_scale": 1.0, "attention_through_cue": True, "epochs_dual": 0,
                                    "gng_response": True, "response_in_cue": True, "rwd_go_thresh": 2.0,
                                    "nogo_target": None, "gng_hold_pin": True}))

    # cuegoc (2026-09-07): = cuego + gng_hold_ceiling=1.3 — the PREMATURE-LICK ceiling instead of
    # the (rejected) pin. Go hold stays one-sided at ≥1 but must sit BELOW 1.3 before the cue; the
    # in-cue lick hinge stays at 2 → the cue must add ≥0.7 in 0.5 s on go trials. nogo hold/response
    # untouched (≤−1, free below; nothing after the cue) — the trained push is read off nogo.
    # GNG stage only, nocue DPA ckpts. Readouts: gng_ceil component >0 (the term fires), go hold
    # 1.0–1.3 pre-cue, go response ≥2 in-cue, nogo Δκ₁ in the cue vs +0.35 (cuego), Wi[:,4] growth,
    # drive decomposition at the go/nogo holds, pure-DPA damage (cuego's hold inflation hurt DPA).
    # --run_filter cuegoc
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_cuegoc", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "cue_scale": 1.0, "attention_through_cue": True, "epochs_dual": 0,
                                    "gng_response": True, "response_in_cue": True, "rwd_go_thresh": 2.0,
                                    "nogo_target": None, "gng_hold_ceiling": 1.3}))

    # ═══ sgd2 (Leon 2026-09-07): THE SAFEGUARD on cue dose 2 — the no-lick rule of the task. ═══
    # = cue2 + one one-sided hinge κ₁ ≤ 0 wherever a lick would be WRONG, and nothing else:
    #   nogo rows  cue-on → test-on (in-cue + late delay; GNG stage: cue-on → end),
    #   DPA rows   the whole delay (sample-off → test-on; Dual stage only — there are none in GNG),
    #   go rows    NOTHING (Leon: no post-cue stop-licking demand on go — that would be a transience
    #              demand on the go attractor, the decay-pin lineage, and would confound attribution).
    # The rule hold (±1 to cue-on), the A/B memory and the pairing are the foundation's, untouched;
    # the DPA stage is untouched (nocue ckpts) so the delta vs cue2 is GNG+Dual only. The pushes
    # (go stimulus, cue on both types, dose 2 = +0.7 naive / +1.3 Dual on nogo) are the task's and
    # are never traded against — so the only way to satisfy the hinge after the cue is to relocate
    # the structure the state lands on: the wells. Pre-registered readouts: expert wells' κ₁ (cue2:
    # +0.27…+0.68) → below 0; nogo late-delay κ₁ (cue2: +0.25…+0.46, steps>0 .81–.96) → ≤0; the
    # nolick component > 0 in the Dual loss line; pairing by-gng (cue2 1/1/1) — does the cost
    # finally appear; retention (after_gng/dpa) and leak vs cue2; GNG-stage term expected ~inert
    # at dose 2 (nogo lands at −0.4 at cue-off). --run_filter sgd2
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_sgd2", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common, "cue_scale": 2.0,
                                    "nolick_weight": 1.0, "nolick_late_delay": False,
                                    "nolick_full_delay": True, "nolick_nogo_in_cue": True,
                                    "nolick_thresh": 0.0}))

    # sgd2g (Leon 2026-09-07, "let's try and constrain go trials to stop licking after cue too"):
    # = sgd2 + nolick_late_delay — go rows get the hinge κ₁≤0 from cue-OFF → test-on (GNG stage:
    # cue-off → end); the in-cue lick stays free. nogo (cue-on →) and DPA rows (whole delay) as in
    # sgd2. Both stages carry the rule (the GNG stage is no longer inert: go must relax after the
    # cue there too — read the naive ckpt separately). Same nocue DPA ckpts. Question: does the
    # go tail add the force that takes the wells BELOW the line, or does it just make the go rule
    # transient (the decay-pin lineage)? --run_filter sgd2g
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_sgd2g", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common, "cue_scale": 2.0,
                                    "nolick_weight": 1.0, "nolick_late_delay": True,
                                    "nolick_full_delay": True, "nolick_nogo_in_cue": True,
                                    "nolick_thresh": 0.0}))

    # sgd2x (2026-09-07): CONTINUE sgd2's Dual stage for 300 more epochs. At 300 the Dual val loss
    # was still falling (0.11–0.16 vs cue2's 0.07–0.10) with the nolick term the largest residual,
    # the DPA-trial wells parked exactly AT the line and the nogo late delay at +0.17…+0.20 — a
    # descent in progress, not an equilibrium. gng_ckpt = sgd2's EXPERT ckpt (the loader takes a
    # plain state dict, so "naive" here = sgd2's expert; the 'after GNG' eval is that state), then
    # 300 Dual epochs with the identical loss. Fresh AdamW state, constant lr — no schedule to
    # resume. Question: do the wells go BELOW the line, or does the go hold give way first?
    # --run_filter sgd2x
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_sgd2x", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 gng_ckpt=f"results/dual/sweep_r2sgd2/s{seed}_sgd2/expert_s{seed}_sgd2.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common, "cue_scale": 2.0,
                                    "nolick_weight": 1.0, "nolick_late_delay": False,
                                    "nolick_full_delay": True, "nolick_nogo_in_cue": True,
                                    "nolick_thresh": 0.0}))

    # ═══ sgd2w3 (2026-09-09): the same rule at DOSE 3 — does the hinge buy DEPTH under noise? ═══
    # = sgd2 with nolick_weight 1.0 → 3.0 and NOTHING else changed (same shape, same windows, same
    # threshold 0, same nocue DPA ckpts; GNG retrained so the arm is self-consistent). This is a dose
    # of the task's own rule, not a new rule — the safeguard framing (§27h) is untouched.
    # WHY IT IS NOT FUTILE (the §25e caveat is a DETERMINISTIC statement): training runs at noise 1.0,
    # σ_eff ≈ 0.37 in κ units, so the term the network actually descends is the noise-SMOOTHED hinge
    #     E[relu(κ₁)²] = (μ²+σ²)Φ(μ/σ) + μσφ(μ/σ),   dE/dμ = 2[μΦ(μ/σ) + σφ(μ/σ)],
    # which is strictly positive for every μ — there IS a restoring force below the line, decaying
    # like the Gaussian tail (100 % of its μ=0 value at 0, 21 % at −1σ, 8 % at −2σ). So the seat is a
    # force BALANCE, not a zero-gradient dead zone, and weight is the legitimate lever on it.
    # PRE-REGISTERED PREDICTION (calibrate the opposing force from sgd2: DPA rows sat at μ ≈ −0.045 at
    # w=1 ⇒ F_opp ≈ 0.25, assumed weight-independent):
    #     w=3 → DPA-row mean κ₁ ≈ −0.30 (0.8 σ below the line, ≈21 % of steps > 0)
    #     w=5 → ≈ −0.41 (1.1 σ, ≈13 %)      [w=5 arm held back pending this result]
    # If the measured μ falls well SHORT of −0.30, F_opp is not constant — the go hold / pairing push
    # back harder as the wells descend, which is the rank-2 entanglement itself and the real finding.
    # Pre-registered costs: go hold pre-cue (sgd2 +0.81…+0.95, already below θ=1) and gng_pos
    # (0.017–0.085); pairing by-gng (1/1/1) and after_gng/dpa (1.00) — does the cost finally appear.
    # --run_filter sgd2w3
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_sgd2w3", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2nocue/s{seed}_nocue/dpa_s{seed}_nocue.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common, "cue_scale": 2.0,
                                    "nolick_weight": 3.0, "nolick_late_delay": False,
                                    "nolick_full_delay": True, "nolick_nogo_in_cue": True,
                                    "nolick_thresh": 0.0}))

    # ═══ w5 (Leon 2026-09-09): the A/B memory hinge only in the LAST 0.5 s before the test. ═══════
    # Leon: "run relu but where the hinge target for dpa is ±1 only during the last 0.5 s of late
    # delay — that would be consistent with the gng targets." Indeed: the GNG identity hold is a
    # 0.25 s window ENDING at cue onset (generate_gng_trials, hold_full_delay=False), i.e. a short
    # hold just before the readout. The A/B memory has always been the odd one out — supervised from
    # SAMPLE ONSET to test onset (6 s, the sample included). New field `dpa_hold_window` (seconds,
    # 0 = legacy); DPA stage only (the Dual generator sets no κ₀ target at all — mem_* ≡ 0 there).
    #
    # WHY THIS IS THE RIGHT KNOB FOR RELU (§28b/§28c). With relu the memory field is linear on each
    # half-line, F₀ = (λ⁺ − 1)κ₀, so a bounded memory REQUIRES λ⁺ < 1 (then a well can only come from
    # the active set shrinking with amplitude, λ_full > 1 > λ⁺). Nothing so far bought λ⁺ < 1: the
    # activity L2 only tuned the growth RATE (λ⁺ stayed 1.09–1.34 in every arm). The legacy window is
    # a reason why. It demands |κ₀| ≥ θ **already during the sample**, so it prices the RISE TIME as
    # well as the amplitude — and recurrent self-amplification (λ⁺ > 1) is the cheapest way to get
    # from 0 to θ inside a 1 s sample. A terminal window prices only "be there when it is read": a
    # subcritical λ⁺ < 1 network that integrates the sample and settles at κ* = drive/(1 − λ⁺) ≥ θ
    # now satisfies the loss exactly, and the decay it would suffer over the delay is no longer
    # scored anywhere. So this is the first change that makes λ⁺ < 1 AFFORDABLE — and it does it by
    # REMOVING supervision, not by adding a term (no engineered constraint; cf. the safeguard rule).
    #
    # Three arms, DPA stage only (epochs_gng=0, epochs_dual=0), 4 seeds each:
    #   w5relu = fdrelu + dpa_hold_window 0.5                    — the window ALONE (one-field delta)
    #   w5rl2  = w5relu + rate_reg_weight 0.01                   — window × activity L2 (rr01's dose):
    #            the window makes λ⁺ < 1 affordable, the L2 removes what is left of the growth
    #            incentive. If either alone fails and this works, the mechanism is the conjunction.
    #   w5lif  = nocue + dpa_hold_window 0.5                     — SUBSTRATE CONTROL: what does the
    #            same windowing do to the foundation? The foundation's wells (±1.2–1.3) come from Φ
    #            saturating, not from the window, so they should be unchanged. If they are NOT, the
    #            relu result is not attributable to φ and the whole comparison moves.
    # Pre-registered readouts (the §28c table): dpa acc · κ₀ at sample-off → test-on · λ⁺ (weights)
    # vs the measured asymptotic slope · whether F₀/κ₀ CROSSES ZERO at finite κ₀ (the well test) ·
    # ⟨r²⟩ and max unit rate · and, new here, the DECAY over the unsupervised delay (κ₀ at 3 s vs at
    # test-on) — with the window the trajectory between them is free for the first time.
    # Launch: --run_filter w5r (w5relu + w5rl2, relu, xlim -5 5) and --run_filter w5lif (separately;
    # lif plots at ±1.5). Substring check: "w5r" does NOT match "w5lif".
    # stop_loss 0.1 (Leon 2026-09-09) instead of the foundation's 0.005 — the practical threshold
    # (CLAUDE.md): at noise 1.0 the val loss floor is well above 0.005, so that guard never fires.
    _w5 = dict(epochs_gng=0, epochs_dual=0, dpa_hold_window=0.5, stop_loss=0.1)
    _w5relu = {**_w5, "nonlinearity": "relu", "max_val_loss": 1e6}
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_w5relu", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common, **_w5relu}))
        configs.append(RunConfig(run_id=f"s{seed}_w5rl2", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common, **_w5relu,
                                    "rate_reg_weight": 0.01}))
        configs.append(RunConfig(run_id=f"s{seed}_w5lif", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common, **_w5}))

    # ═══ sc (Leon 2026-09-09): "is that not an initial condition problem?" — start SUBCRITICAL. ════
    # Every relu arm so far (fdrelu, rr01/rr1/rrb01, w5relu, w5rl2) used the foundation's
    # memory_lambda=3.0. The structured init splits the units symmetrically, so λ⁺ = g·λ₀/2 EXACTLY
    # at init (measured, 4 seeds: memory_lambda 0.8/1.2/1.5/1.8/2.0/2.5/3.0 → λ⁺ 0.40/0.60/0.75/
    # 0.90/1.00/1.25/1.50). memory_lambda 3.0 ⇒ λ⁺ = 1.50: every one of those 20 seed-runs STARTED
    # in the escape regime. Training then walked λ⁺ DOWN (→1.25–1.39 bare, →1.09–1.30 with the
    # activity L2) and stalled just above 1 in all of them — the signature of a BARRIER at λ⁺ = 1,
    # not of a missing solution. Crossing it from above means passing through marginal memory, where
    # the loss gets worse before it gets better; gradient descent will not pay that.
    # λ⁺ = 1 is memory_lambda = 2.0 — nothing has ever been run on the other side of it.
    #
    # WHAT COULD LIVE THERE. The homogeneity argument (F₀ = (λ⁺−1)κ₀, origin the only fixed point)
    # holds only for a STRICTLY homogeneous net. These are not: `attention_gated=True` puts a tonic
    # input on for exactly the delay (sample-off → test-on). With a constant drive a subcritical net
    # has a stable fixed point κ* = c/(1−λ⁺) ≠ 0, and the +κ₀ / −κ₀ half-lines activate different
    # unit sets, hence carry their own λ± and c± — so TWO input-sustained fixed points, one per
    # half-line, are possible: a genuine bistable relu memory at λ⁺ < 1. The terminal hold window is
    # the right partner for it (a settled state need only be there when read), so these are built on
    # w5relu, not fdrelu — one-field delta = memory_lambda.
    #
    # Three arms, DPA stage only, 4 seeds each:  sc12 (λ⁺ 0.60) · sc16 (λ⁺ 0.80) · sc18 (λ⁺ 0.90)
    # THE QUESTION: from below, does training (a) STAY subcritical and build the input-sustained
    # bistable memory, or (b) climb back through λ⁺ = 1 into escape? Note (b) is the honest null:
    # from below, raising λ⁺ monotonically slows the decay, so there IS a smooth gradient pushing it
    # up — unlike the descent from above, nothing here is blocked by a barrier.
    # Pre-registered readouts (the §28c table): λ⁺ trained vs at init (did it cross 1) · F₀/κ₀ sign
    # at κ₀ = 1 and 3 and any zero crossing · κ₀ at sample-off / 3 s / test-on and the ratio (a
    # settled state gives ≈1, an escape ≫1, a decay ≪1) · dpa acc · ⟨r²⟩ and max unit rate.
    # ⚠ risk to read for: attention is RELEASED at test onset, so an input-sustained memory decays
    # from t=8 s — the hold window (7.49–7.99 s) is scored before that, and the pairing readout is
    # driven by the test input, but a seed that fails pairing while holding κ₀ is showing exactly this.
    # --run_filter sc12 / sc16 / sc18  (or "sc1" for all three; none collide with older arms)
    for seed in range(4):
        for tag, ml in [("sc12", 1.2), ("sc16", 1.6), ("sc18", 1.8)]:
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed,
                                     **{**emergent, **shared_unfrozen, **nocue_common, **_w5relu,
                                        "memory_lambda": ml}))

    # ═══ scl (Leon 2026-09-09): "what happens with the foundation lif when using subcritical init?" ══
    # The φ-swap partner of sc12/sc16/sc18 — identical memory_lambda ladder on Φ instead of relu.
    # MEASURED AT INIT (gain 1, attention on, 4 seeds; F₀/κ₀ near the origin):
    #   memory_lambda   0.8    1.2    1.6    1.8    2.0    2.5    3.0    4.0
    #   lif           −0.67  −0.53  −0.40  −0.34  −0.27  −0.12  +0.04  +0.35   → pitchfork ≈ 2.9
    #   relu          −0.58  −0.38  −0.19  −0.09  +0.01  +0.26  +0.51  +1.00   → pitchfork  = 2.0
    # relu's is at λ⁺ = 1 exactly; Φ's is later because its slope at rest is 0.40 and the tonic input
    # sits the operating point further onto the flat part. NOTE the foundation's memory_lambda = 3.0
    # is only BARELY past Φ's pitchfork (F₀/κ₀ = +0.042, wells at init κ₀* ≈ 0.03) — it starts at the
    # bifurcation and training grows it from there (λ₀ 3.0 → 6.1–7.4, wells → ±1.2). So 1.2/1.6/1.8
    # are subcritical for BOTH nonlinearities and the ladder is a clean φ-swap.
    # PREDICTION: lif RECOVERS the foundation. For Φ the crossing is an ordinary supercritical
    # pitchfork — past it the wells grow continuously and saturation bounds them, so raising λ₀
    # monotonically improves the memory and there is no marginal regime to traverse (contrast relu,
    # where λ⁺ = 1 has no bounded solution on either side, which is why sc12/16/18 parked on it).
    # If lif recovers → today's init-dependence is specific to relu. If lif STAYS subcritical and the
    # memory decays → the foundation itself is init-dependent, which reframes the whole baseline.
    # Readouts: F₀/κ₀ zero crossing + well κ₀* (foundation 1.19–1.23) · trained g·λ₀ (foundation
    # 6.1–7.4, w5lif 6.1–7.4) · κ₀ 3 s → test ratio (w5lif 1.25–1.51, rises into the well) · dpa.
    # stop_loss stays 0.1 as in w5lif — a single-field delta, and safe: w5lif's val floor was 0.22–0.99,
    # and a subcritical start only makes the loss harder, so the guard will not fire early here (it
    # DID fire at epoch 10–15 on the relu sc arms, which is why those are only a snapshot).
    # --run_filter scl
    for seed in range(4):
        for tag, ml in [("scl12", 1.2), ("scl16", 1.6), ("scl18", 1.8)]:
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed,
                                     **{**emergent, **shared_unfrozen, **nocue_common, **_w5,
                                        "memory_lambda": ml}))

    # ═══ sclf16 (Leon 2026-09-09): GNG + Dual on the subcritical-lif DPA solution. ════════════════
    # "run sweep_r2scl gng and dual stages, we just run 4 seeds." Continues scl16's trained DPA
    # memory (memory_lambda 1.6, λ₀ init 1.6 → trained 5.3–6.4, wells ±1.06–1.11 in 4/4 seeds) into
    # the full sequence via dpa_ckpt, so the GNG/Dual stages start from the EXACT subcritical-born
    # memory and any delta vs nocue is attributable to how that memory was built.
    # scl16 chosen over scl12/scl18: the middle rung, wells in 4/4 seeds, and its trained λ₀ is the
    # closest to the foundation's (6.4–6.9) — the cleanest retention comparison. One line to switch.
    # ⚠ stop_loss BACK TO 0.005 (the foundation's), NOT the 0.1 used for the DPA-only probes. From
    # nocue's log the GNG stage passes val 0.1 between epoch 10 and 50 (0.2521@10 → 0.0461@50 →
    # 0.0270@100), so at 0.1 the GNG stage would stop at ~epoch 25 and `after_gng/dpa` — THE primary
    # metric — would be measured on a half-trained GNG net (exactly what truncated sc12/16/18 at
    # epoch 10). Dual is unaffected either way (val 0.0920 @300, crossing 0.1 near epoch 290).
    # PRIMARY READOUT: after_gng/dpa vs nocue 0.996–1.000 and w5lif (untested). Then: memory
    # HELD/FLIP/DECAY through GNG (traj_verdict), leak, coupling g·n₀ᵀm₁ (foundation ≤0.4 in
    # magnitude; the entanglement signature is −2.5…−3.6), pairing by-gng (1/1/1), and the expert
    # wells. QUESTION: does a memory built from BELOW the pitchfork retain as well as one built from
    # above? Its wells sit slightly closer in (±1.06–1.11 vs ±1.19–1.23) and some sit off the κ₁=0
    # axis (s1_scl12 at κ₁ ≈ ±0.5), so the answer is not obviously yes.
    # --run_filter sclf   ("sclf" does not match scl12/scl16/scl18)
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_sclf16", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2scl/s{seed}_scl16/dpa_s{seed}_scl16.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6,
                                    "epochs_gng": 100, "epochs_dual": 300, "stop_loss": 0.005}))

    # ═══ sclc2 (Leon 2026-09-09): the subcritical-lif substrate WITH THE CUE ON. ═══════════════════
    # = sclf16 + cue_scale 0 → 2.0, one scalar, nothing else. Same scl16 DPA checkpoints (the DPA task
    # has no cue at all, so the memory solution is bit-identical and the delta is GNG+Dual only) —
    # exactly how cue1/cue2 were built on top of nocue.
    # THE 2x2 this completes (after_gng/dpa):
    #                       cue 0            cue 2
    #   λ₀ 3.0 (foundation) 0.996–1.000      0.987–1.000   ← the cue costs the foundation NOTHING
    #   λ₀ 1.6 (subcritical) 0.886–1.000     ← this arm
    # WHY IT SHOULD BITE HERE AND NOT THERE. The subcritical wells are tilted ANTI-SYMMETRICALLY in
    # κ₁ (measured on the DPA ckpt: +κ₀ well at κ₁ −0.34…−0.48, −κ₀ well at +0.23…+0.49; mean|κ₁| 0.31
    # vs the foundation's 0.11). The cue pushes κ₁ UP on both trial types (+1.3 at dose 2, §27d). On
    # an untilted pair that push is symmetric and absorbed; on a tilted pair it lands on two states
    # that already sit on opposite sides of the lick line, so it should drive the −κ₀ side further
    # into κ₁>0 while merely recentring the +κ₀ side. Prediction: retention degrades MORE than the
    # 0.886–0.939 of sclf16, and the damage is ASYMMETRIC between the A and B memory sides — which
    # the foundation could never show because its wells sit on the axis.
    # Pre-registered readouts: after_gng/dpa (vs sclf16 0.886/0.939/1.000/0.929 and cue2 0.987–1.000)
    # · traj_verdict leak, SPLIT BY A vs B (sclf16: A −0.32…−0.80, B +0.59…+0.64 — already asymmetric
    # WITHOUT a cue) · mem HELD/GROW/FLIP · sep · choice@0 · pairing by-gng · expert well κ₁.
    # Seed s2 is the control within the arm: its DPA wells are near-axis (mean|κ₁| 0.12) and it was
    # the one seed with perfect retention at cue 0 — if the mechanism is the tilt, s2 should again be
    # the least damaged.
    # --run_filter sclc   (distinct from sclf / scl12 / scl16 / scl18)
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_sclc2", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2scl/s{seed}_scl16/dpa_s{seed}_scl16.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6, "cue_scale": 2.0,
                                    "epochs_gng": 100, "epochs_dual": 300, "stop_loss": 0.005}))

    # ═══ sclnl (Leon 2026-09-10): the no-lick rule on NOGO trials, on the subcritical substrate. ═══
    # = sclc2 + nolick_weight 1.0 + nolick_nogo_in_cue. NOGO ROWS ONLY (Leon: "a no lick constraint
    # on nogo trials"): the don't-lick span κ₁ ≤ 0 runs from CUE ONSET → test-on in Dual and
    # cue-on → end in GNG. Deliberately NOT nolick_full_delay (that would add the DPA rows, as sgd2
    # did) and NOT nolick_late_delay (go rows stay free after the cue — the measured negative, §27i).
    # Same scl16 DPA checkpoints; GNG+Dual retrained.
    # WHY THIS IS THE CAUSAL TEST of today's correlation. The subcritical wells are tilted
    # ANTI-SYMMETRICALLY (+κ₀ well at κ₁ −0.34…−0.48, −κ₀ well at +0.23…+0.49) and the cue pushes κ₁
    # UP on both trial types, so on a nogo trial the −κ₀ memory state is carried well into κ₁>0 —
    # which is exactly where sclc2's GNG accuracy died in tilt order (tilt 0.12→0.988, 0.29→0.958,
    # 0.34→0.802, 0.49→0.700). If the tilt is the cause, a hinge that forbids κ₁>0 on precisely
    # those trials should pull the wells back toward the axis and RECOVER the accuracy.
    # ⚠ NOTE the term is NOT inert here, unlike the foundation. In sgd2 (λ₀ 3.0) the GNG-stage hinge
    # did nothing at dose 2 because nogo lands at −0.4 by cue-off; here the tilted −κ₀ well sits at
    # +0.23…+0.49 BEFORE the cue's +1.3 push. A consequence worth stating: this arm breaks the
    # gradient-blindness of the GNG stage to cue amplitude (§27d) — the nolick term IS supervised
    # after cue onset — so unlike sclc2, after_gng/dpa is now free to move and is a real readout.
    # Pre-registered: after_gng/gng (sclc2 0.802/0.700/0.988/0.958 — recovery expected, largest in
    # the most tilted seeds s1/s0) · after_gng/dpa (sclf16/sclc2 both 0.886/0.939/1.000/0.929) ·
    # the DPA-ckpt→expert well tilt (does the hinge pull them onto the axis) · traj_verdict leak,
    # A vs B · and the cost §27h found: go hold below θ, gng_pos residual.
    # --run_filter sclnl
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_sclnl", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_r2scl/s{seed}_scl16/dpa_s{seed}_scl16.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6, "cue_scale": 2.0,
                                    "epochs_gng": 100, "epochs_dual": 300, "stop_loss": 0.005,
                                    "nolick_weight": 1.0, "nolick_nogo_in_cue": True,
                                    "nolick_full_delay": False, "nolick_late_delay": False,
                                    "nolick_thresh": 0.0}))

    # ═══ k1zero (Leon 2026-09-10): pin κ₁ = 0 until TEST during the DPA stage. ═════════════════════
    # = scl16 (subcritical lif, λ₀ init 1.6, terminal 0.5 s memory window) with dpa_prelick_free
    # FALSE — the legacy two-sided 0-pin, `targets[:, :n_on[1], -1] = 0.0` in generate_dpa_trials:
    # the readout κ₁ is held at 0 from trial start through the whole delay, released only at test
    # onset where the pairing decision is made. One field; the DPA stage is RETRAINED (its loss
    # changes), so no dpa_ckpt. Full sequence after it (GNG 100 + Dual 300, stop_loss 0.005).
    # WHY THIS IS THE LEVER §29d ASKED FOR. The subcritical memory is BORN ENTANGLED: at the DPA
    # ckpt g·n₀ᵀm₁ = −2.18/−3.76/+0.28/−0.61 (s0/s1/s2/s3), equivalently the memory wells sit
    # anti-symmetrically OFF the κ₁=0 axis (mean|κ₁| 0.34/0.49/0.12/0.29 vs the foundation's 0.11).
    # Everything downstream followed from that number, in seed order: the retention loss (κ₁ LEAK,
    # not memory loss), the cue's damage to GNG, and the no-lick hinge's repair — and the hinge,
    # acting in GNG, fixed the behaviour WITHOUT moving the wells or reducing the coupling. The
    # coupling is created in the DPA stage, so the constraint has to act there. Pinning κ₁ to 0
    # across the delay forbids the memory from carrying a rank-1 component in the first place.
    # PRE-REGISTERED: g·n₀ᵀm₁ at the DPA ckpt (target: |·| ≤ 0.4, the foundation's range, in 4/4
    # instead of 2/4) · memory-well tilt mean|κ₁| (scl16 0.31 → foundation-like 0.11?) · then whether
    # that buys retention: after_gng/dpa vs sclf16's 0.886/0.939/1.000/0.929 and nocue's 0.996–1.000.
    # ⚠ THE KNOWN COST, stated in the field's own docstring: a two-sided pin "clamps wells ON the
    # line" — it does not merely forbid the tilt, it also removes the freedom for the wells to sit
    # BELOW κ₁=0, which is the project's actual goal. So a clean result here is a diagnostic (it
    # would prove the coupling is what costs retention), NOT the target geometry. Watch also whether
    # the pin fights the pairing readout: κ₁ must still express the decision at test, and the pin
    # runs right up to test onset.
    # --run_filter k1zero
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_k1zero", seed=seed,
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6,
                                    "dpa_prelick_free": False,
                                    "epochs_dpa": 250, "epochs_gng": 100, "epochs_dual": 300,
                                    "stop_loss": 0.005}))

    # ═══ k1zcue (Leon 2026-09-10): the κ₁-pinned substrate WITH THE CUE ON. ═══════════════════════
    # = k1zero + cue_scale 2.0, one scalar. Reuses k1zero's OWN DPA checkpoints (the DPA task has no
    # cue, so the memory solution is bit-identical and the delta is GNG+Dual only) — the same move
    # sclc2 made on top of sclf16, so the two ladders are directly comparable.
    # THE TEST. sclc2 showed the cue destroying GNG accuracy in exact TILT order (tilt 0.12→0.988,
    # 0.29→0.958, 0.34→0.802, 0.49→0.700) because the cue pushes κ₁ up on BOTH trial types and the
    # tilted wells sat on opposite sides of the lick line. k1zero flattened those wells onto the axis
    # (tilt 0.34/0.49/0.12/0.29 → 0.11/0.19/0.02/0.06). If the tilt was the vulnerability, the cue's
    # symmetric push should now be ABSORBED — GNG accuracy holding near 0.98 across all four seeds,
    # the way it does on the foundation (cue2: 0.943–0.984), instead of collapsing to 0.700.
    # ⚠ s1 is expected to stay broken on the MEMORY side regardless: its rank1→memory coupling
    # survived the pin (n₀ᵀm₁ = −2.56 vs −0.51/+0.28/−0.08) and it already lost the memory without a
    # cue (after_gng/dpa 0.534, mem LOST, sep 0.45). The cue cannot repair that, and a further drop
    # there is not evidence about the tilt. Read s0/s2/s3 for the tilt question.
    # Pre-registered: after_gng/gng vs sclc2's 0.802/0.700/0.988/0.958 and the foundation's
    # 0.943–0.984 · after_gng/dpa vs k1zero's 0.988/0.534/0.994/0.998 (does the cue cost retention
    # on a clean substrate at all — on the foundation it costs nothing) · leak, A vs B.
    # --run_filter k1zcue
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_k1zcue", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_lif_sub_k1zero/s{seed}_k1zero/dpa_s{seed}_k1zero.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6,
                                    "dpa_prelick_free": False, "cue_scale": 2.0,
                                    "epochs_gng": 100, "epochs_dual": 300, "stop_loss": 0.005}))

    # ═══ k1zcnl (Leon 2026-09-10): the FULL recipe — κ₁ pinned in DPA + cue + no-lick on nogo. ═════
    # = k1zcue + nolick_weight 1.0 + nolick_nogo_in_cue (nogo rows only, cue-on → test-on in Dual and
    # cue-on → end in GNG; go rows free after the cue per §27i; DPA rows NOT included). Same k1zero
    # DPA checkpoints as k1zcue, so all four cells of the 2x2 load a ckpt and are RNG-matched:
    #                        no hinge        + no-lick nogo
    #   tilted wells         sclc2           sclnl
    #   κ₁ pinned in DPA     k1zcue          k1zcnl  (this arm)
    # WHAT THE OTHER THREE CELLS SAID. On TILTED wells the hinge was a rescue with a bill: GNG
    # 0.802/0.700/0.988/0.958 → 0.981/0.972/0.991/0.982, but retention 0.886/0.939/1.000/0.929 →
    # 0.802/0.761/1.000/0.946 (−0.08, −0.18 in the entangled seeds), and it neither moved the wells
    # (tilt 0.38→0.41) nor cut the coupling (−0.99→−0.97) — it treated the symptom. Pinning κ₁ in the
    # DPA stage instead treats the cause: it cuts n₀ᵀm₁ (−2.18→−0.51, −0.61→−0.08) and already gives
    # GNG 0.979/0.985/0.986 and retention 0.964/0.999/0.998 in the three seeds where it took.
    # THE QUESTION: with the wells already ON the axis and the coupling already cut, is the no-lick
    # rule now FREE? There is little left for it to repair (GNG is already ~0.98), so this measures
    # its COST on a clean substrate. Two outcomes, both informative:
    #   (a) near-inert — retention and GNG unchanged, nolick residual small. Then the §27h/sclnl cost
    #       was a property of the ENTANGLED substrate, not of the hinge, and the task's own no-lick
    #       contingency can be imposed for free once the memory is built right. That is the recipe.
    #   (b) it still bills the memory (retention drops as in sclnl). Then the cost is intrinsic to
    #       sharing one κ₁ axis between the rule and the wells, and rank 2 cannot have both.
    # ⚠ s1 stays broken either way (n₀ᵀm₁ = −2.56 survived the pin; memory LOST at 0.534/0.591 with
    # and without the cue). Read s0/s2/s3.
    # Pre-registered: after_gng/dpa vs k1zcue 0.964/0.591/0.999/0.998 — ANY drop is the hinge's bill ·
    # after_gng/gng vs 0.979/0.770/0.985/0.986 · the `nolick` loss component (inert ⇒ ~0) · well tilt
    # and n₀ᵀm₁ at naive (does the hinge disturb what the pin achieved) · go hold vs θ (§27h's cost).
    # --run_filter k1zcnl
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_k1zcnl", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_lif_sub_k1zero/s{seed}_k1zero/dpa_s{seed}_k1zero.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6,
                                    "dpa_prelick_free": False, "cue_scale": 2.0,
                                    "epochs_gng": 100, "epochs_dual": 300, "stop_loss": 0.005,
                                    "nolick_weight": 1.0, "nolick_nogo_in_cue": True,
                                    "nolick_full_delay": False, "nolick_late_delay": False,
                                    "nolick_thresh": 0.0}))

    # ═══ pin_cue1_nolick (Leon 2026-09-10): the full recipe at CUE DOSE 1. ═════════════════════════
    # = the cue-2 full recipe (κ₁ pinned in DPA + cue + no-lick on nogo) with cue_scale 2.0 → 1.0 and
    # stop_loss 0.005 → 0.1, starting from the PREVIOUS sweep's DPA checkpoints
    # (sweep_lif_sub_k1zero_cue_nolick_nogo/s*/dpa_*.pth — the κ₁-pinned memory, unchanged since it
    # was trained: the DPA task has no cue, so dose is irrelevant to that stage).
    # ⚠ stop_loss 0.1 WILL truncate, and this is measured, not guessed: the previous sweep's GNG
    # stage passes val 0.1 between epoch 30 and 35 (1.44@5 → 0.197@20 → 0.113@30 → 0.094@35), so the
    # GNG stage will run ~32 of its 100 epochs and the Dual stage will stop early too. `after_gng/*`
    # here is therefore measured on a LESS-TRAINED GNG net than the cue-2 arm (which ran 100/300 at
    # stop_loss 0.005) — the dose comparison is confounded by training length, and any retention
    # difference cannot be attributed to the cue dose alone. Kept because Leon asked for 0.1.
    # WHY DOSE 1 IS INTERESTING HERE. On the foundation, cue 1 and cue 2 gave IDENTICAL after_gng
    # numbers because the GNG stage was gradient-blind to cue amplitude (§27d — no supervised step
    # after cue onset). This arm BREAKS that blindness: the no-lick term IS supervised from cue onset
    # on, so dose now reaches the GNG gradients for the first time. The cue push scales with dose
    # (nogo +0.25 at dose 1 vs +0.65 at dose 2, naive), so a smaller push should leave the hinge less
    # to do — the question is whether that costs the RULE anything, or whether dose 1 is simply the
    # cheaper way to the same place.
    # Pre-registered vs the cue-2 recipe (0.967/0.670/0.999/0.996 dpa, 0.984/0.966/0.988/0.994 gng):
    # after_gng/dpa and /gng · the `nolick` residual (cue-2: 0.005–0.019; smaller push ⇒ smaller?) ·
    # well tilt and n₀ᵀm₁ at naive · and the epoch each stage actually stopped at.
    # --run_filter pin_cue1
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_pin_cue1_nolick", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_lif_sub_k1zero_cue_nolick_nogo/s{seed}_k1zcnl/dpa_s{seed}_k1zcnl.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6,
                                    "dpa_prelick_free": False, "cue_scale": 1.0,
                                    "epochs_gng": 100, "epochs_dual": 300, "stop_loss": 0.1,
                                    "nolick_weight": 1.0, "nolick_nogo_in_cue": True,
                                    "nolick_full_delay": False, "nolick_late_delay": False,
                                    "nolick_thresh": 0.0}))

    # ═══ pin_cue1_long_nolick (Leon 2026-09-10): the same recipe with a 1 s CUE. ═══════════════════
    # = pin_cue1_nolick + cue_duration 0.5 → 1.0 s. One field. Same DPA checkpoints again (the DPA
    # task has no cue at all, so the pinned memory is untouched by anything about the cue).
    # DURATION vs AMPLITUDE. The cue's push on κ₁ is an INPUT property (§27d) and we have only ever
    # dosed it by amplitude (cue_scale 1 → 2: nogo push +0.25 → +0.65 at naive). Duration is the
    # other axis and has never been varied. They are not interchangeable: amplitude scales the
    # instantaneous drive and is gain-limited by the held state (drive +0.65 at rest vs +0.21 at the
    # nogo hold, §27g), while duration scales how LONG the state is driven, integrating past that
    # saturation. A 1 s cue at dose 1 may therefore push further than a 0.5 s cue at dose 2 despite
    # the smaller amplitude.
    # Windows: cue ONSET is unchanged, so every loss window keyed to cue-on is identical —
    # nolick_nogo_in_cue still spans cue-on → test-on, and the pre-cue rule hold still ends at cue-on.
    # Only the OFFSET moves (dual 6.5 → 7.0 s, still 1 s clear of test at 8.0; gng 4.5 → 5.0 s inside
    # the 6 s trial), which matters only for windows keyed to cue-off — the gng response window and
    # nolick_late_delay, both OFF in this line.
    # ⚠ Inherits `stop_loss` 0.1 from pin_cue1_nolick, which truncated that arm's GNG stage to ~21–41
    # of 100 epochs. Keeping it so the two are comparable, but neither is comparable to the cue-2
    # recipe (100 GNG epochs at stop_loss 0.005) — report the stopping epoch alongside the accuracy.
    # Pre-registered vs pin_cue1_nolick (dpa 0.998/0.706/1.000/0.986, gng 0.995/0.999/0.995/0.999):
    # after_gng/dpa and /gng · the nogo κ₁ push at naive (does a longer cue push further than a
    # bigger one) · the `nolick` residual · GNG stopping epoch.
    # --run_filter pin_cue1_long
    for seed in range(4):
        configs.append(RunConfig(run_id=f"s{seed}_pin_cue1_long_nolick", seed=seed,
                                 dpa_ckpt=f"results/dual/sweep_lif_sub_k1zero_cue_nolick_nogo/s{seed}_k1zcnl/dpa_s{seed}_k1zcnl.pth",
                                 **{**emergent, **shared_unfrozen, **nocue_common,
                                    "dpa_hold_window": 0.5, "memory_lambda": 1.6,
                                    "dpa_prelick_free": False, "cue_scale": 1.0,
                                    "cue_duration": 1.0,
                                    "epochs_gng": 100, "epochs_dual": 300, "stop_loss": 0.1,
                                    "nolick_weight": 1.0, "nolick_nogo_in_cue": True,
                                    "nolick_full_delay": False, "nolick_late_delay": False,
                                    "nolick_thresh": 0.0}))

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _launch_per_run_screens(configs: list, out_dir: str, n_gpus: int, results_path: str):
    """Launch one detached screen session per config, round-robin across GPUs."""
    import dataclasses, subprocess, tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    for i, cfg in enumerate(configs):
        device  = f"cuda:{i % n_gpus}" if torch.cuda.is_available() else "cpu"
        run_dir = os.path.join(out_dir, cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)
        # Write config to a temp JSON file inside the run dir (persists for debugging)
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(dataclasses.asdict(cfg), f)
        log_path = os.path.join(run_dir, "train.log")
        cmd = (
            f"python {here}/_run_one.py {cfg_path} {device} {results_path}"
            f" 2>&1 | tee {log_path}"
        )
        subprocess.run(["screen", "-dmS", f"sweep_{cfg.run_id}", "bash", "-c", cmd])
        print(f"  launched screen session sweep_{cfg.run_id} on {device}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus",          type=int,  default=2)
    parser.add_argument("--n_workers",       type=int,  default=None,
                        help="Concurrent workers (default: n_gpus). "
                             "Set higher to run multiple jobs per GPU.")
    parser.add_argument("--out_dir",         type=str,  default="../results/dual/vanilla")
    parser.add_argument("--wandb_project",   type=str,  default=None,
                        help="W&B project name. Omit to disable W&B logging.")
    parser.add_argument("--per_run_screen",  action="store_true",
                        help="Launch one screen session per run instead of using multiprocessing.")
    parser.add_argument("--run_filter",      type=str,  default=None,
                        help="Only run configs whose run_id contains this substring.")
    parser.add_argument("--nonlinearity",    type=str,  default="relu",
                        help="Nonlinearity passed to make_configs (e.g. relu, tanh).")
    parser.add_argument("--cue_on_go_input", type=int,  default=1, choices=[0, 1],
                        help="1: cue rides on go channel (input_size 6); 0: cue on own channel (input_size 7).")
    parser.add_argument("--nogo_target",     type=float, default=None,
                        help="Override nogo_target in make_configs (e.g. 0.0 or -1.0).")
    parser.add_argument("--hinge_squared",   type=int,  default=None, choices=[0, 1],
                        help="Override DPA hinge shape: 1=squared (default), 0=linear margin.")
    parser.add_argument("--hinge_gng",       type=int,  default=None, choices=[0, 1],
                        help="Override hinge_gng: 1=one-sided go+nogo hinge; 0=legacy two-sided MSE-to-±1 holds.")
    parser.add_argument("--lr_additive",     type=int,  default=None, choices=[0, 1],
                        help="Override E→E low-rank: 0=multiplicative C·(1+lr) (default), 1=additive C+lr.")
    parser.add_argument("--dense_cee",       type=int,  default=None, choices=[0, 1],
                        help="Override E→E backbone: 0=sparse binary/√K (default), 1=dense ones/N_E.")
    args = parser.parse_args()

    n_gpus        = min(args.n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    n_workers     = args.n_workers if args.n_workers is not None else n_gpus
    out_dir       = args.out_dir
    wandb_project = args.wandb_project
    configs       = make_configs(out_dir, nonlinearity=args.nonlinearity,
                                  cue_on_go_input=bool(args.cue_on_go_input),
                                  nogo_target=args.nogo_target,
                                  hinge_squared=None if args.hinge_squared is None else bool(args.hinge_squared),
                                  lr_additive=None if args.lr_additive is None else bool(args.lr_additive),
                                  dense_cee=None if args.dense_cee is None else bool(args.dense_cee),
                                  hinge_gng=None if args.hinge_gng is None else bool(args.hinge_gng))
    if args.run_filter:
        configs = [c for c in configs if args.run_filter in c.run_id]
        print(f"run_filter={args.run_filter!r}: {len(configs)} matching configs")
    results_path  = os.path.join(out_dir, "results.jsonl")

    os.makedirs(out_dir, exist_ok=True)

    # Skip runs already recorded in results.jsonl
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {json.loads(l)["run_id"] for l in f if l.strip()}
        configs = [c for c in configs if c.run_id not in done]
        if done:
            print(f"Skipping {len(done)} already-completed run(s).")

    print(f"Sweep: {len(configs)} runs | {n_gpus} GPU(s) | {n_workers} workers")
    print(f"Results → {results_path}")
    if wandb_project:
        print(f"W&B project → {wandb_project}")

    if args.per_run_screen:
        print(f"Mode: one screen session per run (sweep_<run_id>)")
        _launch_per_run_screens(configs, out_dir, n_gpus, results_path)
        print(f"All {len(configs)} screen sessions launched.")
        print(f"Monitor: screen -ls | attach: screen -r sweep_<run_id>")
        return

    def _write_result(result: dict):
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    if n_workers == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for cfg in configs:
            result = run_single(cfg, device, wandb_project=wandb_project)
            _write_result(result)
    else:
        mp.set_start_method("spawn", force=True)
        job_queue    = mp.Queue()
        result_queue = mp.Queue()

        for cfg in configs:
            job_queue.put(cfg)
        for _ in range(n_workers):
            job_queue.put(None)          # one sentinel per worker

        workers = [
            mp.Process(target=_worker,
                       args=(worker_id, n_gpus, job_queue, result_queue, out_dir, wandb_project),
                       daemon=True)
            for worker_id in range(n_workers)
        ]
        for w in workers:
            w.start()

        n_ok, n_err = 0, 0
        for _ in configs:
            result = result_queue.get()
            _write_result(result)
            if result["status"] == "ok":
                n_ok  += 1
            else:
                n_err += 1

        for w in workers:
            w.join()

        print(f"\nDone: {n_ok} succeeded, {n_err} failed.  Results in {results_path}")


if __name__ == "__main__":
    main()
