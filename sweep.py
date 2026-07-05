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
from src.train  import Optimization, MaskedMultiTargetLoss, MaskedGNGLoss, MaskedMultiTargetDualLoss, ThresholdLoss, train_val_split
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
    memory_lambda:      float = 0.8
    decision_lambda:    float = 0.5
    target_mn_corr:     float = 0.8
    target_out_mn_corr: float = 0.8
    sample_scale:       float = 1.0
    test_scale:         float = 1.0
    mix_strength:       float = 0.0
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
    nogo_target:      float = 0.0   # target value for nogo response window (-1 or 0)
    go_target:        float = 1.0   # target value for go response window
    input_scale:      float = 1.0   # global multiplier on all stimulus + cue input amplitudes
    attention_input:  bool  = False # tonic attention/context input (last channel, =1 from first stim onset); input_size += 1
    rwd:              bool  = False  # teacher-forced reward feedback
    rwd_scale:        float = 1.0   # amplitude of the reward pulse (default +1)
    # Which stages freeze ALL input dims. Subset of ['dpa', 'gng', 'dual'].
    # GNG always freezes DPA+rwd dims regardless; 'gng' extends that to all channels.
    freeze_input_stages: list = field(default_factory=lambda: ["dual"])
    # Freeze GNG input dims (go/nogo/cue = channels 4..input_size-2) during DPA.
    # Prevents AdamW weight decay from zeroing them before GNG training starts.
    # Has no effect on DPA learning (those channels are always zero during DPA).
    freeze_gng_input_during_dpa: bool = False
    use_scheduler: bool = True  # set False to use constant lr throughout
    optimizer: str = "adamw"    # "adamw" or "adam" (adam has no weight decay)
    dpa_ckpt: str | None = None  # path to existing DPA checkpoint; skips DPA training if set
    gng_ckpt: str | None = None  # path to existing GNG (naive) checkpoint; skips DPA+GNG if set

    # Dual-stage loss selection
    #   "multi"      → MaskedMultiTargetLoss (default; MSE toward ±1/0, all stages identical)
    #   "separated"  → MaskedMultiTargetDualLoss (MSE, split DPA/GNG components)
    #   "threshold"  → ThresholdLoss (squared hinge; zero loss once pred is on correct
    #                  side of thresh; dpa/gng/aux/bl weights ignored)
    # The dpa/gng/aux/bl weights below apply only when dual_loss == "separated".
    dual_loss:  str   = "multi"
    loss_thresh: float = 0.5    # threshold for dual_loss == "threshold"
    dpa_weight:      float = 1.0
    gng_weight:      float = 1.0
    gng_go_weight:   float = 1.0        # relative weight on go trials within gng_loss
    gng_nogo_weight: float = 1.0        # relative weight on nogo trials within gng_loss
    go_hinge_thresh: float | None = None  # if set, go response window uses relu(thresh-pred)² instead of MSE
    nogo_hinge_thresh: float = -1.0       # hinge_gng no-lick threshold during the memory delay (0 after cue)
    dpa_hinge_thresh: float | None = None # if set, DPA ±1 decision uses squared hinge toward ±thresh (DPA + dual stages)
    hinge_squared:   bool = True   # DPA ThresholdLoss: True=relu(...)² (default), False=linear margin relu(...)
    aux_weight:      float = 1.0   # weight on the memory (non-decision) channels
    bl_weight:       float = 1.0   # weight on the pre-sample baseline term
    kappa1_reg_weight: float = 0.0  # penalise gain*n1^T m1/N > 1 during Dual: weight*relu(λ₁-1)²
    kappa1_clamp:    float | None = None  # HARD constraint: rescale m1,n1 so g·λ₁ ≤ this after each Dual step (vs. soft reg)
    kappa_gain_target: float | None = None  # CRITICALITY: pin ALL modes' g·λ to this value (two-sided) after each step, ALL stages
    nolick_weight:   float = 0.0   # one-sided no-lick penalty relu(κ₁)² over free decision windows (GNG+Dual)
    hinge_gng:       bool  = False # unified one-sided decision hinge at κ₁=0 (go+nogo & match/nonmatch, all stages)

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
        if self.dual_loss not in ("multi", "separated", "threshold"):
            raise ValueError(f"dual_loss must be 'multi', 'separated', or 'threshold', got {self.dual_loss!r}")


# ---------------------------------------------------------------------------
# Accuracy helpers  (defined here so they don't live in the general modules)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _dpa_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1, input_scale=1.0, attention_input=False):
    model.eval()
    X, y = generate_dpa_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, input_scale=input_scale,
                                attention_input=attention_input)
    pred         = model(X.to(device), y.to(device))[..., -1].cpu()
    decision_t   = int(timing.n_stim_off[1])
    pred_final   = pred[:, decision_t:].mean(1)
    target_final = y[:, -1, -1]
    return ((pred_final > 0) == (target_final > 0)).float().mean().item()


@torch.no_grad()
def _dpa_accuracy_by_type(model, timing, input_size, noise, device, n_trials=1024, target_rank=1, input_scale=1.0, attention_input=False):
    model.eval()
    X, y = generate_dpa_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, input_scale=input_scale,
                                attention_input=attention_input)
    pred         = model(X.to(device), y.to(device))[..., -1].cpu()
    decision_t   = int(timing.n_stim_off[1])
    pred_final   = pred[:, decision_t:].mean(1)
    target_final = y[:, -1, -1]
    correct      = (pred_final > 0) == (target_final > 0)
    pair_mask    = target_final > 0
    unpair_mask  = target_final < 0
    return {
        "overall": correct.float().mean().item(),
        "pair":    correct[pair_mask].float().mean().item() if pair_mask.any() else float("nan"),
        "unpair":  correct[unpair_mask].float().mean().item() if unpair_mask.any() else float("nan"),
    }


@torch.no_grad()
def _gng_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                  cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False, input_scale=1.0, attention_input=False):
    model.eval()
    X, y = generate_gng_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, cue_on_go_input=cue_on_go_input,
                                cue_scale=cue_scale, nogo_target=nogo_target,
                                go_on_rwd_input=go_on_rwd_input, input_scale=input_scale,
                                attention_input=attention_input)
    pred        = model(X.to(device), y.to(device))[..., -1].cpu()
    stim_epoch  = slice(int(timing.n_stim_on[0]), int(timing.n_stim_off[0]))
    go_ch       = input_size - 1 if go_on_rwd_input else 4
    ngo_ch      = 4              if go_on_rwd_input else 5
    is_go       = X[:, stim_epoch, go_ch].mean(1) > X[:, stim_epoch, ngo_ch].mean(1)
    decision_t  = int(timing.n_stim_off[1])
    pred_final  = pred[:, decision_t:].mean(1)
    thresh      = (1.0 + nogo_target) / 2.0
    return ((pred_final > thresh) == is_go).float().mean().item()


@torch.no_grad()
def _gng_accuracy_by_type(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                           cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False, input_scale=1.0, attention_input=False):
    model.eval()
    X, y = generate_gng_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, cue_on_go_input=cue_on_go_input,
                                cue_scale=cue_scale, nogo_target=nogo_target,
                                go_on_rwd_input=go_on_rwd_input, input_scale=input_scale,
                                attention_input=attention_input)
    pred        = model(X.to(device), y.to(device))[..., -1].cpu()
    stim_epoch  = slice(int(timing.n_stim_on[0]), int(timing.n_stim_off[0]))
    go_ch       = input_size - 1 if go_on_rwd_input else 4
    ngo_ch      = 4              if go_on_rwd_input else 5
    is_go       = X[:, stim_epoch, go_ch].mean(1) > X[:, stim_epoch, ngo_ch].mean(1)
    decision_t  = int(timing.n_stim_off[1])
    pred_final  = pred[:, decision_t:].mean(1)
    thresh      = (1.0 + nogo_target) / 2.0
    correct     = (pred_final > thresh) == is_go
    return {
        "overall": correct.float().mean().item(),
        "go":      correct[is_go].float().mean().item()  if is_go.any()  else float("nan"),
        "nogo":    correct[~is_go].float().mean().item() if (~is_go).any() else float("nan"),
    }


@torch.no_grad()
def _dual_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                   cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False, input_scale=1.0, attention_input=False,
                   go_target=1.0):
    model.eval()
    X, y, _, condition_names = generate_dual_trials(
        n_trials, timing=timing, input_size=input_size, noise=noise, target_rank=target_rank,
        cue_on_go_input=cue_on_go_input, cue_scale=cue_scale, nogo_target=nogo_target,
        go_on_rwd_input=go_on_rwd_input, input_scale=input_scale,
        attention_input=attention_input,
    )
    pred  = model(X.to(device), y.to(device))[..., -1].cpu()
    names = np.asarray(condition_names).astype(str)

    dpa_start = int(timing.n_stim_off[3])
    pred_dpa  = pred[:, dpa_start:].mean(1)
    dpa_acc   = ((pred_dpa > 0) == (y[:, -1, -1] > 0)).float().mean().item()

    # go/nogo: evaluate κ₁ in the AFTER-CUE target window (where the response target actually
    # lives, [n_off[2], n_off[2]+½·(test−cue2)]), and score each side by whether it goes to its
    # target — go reaches the go side, nogo reaches ≤ its target — past the go/nogo midpoint.
    rwd_start = int(timing.n_stim_off[2])
    rwd_stop  = int(timing.n_stim_off[2] + (timing.n_stim_off[3] - timing.n_stim_on[3]) / 2)
    pred_gng  = pred[:, rwd_start:rwd_stop].mean(1)
    is_go     = torch.as_tensor(["_go_"   in n for n in names])
    is_ng     = torch.as_tensor(["_nogo_" in n for n in names])
    thresh    = nogo_target   # decision boundary = the no-lick target: nogo correct iff it goes to ≤ its target
    go_acc    = (pred_gng[is_go] >  thresh).float().mean().item() if is_go.any() else float("nan")
    nogo_acc  = (pred_gng[is_ng] <= thresh).float().mean().item() if is_ng.any() else float("nan")
    gng_acc   = float(np.nanmean([go_acc, nogo_acc]))
    return dpa_acc, gng_acc, go_acc, nogo_acc


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

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
        init_dpa_internal_readout_prepost(
            model, mem=0, out=1,
            memory_lambda=config.memory_lambda,
            decision_lambda=config.decision_lambda,
            target_mn_corr=config.target_mn_corr,
            target_out_mn_corr=config.target_out_mn_corr,
            sample_scale=config.sample_scale,
            test_scale=config.test_scale,
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

    criterion     = MaskedMultiTargetLoss(target_weight=1.0, zero_weight=1.0)
    # DPA stage: squared-hinge targets (relu margin) when dpa_hinge_thresh set, else MSE
    dpa_criterion = (ThresholdLoss(thresh=config.dpa_hinge_thresh, squared=config.hinge_squared)
                     if config.dpa_hinge_thresh is not None else criterion)
    gng_criterion = (MaskedGNGLoss(gng_timing, target_weight=1.0, zero_weight=1.0,
                                   go_hinge_thresh=config.go_hinge_thresh,
                                   nolick_weight=config.nolick_weight,
                                   hinge_gng=config.hinge_gng,
                                   nogo_hinge_thresh=config.nogo_hinge_thresh)
                     if config.nogo_target == 0.0 else criterion)
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
                                    target_rank=config.target_rank, input_scale=config.input_scale, attention_input=config.attention_input)
        gng = _gng_accuracy_by_type(model, gng_timing, config.input_size, noise=noise, device=device,
                                    target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                    cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                    go_on_rwd_input=config.go_on_rwd_input, input_scale=config.input_scale, attention_input=config.attention_input)
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
        _stage_header("DPA", config.epochs_dpa, dpa_freeze_input, [])
        t0 = time.time()
        X, y   = generate_dpa_trials(config.n_batch, dpa_timing, config.input_size, noise=noise, target_rank=config.target_rank, input_scale=config.input_scale, attention_input=config.attention_input)
        print(f"[{rid}]  data: {list(X.shape)} → {list(y.shape)}", flush=True)
        tl, vl     = train_val_split(X.to(device), y.to(device), config.batch_size)
        opt, sched = _opt_and_sched()
        model.noise = model_noise_sigma
        dpa_regularizer = None
        if config.rwd and config.rwd_align_weight > 0.0:
            _w = config.rwd_align_weight
            def dpa_regularizer(m, _w=_w):
                wi_rwd = m.wi.weight[:, -1]
                n1     = m.n[:, 1]
                cos    = torch.dot(wi_rwd, n1) / (wi_rwd.norm() * n1.norm()).clamp_min(1e-8)
                return _w * (1.0 - cos)

        trainer    = Optimization(model, tl, vl, dpa_criterion, opt, sched,
                                  config.grad_clip_norm, num_epochs=config.epochs_dpa,
                                  freeze_input_dims=dpa_freeze_input,
                                  regularizer=dpa_regularizer,
                                  stop_loss=config.stop_loss,
                                  kappa_gain_target=config.kappa_gain_target,
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
        gng_freeze_input = (list(range(config.input_size)) if "gng" in config.freeze_input_stages
                            else [0, 1, 2, 3] + ([config.input_size - 1] if config.rwd else []))
        _stage_header("GNG", config.epochs_gng, gng_freeze_input, [0])
        t0 = time.time()
        X, y   = generate_gng_trials(config.n_batch, gng_timing, config.input_size, noise=noise, target_rank=config.target_rank,
                                      cue_on_go_input=config.cue_on_go_input, cue_scale=config.cue_scale,
                                      nogo_target=config.nogo_target, go_target=config.go_target, go_on_rwd_input=config.go_on_rwd_input,
                                      input_scale=config.input_scale, attention_input=config.attention_input)
        print(f"[{rid}]  data: {list(X.shape)} → {list(y.shape)}", flush=True)
        tl, vl     = train_val_split(X.to(device), y.to(device), config.batch_size)
        opt, sched = _opt_and_sched()
        model.noise = model_noise_sigma
        model.rwd   = config.rwd_gng   # optionally disable reward during GNG training
        trainer    = Optimization(model, tl, vl, gng_criterion, opt, sched,
                                  config.grad_clip_norm, num_epochs=config.epochs_gng,
                                  freeze_low_rank_cols=[0],
                                  freeze_input_dims=gng_freeze_input,
                                  stop_loss=config.stop_loss,
                                  kappa_gain_target=config.kappa_gain_target,
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

    # ------------------------------------------------------------------
    # Stage 2.5 — Dual-paired (MATCH trials only) → overwrites "naive" (curriculum bridge)
    # ------------------------------------------------------------------
    if config.dual_paired_stage:
        paired_freeze_input = list(range(config.input_size)) if "dual" in config.freeze_input_stages else []
        _stage_header("Dual-paired", config.epochs_dual_paired, paired_freeze_input,
                      [0] if config.freeze_rank0_dual else [])
        t0 = time.time()
        Xp, yp, _, _ = generate_dual_trials(config.n_batch, dual_timing, config.input_size, noise=noise,
                                            target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                            cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                            go_target=config.go_target, go_on_rwd_input=config.go_on_rwd_input,
                                            input_scale=config.input_scale, attention_input=config.attention_input,
                                            paired_only=True)
        print(f"[{rid}]  data(paired): {list(Xp.shape)} → {list(yp.shape)}", flush=True)
        tlp, vlp     = train_val_split(Xp.to(device), yp.to(device), config.batch_size)
        optp, schedp = _opt_and_sched()
        model.noise  = model_noise_sigma
        paired_criterion = (MaskedMultiTargetDualLoss(
                timing=dual_timing, dpa_weight=config.dpa_weight, gng_weight=config.gng_weight,
                gng_go_weight=config.gng_go_weight, gng_nogo_weight=config.gng_nogo_weight,
                aux_weight=config.aux_weight, bl_weight=config.bl_weight,
                go_hinge_thresh=config.go_hinge_thresh, dpa_hinge_thresh=config.dpa_hinge_thresh,
                nolick_weight=config.nolick_weight, hinge_gng=config.hinge_gng,
                nogo_hinge_thresh=config.nogo_hinge_thresh)
            if config.dual_loss == "separated" else criterion)
        trainer = Optimization(model, tlp, vlp, paired_criterion, optp, schedp,
                               config.grad_clip_norm, num_epochs=config.epochs_dual_paired,
                               freeze_low_rank_cols=[0] if config.freeze_rank0_dual else None,
                               freeze_input_dims=paired_freeze_input,
                               stop_loss=config.stop_loss,
                               kappa1_clamp=config.kappa1_clamp,
                               kappa_gain_target=config.kappa_gain_target,
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
    _stage_header("Dual", config.epochs_dual, dual_freeze_input, [0] if config.freeze_rank0_dual else [])
    t0 = time.time()
    X, y, _, _ = generate_dual_trials(config.n_batch, dual_timing, config.input_size, noise=noise, target_rank=config.target_rank,
                                       cue_on_go_input=config.cue_on_go_input, cue_scale=config.cue_scale,
                                       nogo_target=config.nogo_target, go_target=config.go_target, go_on_rwd_input=config.go_on_rwd_input,
                                       input_scale=config.input_scale, attention_input=config.attention_input)
    print(f"[{rid}]  data: {list(X.shape)} → {list(y.shape)}", flush=True)
    tl, vl     = train_val_split(X.to(device), y.to(device), config.batch_size)
    opt, sched = _opt_and_sched()
    model.noise = model_noise_sigma

    if config.dual_loss == "separated":
        dual_criterion = MaskedMultiTargetDualLoss(
            timing=dual_timing,
            dpa_weight=config.dpa_weight, gng_weight=config.gng_weight,
            gng_go_weight=config.gng_go_weight, gng_nogo_weight=config.gng_nogo_weight,
            aux_weight=config.aux_weight, bl_weight=config.bl_weight,
            go_hinge_thresh=config.go_hinge_thresh,
            dpa_hinge_thresh=config.dpa_hinge_thresh,
            nolick_weight=config.nolick_weight,
            hinge_gng=config.hinge_gng,
            nogo_hinge_thresh=config.nogo_hinge_thresh,
        )
        print(f"[{rid}]  loss=separated"
              f"  dpa_w={config.dpa_weight}  gng_w={config.gng_weight}"
              f"  go_w={config.gng_go_weight}  nogo_w={config.gng_nogo_weight}"
              f"  aux_w={config.aux_weight}  bl_w={config.bl_weight}"
              f"  go_hinge={config.go_hinge_thresh}  nolick_w={config.nolick_weight}", flush=True)
    elif config.dual_loss == "threshold":
        dual_criterion = ThresholdLoss(thresh=config.loss_thresh)
        print(f"[{rid}]  loss=threshold  thresh={config.loss_thresh}", flush=True)
    else:
        dual_criterion = criterion
        print(f"[{rid}]  loss=multi (MaskedMultiTargetLoss)", flush=True)

    dual_freeze_rank0 = [0] if config.freeze_rank0_dual else None

    dual_regularizer = None
    if config.kappa1_reg_weight > 0.0:
        _w   = config.kappa1_reg_weight
        _gain = float(model.gain) if torch.is_tensor(model.gain) else float(model.gain)
        def dual_regularizer(m, _w=_w, _gain=_gain):
            N     = m.m.shape[0]
            lam1  = _gain * (m.n[:, 1] @ m.m[:, 1]) / N   # gain * n1^T m1 / N
            return _w * torch.relu(lam1 - 1.0) ** 2
        print(f"[{rid}]  κ₁ regularizer: weight={_w}  penalises gain·λ₁ > 1", flush=True)

    trainer    = Optimization(model, tl, vl, dual_criterion, opt, sched,
                              config.grad_clip_norm, num_epochs=config.epochs_dual,
                              freeze_low_rank_cols=dual_freeze_rank0,
                              freeze_input_dims=dual_freeze_input,
                              stop_loss=config.stop_loss,
                              regularizer=dual_regularizer,
                              kappa1_clamp=config.kappa1_clamp,
                              kappa_gain_target=config.kappa_gain_target,
                              verbose=True)
    if config.kappa1_clamp is not None:
        print(f"[{rid}]  κ₁ hard clamp: g·λ₁ ≤ {config.kappa1_clamp} after each Dual step", flush=True)

    train_l, val_l, _ = trainer.fit()
    dual_loss_components = (dict(dual_criterion.last_components) if config.dual_loss == "separated" else None)
    if dual_loss_components:
        print(f"[{rid}]  loss components: "
              f"{ {k: round(v, 4) for k, v in dual_loss_components.items()} }", flush=True)
    losses["dual"] = {"train": train_l, "val": val_l}
    torch.save(model.state_dict(), os.path.join(models_dir, f"expert_{rid}.pth"))
    acc_after_dual = _eval("after Dual")
    dual_dpa, dual_gng, dual_go, dual_nogo = _dual_accuracy(model, dual_timing, config.input_size, noise=noise, device=device,
                                         target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                         cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                         go_on_rwd_input=config.go_on_rwd_input, input_scale=config.input_scale, attention_input=config.attention_input,
                                         go_target=config.go_target)
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
        hidden_size=512, rank=2, target_rank=2,
        gain=2.0,                      # g·λ₀(mem)=1.6 at init (bistable); tanh_asym saturates → no spiral
        init_style="structured",
        memory_lambda=0.8,             # memory mode stays supercritical (deep A/B wells)
        decision_lambda=0.25,          # ↓ from 0.5 → g·λ₁=0.5 at init (decision starts SUBCRITICAL)
        nonlinearity="tanh", nl_gamma=0.0,   # plain odd tanh — symmetry break comes from ATTENTION, not φ
        attention_input=True,          # tonic attention bias (breaks κ-field odd symmetry, replaces tanh_asym)
        nolick_weight=0.5,             # one-sided no-lick pressure over the free delay windows
        hinge_gng=False,               # old MSE loss (matches the tau1 baseline that looked close); go/nogo works via go_hinge_thresh
        rwd_gng=False,                 # no reward-feedback onto the last channel (clean; avoids the rwd/attention collision)
        cue_on_go_input=True,          # cue rides on go channel (attention arm → input_size=7, else 6)
        cue_scale=2.0,
        input_scale=1.0,
        noise=0.5, model_noise=0.0,
        nogo_target=0.0,               # one-sided nogo (consistent no-lick philosophy)
        go_target=1.0,
        go_hinge_thresh=1.0,
        dpa_hinge_thresh=1.0,
        dual_loss="separated",
        freeze_input_stages=["dual"],
        freeze_rank0_dual=False,
        optimizer="adam",              # no weight decay
        learning_rate=0.01,
        use_scheduler=True,
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

    # SLOW-τ ladder: can a SUBCRITICAL decision mode (g·λ₁<1, no autonomous well → ring
    # collapses) hold the go/nogo decision across the ~1 s GNG delay by SLOW TRANSIENT instead
    # of an attractor? Hold time ≈ τ/(1−g·λ); at τ=0.3 s a subcritical mode decays before the
    # delay ends → training is forced supercritical. Slowing τ (×2, ×4) makes the subcritical
    # transient viable. Decision starts subcritical (decision_lambda=0.25); memory stays super-
    # critical (attractor, τ-independent). Prediction: 1× climbs supercritical (ring persists),
    # 2×/4× stay subcritical (ring breaks → two isolated low memory wells).
    fast_arms = [("fast1", 0.30), ("fast2", 0.15), ("fast3", 0.10)]
    for tag, tau in fast_arms:
        for seed in range(3):
            configs.append(RunConfig(run_id=f"s{seed}_{tag}", seed=seed,
                                     tau=tau, **shared))

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
