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

from src.tasks import TaskTiming, generate_dpa_trials, generate_gng_trials, generate_dual_trials
from src.models import LowRankModel
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
    gain:         float = 2.0
    input_size:   int   = 8       # 7 inputs + 1 reward channel
    target_rank:  int   = 2

    # Dynamics
    tau:          float = 0.3
    dt_base:      float = 0.03   # dt = dt_base * tau_rec_frac
    tau_rec_frac: float = 0.75   # scales both dt and tau_rec = tau * tau_rec_frac
    noise:        float = 0.5    # input noise prefactor; sigma = noise * sqrt(1 - exp(-alpha)^2)
    model_noise:  float = 0.0    # recurrent noise prefactor (same sigma formula)

    # Initialisation
    init_style:         str   = "structured"   # "structured" | "random"
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
    project_go_on_n1:   bool  = False  # project go input column onto n₁ direction before GNG
    project_gng_orth_n0: bool = False  # project go+nogo input columns orthogonal to n₀ before GNG
    rwd_gng:            bool  = True   # teacher-forced reward during GNG stage (False = disable)

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

    # Task variant
    cue_on_go_input:  bool  = False
    go_on_rwd_input:  bool  = False  # route go stim + cue through reward channel; sets input_size=6
    cue_scale:        float = 1.0   # amplitude of the GNG cue signal
    nogo_target:      float = 0.0   # target value for nogo response window (-1 or 0)
    rwd:              bool  = True   # teacher-forced reward feedback
    rwd_scale:        float = 1.0   # amplitude of the reward pulse (default +1)
    # Which stages freeze ALL input dims. Subset of ['dpa', 'gng', 'dual'].
    # GNG always freezes DPA+rwd dims regardless; 'gng' extends that to all channels.
    freeze_input_stages: list = field(default_factory=lambda: ["dual"])

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
    gng_go_weight:   float = 1.0   # relative weight on go trials within gng_loss
    gng_nogo_weight: float = 1.0   # relative weight on nogo trials within gng_loss
    aux_weight:      float = 1.0   # weight on the memory (non-decision) channels
    bl_weight:       float = 1.0   # weight on the pre-sample baseline term

    # Output
    out_dir: str = "../results/dual/vanilla"

    def __post_init__(self):
        if self.go_on_rwd_input:
            self.input_size = 8 - 2 - int(not self.rwd)   # go+cue merged into rwd channel
        else:
            self.input_size = 8 - int(self.cue_on_go_input) - int(not self.rwd)
        if self.dual_loss not in ("multi", "separated", "threshold"):
            raise ValueError(f"dual_loss must be 'multi', 'separated', or 'threshold', got {self.dual_loss!r}")


# ---------------------------------------------------------------------------
# Accuracy helpers  (defined here so they don't live in the general modules)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _dpa_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1):
    model.eval()
    X, y = generate_dpa_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank)
    pred         = model(X.to(device), y.to(device))[..., -1].cpu()
    decision_t   = int(timing.n_stim_off[1])
    pred_final   = pred[:, decision_t:].mean(1)
    target_final = y[:, -1, -1]
    return ((pred_final > 0) == (target_final > 0)).float().mean().item()


@torch.no_grad()
def _gng_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                  cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False):
    model.eval()
    X, y = generate_gng_trials(n_trials, timing=timing, input_size=input_size,
                                noise=noise, target_rank=target_rank, cue_on_go_input=cue_on_go_input,
                                cue_scale=cue_scale, nogo_target=nogo_target,
                                go_on_rwd_input=go_on_rwd_input)
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
def _dual_accuracy(model, timing, input_size, noise, device, n_trials=1024, target_rank=1,
                   cue_on_go_input=False, cue_scale=1.0, nogo_target=0.0, go_on_rwd_input=False):
    model.eval()
    X, y, _, condition_names = generate_dual_trials(
        n_trials, timing=timing, input_size=input_size, noise=noise, target_rank=target_rank,
        cue_on_go_input=cue_on_go_input, cue_scale=cue_scale, nogo_target=nogo_target,
        go_on_rwd_input=go_on_rwd_input,
    )
    pred  = model(X.to(device), y.to(device))[..., -1].cpu()
    names = np.asarray(condition_names).astype(str)

    dpa_start = int(timing.n_stim_off[3])
    pred_dpa  = pred[:, dpa_start:].mean(1)
    dpa_acc   = ((pred_dpa > 0) == (y[:, -1, -1] > 0)).float().mean().item()

    rwd_start = int(timing.n_stim_off[2])
    rwd_stop  = int(timing.n_stim_on[3])
    pred_gng  = pred[:, rwd_start:rwd_stop].mean(1)
    is_go     = torch.as_tensor(["_go_"   in n for n in names])
    is_gng    = torch.as_tensor(["_go_"   in n or "_nogo_" in n for n in names])
    thresh    = (1.0 + nogo_target) / 2.0
    gng_acc   = (
        ((pred_gng[is_gng] > thresh) == is_go[is_gng]).float().mean().item()
        if is_gng.any() else float("nan")
    )
    return dpa_acc, gng_acc


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

    dpa_timing  = TaskTiming([2.0, 8.0],             [3.0, 9.0],             10.0, DT)
    gng_timing  = TaskTiming([2.0, 4.0],             [3.0, 5.0],             6.0, DT)
    dual_timing = TaskTiming([2.0, 4.0, 6.0, 8.0],  [3.0, 5.0, 7.0, 9.0],  10.0, DT)

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
        print(f"{p}  │  κ-Jacobian eigvals (gain·n^Tm/N): {eig_str}", flush=True)
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
        lr_final = train_l  # use last value
        p = f"[{rid}]"
        print(f"{p}  {name} done in {elapsed:.1f}s"
              f"  final train={train_l[-1]:.4f}  val={val_l[-1]:.4f}"
              f"  dpa={acc['dpa']:.3f}  gng={acc['gng']:.3f}",
              flush=True)

    model = LowRankModel(
        input_size=config.input_size, hidden_size=config.hidden_size,
        output_size=0, rank=config.rank, gain=config.gain,
        alpha=alpha, alpha_rec=alpha_rec, noise=0.0,
        rwd=config.rwd, rwd_scale=config.rwd_scale, device=device,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p = f"[{rid}]"
    print(f"{p} {'═'*60}", flush=True)
    print(f"{p}  RUN START  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{p}  run_id={rid}  seed={config.seed}  device={device}", flush=True)
    print(f"{p}  arch:  hidden={config.hidden_size}  rank={config.rank}  gain={config.gain}"
          f"  input_size={config.input_size}  params={n_params:,}", flush=True)
    print(f"{p}  task:  cue_on_go={config.cue_on_go_input}  rwd={config.rwd}"
          f"  rwd_scale={config.rwd_scale}  freeze_input_stages={config.freeze_input_stages}  init={config.init_style}", flush=True)
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

    if config.init_style == "structured":
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
    gng_criterion = (MaskedGNGLoss(gng_timing, target_weight=1.0, zero_weight=1.0)
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
        opt  = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sched= optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-5)
        return opt, sched

    def _eval(label):
        model.noise = 0.0
        dpa = _dpa_accuracy(model, dpa_timing, config.input_size, noise=noise, device=device,
                            target_rank=config.target_rank)
        gng = _gng_accuracy(model, gng_timing, config.input_size, noise=noise, device=device,
                            target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                            cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                            go_on_rwd_input=config.go_on_rwd_input)
        print(f"[{rid}]   {label}: dpa={dpa:.3f}  gng={gng:.3f}", flush=True)
        model.noise = 0.0   # keep off during remaining training too
        return {"dpa": dpa, "gng": gng}

    # ------------------------------------------------------------------
    # Stage 1 — DPA
    # ------------------------------------------------------------------
    dpa_freeze_input = list(range(config.input_size)) if "dpa" in config.freeze_input_stages else []
    _stage_header("DPA", config.epochs_dpa, dpa_freeze_input, [])
    t0 = time.time()
    X, y   = generate_dpa_trials(config.n_batch, dpa_timing, config.input_size, noise=noise, target_rank=config.target_rank)
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

    trainer    = Optimization(model, tl, vl, criterion, opt, sched,
                              config.grad_clip_norm, num_epochs=config.epochs_dpa,
                              freeze_input_dims=dpa_freeze_input,
                              regularizer=dpa_regularizer,
                              stop_loss=config.stop_loss,
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
    gng_freeze_input = (list(range(config.input_size)) if "gng" in config.freeze_input_stages
                        else [0, 1, 2, 3] + ([config.input_size - 1] if config.rwd else []))
    _stage_header("GNG", config.epochs_gng, gng_freeze_input, [0])
    t0 = time.time()
    X, y   = generate_gng_trials(config.n_batch, gng_timing, config.input_size, noise=noise, target_rank=config.target_rank,
                                  cue_on_go_input=config.cue_on_go_input, cue_scale=config.cue_scale,
                                  nogo_target=config.nogo_target, go_on_rwd_input=config.go_on_rwd_input)
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
    # Stage 3 — Dual  (freeze all input dims)
    # ------------------------------------------------------------------
    dual_freeze_input = list(range(config.input_size)) if "dual" in config.freeze_input_stages else []
    _stage_header("Dual", config.epochs_dual, dual_freeze_input, [0] if config.freeze_rank0_dual else [])
    t0 = time.time()
    X, y, _, _ = generate_dual_trials(config.n_batch, dual_timing, config.input_size, noise=noise, target_rank=config.target_rank,
                                       cue_on_go_input=config.cue_on_go_input, cue_scale=config.cue_scale,
                                       nogo_target=config.nogo_target, go_on_rwd_input=config.go_on_rwd_input)
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
        )
        print(f"[{rid}]  loss=separated"
              f"  dpa_w={config.dpa_weight}  gng_w={config.gng_weight}"
              f"  go_w={config.gng_go_weight}  nogo_w={config.gng_nogo_weight}"
              f"  aux_w={config.aux_weight}  bl_w={config.bl_weight}", flush=True)
    elif config.dual_loss == "threshold":
        dual_criterion = ThresholdLoss(thresh=config.loss_thresh)
        print(f"[{rid}]  loss=threshold  thresh={config.loss_thresh}", flush=True)
    else:
        dual_criterion = criterion
        print(f"[{rid}]  loss=multi (MaskedMultiTargetLoss)", flush=True)

    dual_freeze_rank0 = [0] if config.freeze_rank0_dual else None
    trainer    = Optimization(model, tl, vl, dual_criterion, opt, sched,
                              config.grad_clip_norm, num_epochs=config.epochs_dual,
                              freeze_low_rank_cols=dual_freeze_rank0,
                              freeze_input_dims=dual_freeze_input,
                              stop_loss=config.stop_loss,
                              verbose=True)

    train_l, val_l, _ = trainer.fit()
    dual_loss_components = (dict(dual_criterion.last_components) if config.dual_loss == "separated" else None)
    if dual_loss_components:
        print(f"[{rid}]  loss components: "
              f"{ {k: round(v, 4) for k, v in dual_loss_components.items()} }", flush=True)
    losses["dual"] = {"train": train_l, "val": val_l}
    torch.save(model.state_dict(), os.path.join(models_dir, f"expert_{rid}.pth"))
    acc_after_dual = _eval("after Dual")
    dual_dpa, dual_gng = _dual_accuracy(model, dual_timing, config.input_size, noise=noise, device=device,
                                         target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                         cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                         go_on_rwd_input=config.go_on_rwd_input)
    _stage_summary("Dual", train_l, val_l, acc_after_dual, t0)
    _log_params("after Dual")

    acc = {
        "after_dpa":  acc_after_dpa,
        "after_gng":  acc_after_gng,
        "after_dual": {**acc_after_dual, "dual_dpa": dual_dpa, "dual_gng": dual_gng},
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
    print(f"{p}    after Dual: dpa={acc_after_dual['dpa']:.3f}  gng={acc_after_dual['gng']:.3f}"
          f"  dual_dpa={dual_dpa:.3f}  dual_gng={dual_gng:.3f}", flush=True)
    print(f"{p} {'═'*60}", flush=True)

    return {
        "run_id": rid,
        "status": "ok",
        "config": dataclasses.asdict(config),
        "accuracy": acc,
        "final_train_loss": {stage: v["train"][-1] for stage, v in losses.items()},
        "final_val_loss":   {stage: v["val"][-1]   for stage, v in losses.items()},
        "loss_curves": {stage: {"train": v["train"], "val": v["val"]}
                        for stage, v in losses.items()},
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

def make_configs(out_dir: str) -> list[RunConfig]:
    """
    Return the list of runs to execute.  Edit freely.

    Tips
    ----
    - run_id must be unique across all configs (it names the checkpoint files).
    - Add / remove loops to vary more or fewer axes.
    - Use dataclasses.replace(base_cfg, seed=s, ...) to share defaults cleanly.
    """
    configs = []

    base = dict(
        init_style="random",
        gain=1.0,
        noise=1.0,
        model_noise=0.0,
        cue_on_go_input=True,
        go_on_rwd_input=False,
        freeze_input_stages=["dual"],
        freeze_rank0_dual=True,
        nogo_target=0.0,
        cue_scale=2.0,
        stop_loss=0.1,
        dual_loss="separated",
        epochs_dpa=100,
        epochs_gng=100,
        epochs_dual=100,
        out_dir=out_dir,
    )

    for seed in range(10):
        configs.append(RunConfig(
            run_id=f"s{seed}_orth_n0",
            seed=seed,
            project_gng_orth_n0=True,
            **base,
        ))

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
    args = parser.parse_args()

    n_gpus        = min(args.n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    n_workers     = args.n_workers if args.n_workers is not None else n_gpus
    out_dir       = args.out_dir
    wandb_project = args.wandb_project
    configs       = make_configs(out_dir)
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
