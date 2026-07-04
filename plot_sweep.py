#!/usr/bin/env python3
"""
plot_sweep.py — comprehensive plotting for a low-rank RNN sweep.

Usage
-----
    python plot_sweep.py --sweep_dir /home/leon/results/dual/sweep_cue
    python plot_sweep.py --sweep_dir /home/leon/results/dual/sweep_cue --no_individual
    python plot_sweep.py --sweep_dir /home/leon/results/dual/sweep_cue \\
                         --run_ids s0_struct_ml0.95_dl0.5_cue s3_random_cue

Output
------
    ./results/figures/{sweep_name}/
        summary/
            accuracy_stages.pdf
            accuracy_by_trialtype.pdf
            fp_scatter_by_stage.pdf              (autonomous, 3 stage panels)
            fp_scatter_by_input_{cond}.pdf       (one by-stage figure per input cond)
            traj_{dpa,naive,expert}_{dpa,go,nogo}.pdf
        individual/{run_id}/
            accuracy_stages.pdf
            accuracy_by_trialtype.pdf
            traj_{dpa,naive,expert}_{dpa,go,nogo}.pdf
            scatter/fp_scatter.pdf
            flow/fp_{dpa,naive,expert}.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import torch

from analyze import load_results
from src.dynamics import (
    find_all_fixed_points, classify_fixed_points, make_input,
    plot_task_flow_fields,
)
from src.models import LowRankModel, EILowRankModel, EISTPModel
from src.tasks import (
    TaskTiming, make_timings,
    generate_dpa_trials, generate_gng_trials, generate_dual_trials,
)

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

GOLDEN          = (5 ** 0.5 - 1) / 2
STYLE_COLORS    = {"structured": "#4C72B0", "random": "#DD8452"}
STAGE_LABELS    = ["After DPA", "After GNG", "After Dual"]
STAGE_KEYS      = ["after_dpa", "after_gng", "after_dual"]
STAGE_X         = np.arange(3)
STAGE_TASK      = {"dpa": "dpa", "naive": "gng", "expert": "dual"}
CONDITION_COLS  = ("tab:blue", "tab:orange")
XLIM = YLIM     = (-1.5, 1.5)

# ── Central color config ──────────────────────────────────────────────────────
# Each entry: (primary, light_shade).  Used by trajectories, accuracy plots,
# and FP scatter — change here to update everywhere.
TRIAL_COLORS = {
    "dpa":  ("#c0392b", "#e8867d"),   # red
    "go":   ("#1a6fa8", "#6aafd4"),   # blue
    "nogo": ("#1e8449", "#6dbb8a"),   # green
}

# Derived — do not edit directly
TRAJ_PALETTE = {           # keyed by gng_status ("none"=DPA-only, "go", "nogo")
    "none": TRIAL_COLORS["dpa"],
    "go":   TRIAL_COLORS["go"],
    "nogo": TRIAL_COLORS["nogo"],
}

DT              = 0.03 * 0.75
_ALPHA          = DT / 0.3
_ALPHA_REC      = DT / (0.3 * 0.75)


# ---------------------------------------------------------------------------
# Per-run metadata
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    run_id:         str
    init_style:     str
    gain:           float
    cue_on_go_input:bool
    input_size:     int      # already post-__post_init__ (decremented for cue_on_go_input and/or rwd=False)
    noise:          float    # input noise prefactor
    model_noise:    float = 0.0  # recurrent noise prefactor (used during eval)
    accuracy:       dict  = None
    rwd:            bool  = True
    rwd_scale:      float = 1.0
    cue_scale:      float = 1.0
    nogo_target:    float = 0.0
    tau:            float = 0.3
    dt_base:        float = 0.03
    tau_rec_frac:   float = 0.75
    nonlinearity:   str   = "tanh"
    nl_gamma:       float = 0.0
    use_unit_bias:  bool  = False
    unit_bias_trainable: bool = True
    model_type:     str   = "lowrank"
    n_inh:          int   = 128
    static_radius:  float = 1.5
    low_rank_scale: float = 0.3
    low_rank_full:  bool  = False
    use_stp:        bool  = False
    stp_U:          float = 0.2
    stp_tau_f:      float = 1.5
    stp_tau_d:      float = 0.3
    stp_dt:         float = 0.0225
    n_neuron:       int   = 2000
    eistp_K:        float = 250.0
    j_stp:          float = 1.0
    eistp_lr_scale: str   = "N"
    eistp_lr_additive: bool = False
    eistp_dense_cee: bool = False
    eistp_r_max:    float | None = None
    use_fixed_weights: bool = False
    fixed_weight_scale: float = 0.8
    fixed_weight_orthogonalize: bool = True
    fixed_weight_sparsity: float = 1.0

    @property
    def alpha(self) -> float:
        dt = self.dt_base * self.tau_rec_frac
        return dt / self.tau

    @property
    def alpha_rec(self) -> float:
        dt = self.dt_base * self.tau_rec_frac
        return dt / (self.tau * self.tau_rec_frac)

    def noise_sigma(self) -> float:
        return float(self.noise * np.sqrt(1.0 - np.exp(-self.alpha) ** 2))

    def model_noise_sigma(self) -> float:
        return float(self.model_noise * np.sqrt(1.0 - np.exp(-self.alpha) ** 2))


def _load_sweep_meta(sweep_dir: str) -> list[RunMeta]:
    rows = []
    for line in open(os.path.join(sweep_dir, "results.jsonl")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        cfg  = r["config"]
        cue  = bool(cfg.get("cue_on_go_input", False))
        isize= int(cfg.get("input_size", 8))
        # input_size in results.jsonl is already post-__post_init__ (decremented if cue_on_go_input)
        rows.append(RunMeta(
            run_id          = r["run_id"],
            init_style      = cfg.get("init_style", "random"),
            gain            = float(cfg.get("gain", 2.0)),
            cue_on_go_input = cue,
            input_size      = isize,
            noise           = float(cfg.get("noise", 0.0)),
            model_noise     = float(cfg.get("model_noise", 0.0)),
            accuracy        = r.get("accuracy", {}),
            rwd             = bool(cfg.get("rwd", True)),
            rwd_scale       = float(cfg.get("rwd_scale", 1.0)),
            cue_scale       = float(cfg.get("cue_scale", 1.0)),
            nogo_target     = float(cfg.get("nogo_target", 0.0)),
            tau             = float(cfg.get("tau", 0.3)),
            dt_base         = float(cfg.get("dt_base", 0.03)),
            tau_rec_frac    = float(cfg.get("tau_rec_frac", 0.75)),
            nonlinearity    = cfg.get("nonlinearity", "tanh"),
            nl_gamma        = float(cfg.get("nl_gamma", 0.0)),
            use_unit_bias   = bool(cfg.get("use_unit_bias", False)),
            unit_bias_trainable = bool(cfg.get("unit_bias_trainable", True)),
            model_type      = cfg.get("model_type", "lowrank"),
            n_inh           = int(cfg.get("n_inh", 128)),
            static_radius   = float(cfg.get("static_radius", 1.5)),
            low_rank_scale  = float(cfg.get("low_rank_scale", 0.3)),
            low_rank_full   = bool(cfg.get("low_rank_full", False)),
            use_stp         = bool(cfg.get("use_stp", False)),
            stp_U           = float(cfg.get("stp_U", 0.2)),
            stp_tau_f       = float(cfg.get("stp_tau_f", 1.5)),
            stp_tau_d       = float(cfg.get("stp_tau_d", 0.3)),
            stp_dt          = float(cfg.get("dt_base", 0.03)) * float(cfg.get("tau_rec_frac", 0.75)),
            n_neuron        = int(cfg.get("n_neuron", 2000)),
            eistp_K         = float(cfg.get("eistp_K", 250.0)),
            j_stp           = float(cfg.get("j_stp", 1.0)),
            eistp_lr_scale  = str(cfg.get("eistp_lr_scale", "N")),
            eistp_lr_additive = bool(cfg.get("eistp_lr_additive", False)),
            eistp_dense_cee = bool(cfg.get("eistp_dense_cee", False)),
            eistp_r_max     = cfg.get("eistp_r_max", None),
            use_fixed_weights = bool(cfg.get("use_fixed_weights", False)),
            fixed_weight_scale = float(cfg.get("fixed_weight_scale", 0.8)),
            fixed_weight_orthogonalize = bool(cfg.get("fixed_weight_orthogonalize", True)),
            fixed_weight_sparsity = float(cfg.get("fixed_weight_sparsity", 1.0)),
        ))
    return rows


# ---------------------------------------------------------------------------
# Model / checkpoint helpers
# ---------------------------------------------------------------------------

def _build_model(meta: RunMeta, device: str):
    if meta.model_type == "eistp":
        return EISTPModel(
            n_neuron=meta.n_neuron, K=meta.eistp_K, rank=2, gain=meta.gain,
            dt=meta.stp_dt, input_size=meta.input_size,
            stp_use=meta.stp_U, stp_tau_fac=meta.stp_tau_f, stp_tau_rec=meta.stp_tau_d,
            j_stp=meta.j_stp, lr_ini=meta.low_rank_scale, lr_scale=meta.eistp_lr_scale,
            lr_additive=meta.eistp_lr_additive, dense_cee=meta.eistp_dense_cee,
            r_max=meta.eistp_r_max,
            train_inputs=False, nonlinearity="relu", device=device,
        )
    if meta.model_type == "ei":
        return EILowRankModel(
            input_size    = meta.input_size,
            output_size   = 0,
            rank          = 2,
            n_exc         = 512,
            n_inh         = meta.n_inh,
            gain          = meta.gain,
            alpha         = meta.alpha,
            alpha_rec     = meta.alpha_rec,
            noise         = 0.0,
            static_radius = meta.static_radius,
            low_rank_scale= meta.low_rank_scale,
            low_rank_full = meta.low_rank_full,
            use_stp       = meta.use_stp,
            stp_U         = meta.stp_U,
            stp_tau_f     = meta.stp_tau_f,
            stp_tau_d     = meta.stp_tau_d,
            stp_dt        = meta.stp_dt,
            rwd           = meta.rwd,
            rwd_scale     = meta.rwd_scale,
            nonlinearity  = meta.nonlinearity,
            device        = device,
        )
    return LowRankModel(
        input_size    = meta.input_size,
        hidden_size   = 512,
        output_size   = 0,
        rank          = 2,
        gain          = meta.gain,
        alpha         = meta.alpha,
        alpha_rec     = meta.alpha_rec,
        noise         = 0.0,
        rwd           = meta.rwd,
        rwd_scale     = meta.rwd_scale,
        nonlinearity  = meta.nonlinearity,
        nl_gamma      = meta.nl_gamma,
        use_unit_bias = meta.use_unit_bias,
        unit_bias_trainable = meta.unit_bias_trainable,
        use_fixed_weights = meta.use_fixed_weights,
        fixed_weight_scale = meta.fixed_weight_scale,
        fixed_weight_orthogonalize = meta.fixed_weight_orthogonalize,
        fixed_weight_sparsity = meta.fixed_weight_sparsity,
        device        = device,
    )


def _load_ckpt(model: LowRankModel, ckpt_dir: str, stage: str,
               run_id: str, device: str) -> bool:
    # new layout: <sweep_dir>/<run_id>/<stage>_<run_id>.pth
    path = os.path.join(ckpt_dir, run_id, f"{stage}_{run_id}.pth")
    if not os.path.exists(path):
        # legacy layout: <sweep_dir>/models/<stage>_<run_id>.pth
        path = os.path.join(ckpt_dir, "models", f"{stage}_{run_id}.pth")
    if not os.path.exists(path):
        return False
    sd = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_real = [k for k in missing if k != "gain"]
    if missing_real or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing_real}, unexpected={unexpected}")
    model.eval()
    return True


def _noise_sigma(prefactor: float) -> float:
    return float(prefactor * np.sqrt(1.0 - np.exp(-_ALPHA) ** 2))


TIMINGS = make_timings(DT)


# ---------------------------------------------------------------------------
# Input conditions for FP scatter (per task)
# ---------------------------------------------------------------------------

def _input_conditions(task: str, input_size: int, cue_on_go_input: bool) -> list:
    """(label, active_dims, color, marker)"""
    base = [("Autonomous", None, "black", "o")]
    if task in ("dpa", "dual"):
        base += [
            ("Sample A", [0], "tab:blue",   "s"),
            ("Sample B", [1], "tab:orange", "D"),
            ("Test C",   [2], "tab:green",  "^"),
            ("Test D",   [3], "tab:purple", "v"),
        ]
    if task in ("gng", "dual"):
        base += [
            ("Go",   [4], TRIAL_COLORS["go"][0],   "P"),
            ("NoGo", [5], TRIAL_COLORS["nogo"][0], "h"),
        ]
        if not cue_on_go_input:
            base.append(("Cue", [6], "tab:pink", "*"))
    return base


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_dual_by_trialtype(model, meta: RunMeta, device: str,
                             n_trials: int = 1024,
                             per_trial: bool = False) -> dict:
    """Accuracy for DPA-only / Go / NoGo subsets of dual trials.

    per_trial=False (default): returns scalar mean per key.
    per_trial=True: returns numpy array of per-trial 0/1 outcomes per key,
                    for computing SEM = std/sqrt(n) across individual trials.
    """
    model.eval()
    timing = TIMINGS["dual"]
    X, y, _, cnames = generate_dual_trials(
        n_trials, timing=timing, input_size=meta.input_size,
        noise=meta.noise_sigma(), target_rank=2,
        cue_on_go_input=meta.cue_on_go_input, cue_scale=meta.cue_scale,
        nogo_target=meta.nogo_target,
    )
    X, y = X.to(device), y.to(device)
    pred  = model(X, y)[..., -1].cpu()
    y_cpu = y.cpu()
    names = np.asarray(cnames).astype(str)

    dpa_start = int(timing.n_stim_off[3])
    rwd_start = int(timing.n_stim_off[2])
    rwd_stop  = int(timing.n_stim_on[3])

    pred_dpa = pred[:, dpa_start:].mean(1)
    pred_gng = pred[:, rwd_start:rwd_stop].mean(1)

    is_dpa_t  = torch.as_tensor(["_go_" not in n and "_nogo_" not in n for n in names])
    is_go_t   = torch.as_tensor(["_go_"   in n for n in names])
    is_nogo_t = torch.as_tensor(["_nogo_" in n for n in names])

    out = {}
    for mask, key in [(is_dpa_t, "dpa_only"), (is_go_t, "go"), (is_nogo_t, "nogo")]:
        if mask.any():
            correct = ((pred_dpa[mask] > 0) == (y_cpu[mask, -1, -1] > 0)).float()
            out[f"{key}_dpa"] = correct.numpy() if per_trial else correct.mean().item()

    if is_go_t.any():
        correct = (pred_gng[is_go_t] > 0.5).float()
        out["go_gng"]   = correct.numpy() if per_trial else correct.mean().item()
    if is_nogo_t.any():
        correct = (1.0 - (pred_gng[is_nogo_t] > 0.5).float())
        out["nogo_gng"] = correct.numpy() if per_trial else correct.mean().item()
    return out


@torch.no_grad()
def _eval_gng_by_trialtype(model, meta: RunMeta, device: str,
                            n_trials: int = 1024) -> dict:
    """GNG accuracy split by Go / NoGo trial type."""
    model.eval()
    timing = TIMINGS["gng"]
    X, y = generate_gng_trials(
        n_trials, timing=timing, input_size=meta.input_size,
        noise=meta.noise_sigma(), target_rank=2,
        cue_on_go_input=meta.cue_on_go_input, cue_scale=meta.cue_scale,
        nogo_target=meta.nogo_target,
    )
    X, y = X.to(device), y.to(device)
    pred  = model(X, y)[..., -1].cpu()
    X_cpu = X.cpu()

    stim_epoch = slice(int(timing.n_stim_on[0]), int(timing.n_stim_off[0]))
    is_go      = X_cpu[:, stim_epoch, 4:6].mean(1).argmax(1) == 0
    decision_t = int(timing.n_stim_off[1])
    pred_final = pred[:, decision_t:].mean(1)

    return {
        "go":   (pred_final[is_go]  > 0.5).float().mean().item(),
        "nogo": (1.0 - (pred_final[~is_go] > 0.5).float()).mean().item(),
    }


# ---------------------------------------------------------------------------
# Shared kappa trajectory computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_kappa(model, X_t: torch.Tensor, y_t: torch.Tensor, device: str,
                   model_noise: float = 0.0) -> np.ndarray:
    model.eval()
    model.noise = model_noise
    readout, _, _ = model(X_t.to(device), targets=y_t.to(device), ret_rates=True)
    return readout.detach().cpu().numpy()


def _make_gng_batch(ref_meta: RunMeta, n_batch: int = 512, noise: float | None = None,
                    seed: int = 0) -> tuple:
    """Returns (X_t, y_t, is_go) for a pure GNG trial batch.
    is_go derived from targets: go trials have decision target == 1.0 after the cue."""
    rng_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)

    timing  = TIMINGS["gng"]
    n_sigma = _noise_sigma(ref_meta.noise) if noise is None else noise
    X, y = generate_gng_trials(
        n_batch, timing=timing, input_size=ref_meta.input_size,
        noise=n_sigma, target_rank=2, cue_on_go_input=ref_meta.cue_on_go_input,
        cue_scale=ref_meta.cue_scale, nogo_target=ref_meta.nogo_target,
    )
    np.random.set_state(rng_state)
    torch.set_rng_state(torch_state)
    # go trials have decision target == 1.0 after the cue (last non-nan value)
    is_go = (y[:, -1, -1] > 0.5).numpy()
    return (
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        is_go,
    )


def _make_dual_batch(ref_meta: RunMeta, n_batch: int = 512, noise: float | None = None,
                     seed: int = 0) -> tuple:
    rng_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)

    timing = TIMINGS["dual"]
    n_sigma = _noise_sigma(ref_meta.noise) if noise is None else noise
    X, y, _, cnames = generate_dual_trials(
        n_batch, timing=timing, input_size=ref_meta.input_size,
        noise=n_sigma, target_rank=2, cue_on_go_input=ref_meta.cue_on_go_input,
        cue_scale=ref_meta.cue_scale, nogo_target=ref_meta.nogo_target,
    )
    np.random.set_state(rng_state)
    torch.set_rng_state(torch_state)
    return (
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        np.asarray(cnames).astype(str),
    )


# ---------------------------------------------------------------------------
# FP helpers
# ---------------------------------------------------------------------------

def _fps_for_task(model, input_size: int, task: str, device: str,
                  cue_on_go_input: bool, n_seeds: int = 21, slow_tol=None) -> list:
    """[(label, color, marker, fps_array, stabs_array), ...]"""
    dtype  = next(model.parameters()).dtype
    conds  = _input_conditions(task, input_size, cue_on_go_input)
    result = []
    for label, dims, color, marker in conds:
        ff       = make_input(input_size, active_dims=dims, value=1.0,
                              device=device, dtype=dtype)
        fps, _   = find_all_fixed_points(model, xlim=XLIM, ylim=YLIM, ff_input=ff,
                                         n_seeds=n_seeds, residual_tol=1e-8, merge_tol=5e-2)
        stabs, _ = classify_fixed_points(model, fps, ff_input=ff, slow_tol=slow_tol)
        result.append((label, color, marker, fps, stabs))
    return result


def _model_has_backbone(model) -> bool:
    """True if the analytic low-rank FP reduction is invalid (static backbone / EI / EI+STP).
    Such models route to the simulation-based fixed-point finder (_sim_fps_for_conditions)
    instead of the analytic find_all_fixed_points (which assumes a single-timescale LowRankModel)."""
    return (getattr(model, "w_fixed", None) is not None
            or hasattr(model, "_W_rec_eff")
            or model.__class__.__name__ == "EISTPModel")


def _scatter_lim(meta) -> tuple:
    """Axis limits for FP scatters: eistp κ runs ~±10 → use ±15, else the global XLIM."""
    return (-15.0, 15.0) if getattr(meta, "model_type", None) == "eistp" else XLIM


def _sim_fps_for_conditions(model, input_size, conditions, device,
                            R=3.0, gsize=18, k=4, merge_tol=0.25):
    """True (simulation-based) attractors per input condition, via the grid-sim used by
    the binned flow (ei_flow.run_grid) + KMeans on trajectory endpoints. For models whose
    analytic κ-reduction is invalid (fixed-weight backbone, EI). All returned points are
    'attractor' (the sim only reveals attractors)."""
    import ei_flow
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # EISTPModel lives on a much wider κ range (~±10) with two-timescale + STP dynamics →
    # wider grid, shorter T and FP tolerances scaled to that κ scale (mirrors ei_flow.make_stage_figure).
    T = 1333; fp_kw = {}
    if model.__class__.__name__ == "EISTPModel":
        R, gsize, T = 15.0, 28, 600
        fp_kw = dict(vel_tol=0.03, recent=12, link_tol=2.0, point_tol=4.0, min_pts=10)
    out = []
    for label, dims, color, marker in conditions:
        cond_x = None if dims is None else make_input(input_size, dims, 1.0, device=dev, dtype=dtype)
        traj, _ = ei_flow.run_grid(model, cond_x, R=R, gsize=gsize, set_w=40, T=T, I0=1.0, device=dev)
        pts, man, _ = ei_flow.fixed_points(traj, **fp_kw)   # isolated point fps + continuous-manifold locus
        fps   = list(pts)
        stabs = ["attractor"] * len(pts)
        if man is not None and len(man):           # continuous attractor → subsampled gold locus
            sub = man[:: max(1, len(man) // 60)]
            fps.extend(sub); stabs.extend(["marginal"] * len(sub))
        fps = np.array(fps) if fps else np.empty((0, 2))
        out.append((label, color, marker, fps, stabs))
    return out


def _fp_marker_style(stab, color):
    """(facecolor, edgecolor, size, linewidth) for a fixed-point stability label.
    slow_attractor → orange ring, marginal → gold ring (matches the flow plots)."""
    if stab == "attractor":        return color,   color,   90, 1.0
    if stab == "slow_attractor":   return color,   "orange", 95, 2.0
    if stab == "marginal":         return "none",  "gold",   85, 2.0
    if stab == "saddle":           return "white", color,    70, 1.4
    return "white", color, 50, 0.8   # repeller / nonhyperbolic


# ---------------------------------------------------------------------------
# Trajectory figure helpers (κ vs time)
# ---------------------------------------------------------------------------

GNG_SPECS = {
    "dpa":  ("DPA-only trials", "none"),
    "go":   ("Go trials",       "go"),
    "nogo": ("NoGo trials",     "nogo"),
}


def _concrete_conds(pair_status: str, gng_status: str) -> list[str]:
    base = [("A", "C"), ("B", "D")] if pair_status == "pair" else [("A", "D"), ("B", "C")]
    if gng_status == "none":
        return [f"{s}_{t}" for s, t in base]
    return [f"{s}_{gng_status}_{t}" for s, t in base]


def _plot_traj_figure(kappa_dict: dict[str, np.ndarray],
                      targets_np: np.ndarray,
                      cnames: np.ndarray,
                      timing: TaskTiming,
                      gng_status: str,
                      title: str) -> plt.Figure:
    """
    One figure for a GNG condition, one column-pair per group in kappa_dict.
    Rows: pair / unpair.  Cols (per group): κ₁ / κ₂.
    """
    groups   = list(kappa_dict.keys())
    n_groups = len(groups)
    n_dims   = 2
    n_cols   = n_dims * n_groups

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(n_dims * n_groups * 5.0, 2 * 3.1),
        sharex=True,
        constrained_layout=True,
    )
    if n_groups == 1 and n_cols == 2:
        axes = axes.reshape(2, 2)
    fig.suptitle(title, fontsize=13)

    t = np.arange(targets_np.shape[1]) * timing.dt

    for gi, (group_name, kappa_np) in enumerate(kappa_dict.items()):
        for row, (row_lbl, pair_st) in enumerate([("Paired", "pair"), ("Unpaired", "unpair")]):
            conds = _concrete_conds(pair_st, gng_status)
            for col in range(n_dims):
                ax   = axes[row, gi * n_dims + col]
                k_lim= np.nanpercentile(np.abs(kappa_np[:, :, col]), 99)
                ylim = 1.1 * max(float(k_lim), 1.5)
                ax.set_ylim(-ylim, ylim)

                for on, off in zip(timing.stim_on, timing.stim_off):
                    ax.axvspan(on, off, alpha=0.10, color="tab:blue", lw=0, zorder=0)

                ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.35, zorder=1)

                palette = TRAJ_PALETTE.get(gng_status, TRAJ_PALETTE["none"])
                for ci, cond in enumerate(conds):
                    idxs = np.where(cnames == cond)[0]
                    if not len(idxs):
                        continue
                    color    = palette[ci % len(palette)]
                    k_trials = kappa_np[idxs, :, col]
                    k_mean   = k_trials.mean(0)
                    k_std    = k_trials.std(0, ddof=1)
                    tgt_dim  = col if col < targets_np.shape[-1] else -1
                    tgt_mean = np.nanmean(targets_np[idxs, :, tgt_dim], axis=0)
                    ax.fill_between(t, k_mean - k_std, k_mean + k_std,
                                    color=color, alpha=0.25, lw=0, zorder=8)
                    if np.any(np.isfinite(tgt_mean[int(timing.n_stim_on[0]):])):
                        ax.plot(t, tgt_mean, color="k", lw=2.0, alpha=0.8, zorder=9)
                    ax.plot(t, k_mean,   color=color, lw=2.5, alpha=0.9, zorder=10)

                if row == 0:
                    ax.set_title(f"{group_name} — κ{col}", fontsize=10)
                if row == 1:
                    ax.set_xlabel("Time (s)")
                if gi == 0 and col == 0:
                    ax.set_ylabel(f"κ{col}\n{row_lbl}", fontsize=9)
                elif col == 0:
                    ax.set_ylabel(row_lbl, fontsize=9)

    palette = TRAJ_PALETTE.get(gng_status, TRAJ_PALETTE["none"])
    legend_handles = [
        mlines.Line2D([], [], color="k",        lw=2.0, label="Target"),
        mlines.Line2D([], [], color=palette[0], lw=2.5, label="A-sample"),
        mlines.Line2D([], [], color=palette[1], lw=2.5, label="B-sample"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.04))
    return fig


def _plot_gng_traj_figure(kappa_dict: dict[str, np.ndarray],
                          targets_np: np.ndarray,
                          is_go: np.ndarray,
                          timing: TaskTiming,
                          title: str) -> plt.Figure:
    """GNG trajectory figure: rows = Go / NoGo, cols (per group) = κ₁ / κ₂."""
    groups   = list(kappa_dict.keys())
    n_groups = len(groups)
    n_dims   = 2
    n_cols   = n_dims * n_groups

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(n_dims * n_groups * 5.0, 2 * 3.1),
        sharex=True,
        constrained_layout=True,
    )
    if n_groups == 1 and n_cols == 2:
        axes = axes.reshape(2, 2)
    fig.suptitle(title, fontsize=13)

    t = np.arange(targets_np.shape[1]) * timing.dt
    row_specs = [("Go", is_go, TRIAL_COLORS["go"][0]),
                 ("NoGo", ~is_go, TRIAL_COLORS["nogo"][0])]

    for gi, (group_name, kappa_np) in enumerate(kappa_dict.items()):
        for row, (row_lbl, mask, color) in enumerate(row_specs):
            for col in range(n_dims):
                ax = axes[row, gi * n_dims + col]
                k_lim = np.nanpercentile(np.abs(kappa_np[:, :, col]), 99)
                ylim  = 1.1 * max(float(k_lim), 1.5)
                ax.set_ylim(-ylim, ylim)

                for on, off in zip(timing.stim_on, timing.stim_off):
                    ax.axvspan(on, off, alpha=0.10, color="tab:blue", lw=0, zorder=0)
                ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.35, zorder=1)

                if mask.any():
                    k_trials = kappa_np[mask, :, col]
                    k_mean   = k_trials.mean(0)
                    k_std    = k_trials.std(0, ddof=1)
                    tgt_dim  = col if col < targets_np.shape[-1] else -1
                    tgt_mean = np.nanmean(targets_np[mask, :, tgt_dim], axis=0)
                    ax.fill_between(t, k_mean - k_std, k_mean + k_std,
                                    color=color, alpha=0.25, lw=0, zorder=8)
                    if np.any(np.isfinite(tgt_mean[int(timing.n_stim_on[0]):])):
                        ax.plot(t, tgt_mean, color="k", lw=2.0, alpha=0.8, zorder=9)
                    ax.plot(t, k_mean,   color=color, lw=2.5, alpha=0.9, zorder=10)

                if row == 0:
                    ax.set_title(f"{group_name} — κ{col}", fontsize=10)
                if row == 1:
                    ax.set_xlabel("Time (s)")
                if gi == 0 and col == 0:
                    ax.set_ylabel(f"κ{col}\n{row_lbl}", fontsize=9)
                elif col == 0:
                    ax.set_ylabel(row_lbl, fontsize=9)

    fig.legend(
        handles=[mlines.Line2D([], [], color="k", lw=2.0, label="Target"),
                 mlines.Line2D([], [], color=TRIAL_COLORS["go"][0],   lw=2.5, label="Go"),
                 mlines.Line2D([], [], color=TRIAL_COLORS["nogo"][0], lw=2.5, label="NoGo")],
        loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.04),
    )
    return fig


# ============================================================
# SUMMARY FIGURES
# ============================================================

# --- 1. Accuracy stages ---

def summary_accuracy_stages(df, out_dir: str):
    from plots.plot import plot_accuracy_stages
    plot_accuracy_stages(df, group_by="init_style",
                         out_path=os.path.join(out_dir, "accuracy_stages.pdf"))
    plt.close("all")


# --- 2. Accuracy by trial type ---
# Same format as accuracy_stages: x = training stage, one line per trial type.
# Panel 1 — DPA accuracy:  lines = DPA-only / Go / NoGo
# Panel 2 — GNG accuracy:  lines = Go / NoGo

TRIALTYPE_COLORS = {
    "dpa_only": TRIAL_COLORS["dpa"][0],
    "go":       TRIAL_COLORS["go"][0],
    "nogo":     TRIAL_COLORS["nogo"][0],
}
TRIALTYPE_LABELS = {"dpa_only": "DPA-only", "go": "Go", "nogo": "NoGo"}


def _collect_trialtype_accuracy(all_metas: list[RunMeta], ckpt_dir: str,
                                device: str, per_trial: bool = False) -> dict:
    """
    Returns nested dict where each cell is {"means": [...], "sem": float|None}.

    per_trial=False (sweep mode):
        means = one scalar per run; sem = None (plot computes std/sqrt(n) across runs).
    per_trial=True (individual mode):
        means = [single mean]; sem = binomial SEM = sqrt(p*(1-p)/N) across trials.
        Scatter is suppressed — only the error bar is shown.
    """
    stages      = ["dpa", "naive", "expert"]
    dpa_types   = ["dpa_only", "go", "nogo"]
    gng_types   = ["go", "nogo"]

    def _cell_dict(keys):
        return {t: {s: {"means": [], "sem": None} for s in stages} for t in keys}

    dpa_data = _cell_dict(dpa_types)
    gng_data = _cell_dict(gng_types)

    dpa_key_map = {"dpa_only": "dpa_only_dpa", "go": "go_dpa",  "nogo": "nogo_dpa"}
    gng_key_map = {"go":       "go_gng",        "nogo": "nogo_gng"}

    def _store(cell, v):
        if per_trial and isinstance(v, np.ndarray):
            n = len(v)
            p = float(v.mean()) if n else float("nan")
            cell["means"] = [p]
            cell["sem"]   = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
        else:
            cell["means"].append(float(v))

    for meta in all_metas:
        model = _build_model(meta, device)
        for stage in stages:
            if not _load_ckpt(model, ckpt_dir, stage, meta.run_id, device):
                for t in dpa_types: dpa_data[t][stage]["means"].append(float("nan"))
                for t in gng_types: gng_data[t][stage]["means"].append(float("nan"))
                continue
            ev = _eval_dual_by_trialtype(model, meta, device, per_trial=per_trial)
            for t, key in dpa_key_map.items():
                _store(dpa_data[t][stage], ev.get(key, float("nan")))
            for t, key in gng_key_map.items():
                _store(gng_data[t][stage], ev.get(key, float("nan")))
        del model
        print(f"  trialtype acc: {meta.run_id}")

    return {"dpa_acc": dpa_data, "gng_acc": gng_data}


def _plot_trialtype_accuracy(data: dict, title: str, out_path: str):
    """
    2-panel figure (DPA acc / GNG acc), x = stages, lines = trial types.
    Mirrors the accuracy_stages style.
    """
    stages       = ["dpa", "naive", "expert"]
    dpa_types    = ["dpa_only", "go", "nogo"]
    gng_types    = ["go", "nogo"]
    n_dpa = len(dpa_types)
    n_gng = len(gng_types)
    offsets_dpa  = np.linspace(-0.18, 0.18, n_dpa)
    offsets_gng  = np.linspace(-0.12, 0.12, n_gng)
    rng          = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(2.2 * 6, 6 * GOLDEN), sharey=True)

    for ax, (panel_types, offsets, panel_data, panel_title) in zip(axes, [
        (dpa_types, offsets_dpa, data["dpa_acc"], "DPA accuracy"),
        (gng_types, offsets_gng, data["gng_acc"], "GNG accuracy"),
    ]):
        for ti, tt in enumerate(panel_types):
            color = TRIALTYPE_COLORS[tt]
            xs, means, sems = [], [], []
            for si, stage in enumerate(stages):
                cell  = panel_data[tt][stage]
                raw   = cell["means"]
                vals  = [v for v in raw if not np.isnan(v)]
                x     = STAGE_X[si] + offsets[ti]
                xs.append(x)
                if cell["sem"] is not None:
                    # individual-net mode: precomputed binomial SEM, no scatter
                    mean = vals[0] if vals else float("nan")
                    sem  = cell["sem"]
                else:
                    # sweep mode: scatter one dot per run, cross-run SEM
                    if vals:
                        jitter = rng.normal(0, 0.03, len(vals))
                        ax.scatter(np.full(len(vals), x) + jitter, vals,
                                   s=22, color=color, alpha=0.3, linewidths=0, zorder=2)
                    mean = float(np.mean(vals)) if vals else float("nan")
                    sem  = (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0)
                means.append(mean)
                sems.append(sem)

            means_arr = np.array(means)
            sems_arr  = np.array(sems)
            ax.plot(xs, means_arr, color=color, lw=2.0, alpha=0.85, zorder=3)
            ax.errorbar(xs, means_arr, yerr=sems_arr, fmt="o", ms=9,
                        mfc=color, mec="white", mew=1.6,
                        ecolor=color, elinewidth=2.0, capsize=5, capthick=2.0,
                        zorder=4, label=TRIALTYPE_LABELS[tt])

        ax.axhline(0.5, color="0.6", ls="--", lw=1.2, zorder=1)
        ax.set_title(panel_title, pad=10)
        ax.set_xticks(STAGE_X)
        ax.set_xticklabels(STAGE_LABELS)
        ax.set_xlim(-0.5, len(STAGE_LABELS) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Checkpoint")
        ax.grid(axis="y", color="0.92", lw=1.0)
        ax.set_axisbelow(True)
        ax.legend(title="Trial type", frameon=False, fontsize=9)

    axes[0].set_ylabel("Accuracy")
    sns.despine(fig=fig, trim=True)
    fig.suptitle(title, y=1.02)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def summary_accuracy_by_trialtype(all_metas: list[RunMeta], ckpt_dir: str,
                                   out_dir: str, device: str):
    data = _collect_trialtype_accuracy(all_metas, ckpt_dir, device)
    _plot_trialtype_accuracy(
        data,
        title    = "Accuracy by trial type (all runs)",
        out_path = os.path.join(out_dir, "accuracy_by_trialtype.pdf"),
    )


# --- 3+4. FP scatter figures (by stage, all runs) ---
# One collection pass loads each (run, stage) checkpoint once and computes
# fixed points for every input condition.  From that we render:
#   fp_scatter_by_stage.pdf          — autonomous FPs, 3 stage panels
#   fp_scatter_by_input_{slug}.pdf   — one by-stage figure per input-driven cond
STAGES = ["dpa", "naive", "expert"]


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def _draw_fp_points(ax, init_style: str, fps, stabs):
    """Scatter one run's fixed points onto ax (color = init_style, fill = stability).
    slow_attractor → orange edge, marginal → gold hollow (matches flow plots)."""
    rc = STYLE_COLORS.get(init_style, "gray")
    for fp, stab in zip(fps, stabs):
        if stab == "attractor":
            fc, ec, sz, lw = rc, rc, 60, 0.8
        elif stab == "slow_attractor":
            fc, ec, sz, lw = rc, "orange", 64, 1.8
        elif stab == "marginal":
            fc, ec, sz, lw = "none", "gold", 58, 1.8
        elif stab == "saddle":
            fc, ec, sz, lw = "white", rc, 45, 1.2
        else:
            fc, ec, sz, lw = "white", rc, 35, 0.7
        ax.scatter(fp[0], fp[1], s=sz, marker="o",
                   facecolors=fc, edgecolors=ec, linewidths=lw, zorder=5, alpha=0.6)


def _style_fp_ax(ax, title: str, ylabel: bool, lim: tuple = None):
    lim = lim if lim is not None else XLIM
    ax.axhline(0, color="lightgray", lw=0.7, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.7, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\kappa_0$", fontsize=9)
    ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(r"$\kappa_1$", fontsize=9)


def _fp_legend(fig, all_metas):
    patches = [mpatches.Patch(color=STYLE_COLORS.get(s, "gray"), label=s)
               for s in ["structured", "random"]
               if any(m.init_style == s for m in all_metas)]
    stab_handles = [
        plt.scatter([], [], marker="o", s=60, facecolors="gray",  edgecolors="gray",  label="attractor"),
        plt.scatter([], [], marker="o", s=60, facecolors="gray",  edgecolors="orange", linewidths=1.8, label="slow attractor"),
        plt.scatter([], [], marker="o", s=60, facecolors="none",  edgecolors="gold",   linewidths=1.8, label="marginal"),
        plt.scatter([], [], marker="o", s=45, facecolors="white", edgecolors="gray",  linewidths=1.2, label="saddle"),
        plt.scatter([], [], marker="o", s=35, facecolors="white", edgecolors="gray",  linewidths=0.7, label="repeller"),
    ]
    fig.legend(handles=patches + stab_handles, loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))


def _collect_fp_scatter(all_metas, ckpt_dir, device, conditions, n_seeds, slow_tol=None):
    """data[label][stage] = [(init_style, fps, stabs), ...] across all runs."""
    data = {label: {s: [] for s in STAGES} for label, *_ in conditions}
    for meta in all_metas:
        for stage in STAGES:
            model = _build_model(meta, device)
            if not _load_ckpt(model, ckpt_dir, stage, meta.run_id, device):
                del model; continue
            if _model_has_backbone(model):
                # true (simulation-based) attractors — analytic reduction ignores the backbone
                for label, _c, _m, fps, stabs in _sim_fps_for_conditions(model, meta.input_size, conditions, device):
                    data[label][stage].append((meta.init_style, fps, stabs))
            else:
                dtype = next(model.parameters()).dtype
                for label, dims, _color, _marker in conditions:
                    ff = make_input(meta.input_size, active_dims=dims, value=1.0,
                                    device=device, dtype=dtype)
                    fps, _   = find_all_fixed_points(model, xlim=XLIM, ylim=YLIM, ff_input=ff,
                                                     n_seeds=n_seeds, residual_tol=1e-8, merge_tol=5e-2)
                    stabs, _ = classify_fixed_points(model, fps, ff_input=ff, slow_tol=slow_tol)
                    data[label][stage].append((meta.init_style, fps, stabs))
            del model
        print(f"  fp_scatter collected: {meta.run_id}")
    return data


def _render_fp_by_stage(data_for_label, all_metas, title, out_path, lim=None):
    """3-panel (stage) figure for a single input condition."""
    fig, axes = plt.subplots(1, len(STAGES), figsize=(13, 4.5), constrained_layout=True)
    for ax, stage in zip(axes, STAGES):
        for init_style, fps, stabs in data_for_label[stage]:
            _draw_fp_points(ax, init_style, fps, stabs)
        _style_fp_ax(ax, f"{stage} ({STAGE_TASK[stage].upper()})", ylabel=(ax is axes[0]), lim=lim)
    _fp_legend(fig, all_metas)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def summary_fp_scatters(all_metas: list[RunMeta], ckpt_dir: str,
                        out_dir: str, device: str, n_seeds: int = 21, slow_tol=None):
    ref        = all_metas[0]
    lim        = _scatter_lim(ref)
    conditions = _input_conditions("dual", ref.input_size, ref.cue_on_go_input)
    data       = _collect_fp_scatter(all_metas, ckpt_dir, device, conditions, n_seeds, slow_tol=slow_tol)

    # Autonomous → fp_scatter_by_stage.pdf
    _render_fp_by_stage(
        data["Autonomous"], all_metas,
        "Autonomous fixed points across stages — all runs",
        os.path.join(out_dir, "fp_scatter_by_stage.pdf"), lim=lim,
    )

    # Each input-driven condition → one by-stage figure
    for label, dims, _color, _marker in conditions:
        if label == "Autonomous":
            continue
        _render_fp_by_stage(
            data[label], all_metas,
            f"Input-driven fixed points across stages — {label} (all runs)",
            os.path.join(out_dir, f"fp_scatter_by_input_{_slug(label)}.pdf"), lim=lim,
        )


# --- 5. Average trajectories (κ vs time), grouped by init_style ---

def summary_avg_trajectories(all_metas: list[RunMeta], ckpt_dir: str,
                              out_dir: str, device: str, n_batch: int = 512):
    ref = all_metas[0]
    X_dual, y_dual, cnames = _make_dual_batch(ref, n_batch=n_batch, seed=0)
    X_gng,  y_gng,  is_go  = _make_gng_batch(ref,  n_batch=n_batch, seed=0)

    groups: dict[str, list[RunMeta]] = {}
    for m in all_metas:
        groups.setdefault(m.init_style, []).append(m)

    stages       = ["dpa", "naive", "expert"]
    stage_labels = {"dpa": "DPA stage", "naive": "After GNG", "expert": "After Dual"}

    for stage in stages:
        group_kappas_dual: dict[str, np.ndarray | None] = {}
        group_kappas_gng:  dict[str, np.ndarray | None] = {}
        for group_name, metas in groups.items():
            sum_dual = sum_gng = None
            count = 0
            for meta in metas:
                model = _build_model(meta, device)
                if not _load_ckpt(model, ckpt_dir, stage, meta.run_id, device):
                    del model; continue
                k_dual = _compute_kappa(model, X_dual, y_dual, device, meta.model_noise_sigma())
                k_gng  = _compute_kappa(model, X_gng,  y_gng,  device, meta.model_noise_sigma())
                del model
                sum_dual = k_dual if sum_dual is None else sum_dual + k_dual
                sum_gng  = k_gng  if sum_gng  is None else sum_gng  + k_gng
                count += 1
            if count:
                group_kappas_dual[group_name] = sum_dual / count
                group_kappas_gng[group_name]  = sum_gng  / count
        if not group_kappas_dual:
            continue

        for gng_key, (gng_title, gng_status) in GNG_SPECS.items():
            fig = _plot_traj_figure(
                group_kappas_dual, y_dual.numpy(), cnames, TIMINGS["dual"],
                gng_status,
                f"{stage_labels[stage]} — {gng_title} (mean across runs)",
            )
            out_path = os.path.join(out_dir, f"traj_{stage}_{gng_key}.pdf")
            fig.savefig(out_path, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.close(fig)

        fig = _plot_gng_traj_figure(
            group_kappas_gng, y_gng.numpy(), is_go, TIMINGS["gng"],
            f"{stage_labels[stage]} — GNG task (mean across runs)",
        )
        out_path = os.path.join(out_dir, f"traj_{stage}_gng_task.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)


# ============================================================
# INDIVIDUAL FIGURES
# ============================================================


def individual_accuracy_stages(meta: RunMeta, out_dir: str):
    acc = meta.accuracy
    dpa_vals = []
    gng_vals = []
    for sk in ["after_dpa", "after_gng", "after_dual"]:
        a = acc.get(sk, {})
        if sk == "after_dual":
            dpa_vals.append(a.get("dual_dpa", a.get("dpa", float("nan"))))
            gng_vals.append(a.get("dual_gng", a.get("gng", float("nan"))))
        else:
            dpa_vals.append(a.get("dpa", float("nan")))
            gng_vals.append(a.get("gng", float("nan")))

    color = STYLE_COLORS.get(meta.init_style, "gray")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, vals, title in zip(axes, [dpa_vals, gng_vals], ["DPA accuracy", "GNG accuracy"]):
        clean = [v if not np.isnan(v) else 0.0 for v in vals]
        bars  = ax.bar(STAGE_X, clean, color=color, alpha=0.75)
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                bar.set_alpha(0.15)
        ax.axhline(0.5, color="0.6", ls="--", lw=1.2)
        ax.set_xticks(STAGE_X); ax.set_xticklabels(STAGE_LABELS)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", color="0.92", lw=1.0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy")
    sns.despine(fig=fig, trim=True)
    fig.suptitle(meta.run_id, fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "accuracy_stages.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def individual_accuracy_by_trialtype(meta: RunMeta, ckpt_dir: str,
                                      out_dir: str, device: str):
    # per_trial=True: SEM is computed across individual trial outcomes
    data = _collect_trialtype_accuracy([meta], ckpt_dir, device, per_trial=True)
    _plot_trialtype_accuracy(
        data,
        title    = f"{meta.run_id} — accuracy by trial type",
        out_path = os.path.join(out_dir, "accuracy_by_trialtype.pdf"),
    )


def individual_trajectories(meta: RunMeta, ckpt_dir: str, out_dir: str, device: str,
                             n_batch: int = 512):
    X_dual, y_dual, cnames = _make_dual_batch(meta, n_batch=n_batch, seed=0)
    X_gng,  y_gng,  is_go  = _make_gng_batch(meta,  n_batch=n_batch, seed=0)
    stages       = ["dpa", "naive", "expert"]
    stage_labels = {"dpa": "DPA stage", "naive": "After GNG", "expert": "After Dual"}

    for stage in stages:
        model = _build_model(meta, device)
        if not _load_ckpt(model, ckpt_dir, stage, meta.run_id, device):
            del model; continue
        kappa_dual = _compute_kappa(model, X_dual, y_dual, device, meta.model_noise_sigma())
        kappa_gng  = _compute_kappa(model, X_gng,  y_gng,  device, meta.model_noise_sigma())
        del model

        for gng_key, (gng_title, gng_status) in GNG_SPECS.items():
            fig = _plot_traj_figure(
                {meta.run_id: kappa_dual}, y_dual.numpy(), cnames, TIMINGS["dual"],
                gng_status, f"{stage_labels[stage]} — {gng_title}",
            )
            out_path = os.path.join(out_dir, f"traj_{stage}_{gng_key}.pdf")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        fig = _plot_gng_traj_figure(
            {meta.run_id: kappa_gng}, y_gng.numpy(), is_go, TIMINGS["gng"],
            f"{stage_labels[stage]} — GNG task",
        )
        out_path = os.path.join(out_dir, f"traj_{stage}_gng_task.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"  trajectories saved: {out_dir}")


def individual_fp_scatter(meta: RunMeta, ckpt_dir: str, out_dir: str,
                          device: str, n_seeds: int = 41, slow_tol=None):
    scatter_dir = os.path.join(out_dir, "scatter")
    os.makedirs(scatter_dir, exist_ok=True)
    cue = meta.cue_on_go_input

    stages = ["dpa", "naive", "expert"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)

    for ax, stage in zip(axes, stages):
        task = STAGE_TASK[stage]
        model = _build_model(meta, device)
        if not _load_ckpt(model, ckpt_dir, stage, meta.run_id, device):
            ax.set_visible(False); del model; continue

        if _model_has_backbone(model):
            fp_data = _sim_fps_for_conditions(model, meta.input_size,
                                              _input_conditions(task, meta.input_size, cue), device)
        else:
            fp_data = _fps_for_task(model, meta.input_size, task, device, cue, n_seeds, slow_tol=slow_tol)
        del model

        ax.axhline(0, color="lightgray", lw=0.7, zorder=0)
        ax.axvline(0, color="lightgray", lw=0.7, zorder=0)
        for label, color, marker, fps, stabs in fp_data:
            for fp, stab in zip(fps, stabs):
                fc, ec, sz, lw = _fp_marker_style(stab, color)
                ax.scatter(fp[0], fp[1], marker=marker, s=sz,
                           facecolors=fc, edgecolors=ec, linewidths=lw, zorder=5)
        _lim = _scatter_lim(meta)
        ax.set_xlim(_lim); ax.set_ylim(_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\kappa_0$", fontsize=8)
        ax.set_title(f"{stage} ({task.upper()})", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(r"$\kappa_1$", fontsize=8)

    # legend: one patch per input condition (union of dual conditions)
    conds_legend = _input_conditions("dual", meta.input_size, cue)
    cond_handles = [mpatches.Patch(color=c, label=l) for l, _, c, _ in conds_legend]
    stab_handles = [
        plt.scatter([], [], marker="o", s=70, facecolors="gray",  edgecolors="gray",  label="attractor"),
        plt.scatter([], [], marker="o", s=70, facecolors="gray",  edgecolors="orange", linewidths=2.0, label="slow attractor"),
        plt.scatter([], [], marker="o", s=70, facecolors="none",  edgecolors="gold",   linewidths=2.0, label="marginal"),
        plt.scatter([], [], marker="o", s=70, facecolors="white", edgecolors="gray",  linewidths=1.5, label="saddle"),
        plt.scatter([], [], marker="o", s=50, facecolors="white", edgecolors="gray",  linewidths=1.0, label="repeller"),
    ]
    a = meta.accuracy.get("after_dual", {})
    fig.suptitle(f"{meta.run_id}   dual_dpa={a.get('dual_dpa', float('nan')):.3f}  "
                 f"dual_gng={a.get('dual_gng', float('nan')):.3f}", fontsize=9)
    fig.legend(handles=cond_handles + stab_handles, loc="lower center",
               ncol=6, fontsize=6, bbox_to_anchor=(0.5, -0.08), frameon=False)

    out_path = os.path.join(scatter_dir, "fp_scatter.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  scatter saved: {out_path}")
    plt.close(fig)


def individual_flow(meta: RunMeta, ckpt_dir: str, out_dir: str, device: str,
                    n_batch: int = 256, n_fp_seeds: int = 21,
                    use_sim_field: bool = False, sim_n_warmup: int = 0,
                    auto_xlim: bool = False, n_grid: int = 151, slow_tol=None,
                    show_slow_manifold: bool = False, slow_manifold_thresh: float = 0.12):
    flow_dir = os.path.join(out_dir, "flow")
    os.makedirs(flow_dir, exist_ok=True)
    cue = meta.cue_on_go_input

    stage_task = {"dpa": "dpa", "naive": "gng", "expert": "dual"}

    # EI+STP models have no analytic κ-reduction → the analytic flow (plot_task_flow_fields)
    # is invalid. Delegate to the simulation-based binned flow (ei_flow), exactly as the
    # standalone ei_flow.py entrypoint does, writing the same flow/fp_{stage}.pdf paths.
    if meta.model_type == "eistp":
        import ei_flow
        for stage, task in stage_task.items():
            ckpt = os.path.join(ckpt_dir, meta.run_id, f"{stage}_{meta.run_id}.pth")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(ckpt_dir, "models", f"{stage}_{meta.run_id}.pth")
            if not os.path.exists(ckpt):
                continue
            model = _build_model(meta, device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
            model.eval()
            ei_flow.make_stage_figure(model, task, meta.run_id, stage, flow_dir, device,
                                      cue_scale=meta.cue_scale, cue_on_go=cue, style="magma")
            del model
        print(f"  flow fields (ei_flow/eistp sim) saved: {flow_dir}")
        return

    for stage, task in stage_task.items():
        ckpt = os.path.join(ckpt_dir, meta.run_id, f"{stage}_{meta.run_id}.pth")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(ckpt_dir, "models", f"{stage}_{meta.run_id}.pth")
        if not os.path.exists(ckpt):
            continue
        model = _build_model(meta, device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        timing = TIMINGS[task]

        if task == "dpa":
            inputs, targets = generate_dpa_trials(
                n_batch, timing, input_size=meta.input_size,
                noise=meta.noise_sigma())
            cnames = None
        elif task == "gng":
            inputs, targets = generate_gng_trials(
                n_batch, timing, input_size=meta.input_size,
                noise=meta.noise_sigma(), cue_on_go_input=cue,
                cue_scale=meta.cue_scale, nogo_target=meta.nogo_target)
            cnames = None
        else:
            inputs, targets, _, cnames = generate_dual_trials(
                n_batch, timing, input_size=meta.input_size,
                noise=meta.noise_sigma(), cue_on_go_input=cue,
                cue_scale=meta.cue_scale, nogo_target=meta.nogo_target)

        inputs_t  = torch.as_tensor(inputs,  dtype=torch.float32)
        targets_t = torch.as_tensor(targets, dtype=torch.float32)

        # When auto_xlim, scale grid density to match default ±1.5/151 spacing.
        # n_grid_per_unit = 151 / 3.0 ≈ 50.3; capped at 401 to stay fast.
        n_grid_per_unit = (n_grid / 3.0) if auto_xlim else None
        fig, _, _ = plot_task_flow_fields(
            model, inputs_t, timing, task,
            targets         = targets_t,
            condition_names = cnames,
            n_fp_seeds      = n_fp_seeds,
            cue_on_go_input = cue,
            cue_scale       = meta.cue_scale,
            xlim            = None if auto_xlim else XLIM,
            ylim            = None if auto_xlim else YLIM,
            n_grid          = n_grid,
            n_grid_per_unit = n_grid_per_unit,
            use_sim_field   = use_sim_field,
            sim_n_warmup    = sim_n_warmup,
            slow_tol        = slow_tol,
            show_slow_manifold   = show_slow_manifold,
            slow_manifold_thresh = slow_manifold_thresh,
        )
        fig.suptitle(f"{meta.run_id} — {stage} ({task.upper()})", y=1.01)
        out_path = os.path.join(flow_dir, f"fp_{stage}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        del model
    print(f"  flow fields saved: {flow_dir}")


# ============================================================
# MAIN
# ============================================================

_ALL_PLOTS = {"acc", "traj", "scatter", "flow"}

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive sweep plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
--plots components (default: all):
  acc      accuracy_stages + accuracy_by_trialtype
  traj     κ trajectories (mean ± SEM)
  scatter  fixed-point scatter
  flow     phase-portrait flow fields (slowest)

Examples:
  python plot_sweep.py --sweep_dir results/dual/myrun
  python plot_sweep.py --sweep_dir results/dual/myrun --plots acc traj
  python plot_sweep.py --sweep_dir results/dual/myrun --plots flow --run_ids s0_random
""")
    parser.add_argument("--sweep_dir",    type=str, required=True,
                        help="Path to sweep directory (contains results.jsonl and models/)")
    parser.add_argument("--out_root",     type=str, default=None,
                        help="Root output dir (default: ./results/figures/)")
    parser.add_argument("--run_ids",      type=str, nargs="*", default=None,
                        help="Subset of run IDs for individual plots (default: all)")
    parser.add_argument("--plots",        type=str, nargs="*", default=None,
                        choices=list(_ALL_PLOTS), metavar="COMPONENT",
                        help=f"Which components to plot: {{{', '.join(sorted(_ALL_PLOTS))}}} (default: all)")
    parser.add_argument("--no_summary",   action="store_true",
                        help="Skip summary figures")
    parser.add_argument("--no_individual",action="store_true",
                        help="Skip individual-run figures")
    parser.add_argument("--n_fp_seeds",   type=int, default=21,
                        help="Grid seeds for fixed-point finding (default: 21)")
    parser.add_argument("--use_sim_field", action="store_true",
                        help="Use simulation-based flow field instead of analytical κ-plane")
    parser.add_argument("--sim_n_warmup", type=int, default=0,
                        help="Warmup steps for sim flow field (0=grid-aligned/streamplot, >0=scattered/quiver)")
    parser.add_argument("--mark_slow", action="store_true",
                        help="Annotate shallow attractors (1-max|λ| <= slow_tol) as 'slow_attractor' (orange ring).")
    parser.add_argument("--slow_tol", type=float, default=0.06,
                        help="Slowness threshold for --mark_slow (default 0.06).")
    parser.add_argument("--n_batch", type=int, default=256,
                        help="Trials used for overlaid flow-field trajectories (default 256).")
    parser.add_argument("--show_slow_manifold", action="store_true",
                        help="Overlay the low-velocity ridge (slow manifold / remnant ring) on each flow panel.")
    parser.add_argument("--slow_manifold_thresh", type=float, default=0.12,
                        help="|F| threshold for the slow-manifold overlay (default 0.12).")
    parser.add_argument("--device",       type=str, default=None)
    parser.add_argument("--xlim",         type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="κ-plane axis limits for flow/scatter (default: ±1.5). "
                             "Use -5 5 for relu/elu/softplus sweeps.")
    parser.add_argument("--auto_xlim",    action="store_true",
                        help="Auto-detect κ-plane limits from trajectory percentiles "
                             "(overrides --xlim for flow fields).")
    parser.add_argument("--n_grid",       type=int, default=151,
                        help="Flow-field grid resolution (default: 151). "
                             "With --auto_xlim, this is scaled proportionally to the "
                             "detected range to maintain constant point density.")
    args = parser.parse_args()

    # Resolve which components to plot
    want = set(args.plots) if args.plots is not None else _ALL_PLOTS

    if args.xlim is not None:
        global XLIM, YLIM
        XLIM = YLIM = tuple(args.xlim)

    device     = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    sweep_name = os.path.basename(os.path.normpath(args.sweep_dir))
    out_root   = args.out_root or os.path.join(
        os.path.dirname(__file__), "results", "figures"
    )
    print(f"Sweep: {sweep_name}  device: {device}  out: {out_root}/{sweep_name}")
    if want != _ALL_PLOTS:
        print(f"  plotting: {', '.join(sorted(want))}")

    all_metas = _load_sweep_meta(args.sweep_dir)
    print(f"Loaded {len(all_metas)} runs")
    if args.run_ids:
        id_set    = set(args.run_ids)
        all_metas = [m for m in all_metas if m.run_id in id_set]
        print(f"  filtered to {len(all_metas)} runs by --run_ids")

    # ---- SUMMARY ----
    if not args.no_summary:
        sum_dir = os.path.join(out_root, sweep_name, "summary")
        os.makedirs(sum_dir, exist_ok=True)

        if "acc" in want:
            df = load_results(os.path.join(args.sweep_dir, "results.jsonl"))
            if args.run_ids:
                df = df[df["run_id"].isin(set(args.run_ids))]
            print("\n[summary] accuracy_stages")
            summary_accuracy_stages(df, sum_dir)
            print("\n[summary] accuracy_by_trialtype")
            summary_accuracy_by_trialtype(all_metas, args.sweep_dir, sum_dir, device)

        if "scatter" in want:
            print("\n[summary] fp_scatters (by stage + per input condition)")
            summary_fp_scatters(all_metas, args.sweep_dir, sum_dir, device,
                                n_seeds=args.n_fp_seeds,
                                slow_tol=args.slow_tol if args.mark_slow else None)

        if "traj" in want:
            print("\n[summary] avg_trajectories")
            summary_avg_trajectories(all_metas, args.sweep_dir, sum_dir, device)

    # ---- INDIVIDUAL ----
    if not args.no_individual:
        for meta in all_metas:
            ind_dir = os.path.join(out_root, sweep_name, "individual", meta.run_id)
            os.makedirs(ind_dir, exist_ok=True)
            print(f"\n[individual] {meta.run_id}")

            if "acc" in want:
                individual_accuracy_stages(meta, ind_dir)
                individual_accuracy_by_trialtype(meta, args.sweep_dir, ind_dir, device)

            if "traj" in want:
                individual_trajectories(meta, args.sweep_dir, ind_dir, device)

            if "scatter" in want:
                individual_fp_scatter(meta, args.sweep_dir, ind_dir, device,
                                      n_seeds=args.n_fp_seeds,
                                      slow_tol=args.slow_tol if args.mark_slow else None)

            if "flow" in want:
                individual_flow(meta, args.sweep_dir, ind_dir, device,
                                n_batch=args.n_batch,
                                n_fp_seeds=args.n_fp_seeds,
                                use_sim_field=args.use_sim_field,
                                sim_n_warmup=args.sim_n_warmup,
                                auto_xlim=args.auto_xlim,
                                n_grid=args.n_grid,
                                slow_tol=args.slow_tol if args.mark_slow else None,
                                show_slow_manifold=args.show_slow_manifold,
                                slow_manifold_thresh=args.slow_manifold_thresh)


if __name__ == "__main__":
    main()
