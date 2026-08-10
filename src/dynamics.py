from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.special
from scipy.optimize import root

from .tasks import TaskTiming

# Flow code split into flow_field / flow_fixedpoints / flow_rank2 (2026-08). Re-exported here so
# `from src.dynamics import X` keeps working; new code should import from the flow_* modules.
from .flow_field import (
    _model_device, _model_dtype, make_input, low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np, low_rank_jacobian_map_np, _phi_avgs, solve_sc_variance, _sc_a_bar, low_rank_field_sc_np, low_rank_jacobian_sc_np, trace_slow_manifold, project_rec_inputs_to_kappa, run_low_rank_with_effective_inputs, auto_limits_from_trajs, _canonical_flow_panels, sim_kappa_field, _sim_step_single, integrate_kappa_trajectories,
)
from .flow_fixedpoints import (
    merge_roots, find_all_fixed_points, classify_fixed_points, find_sim_fixed_points, classify_sim_fixed_points,
)
from .flow_rank2 import (
    make_vector_field_grid, _canonical_input_mask, _segments_from_mask, default_arrow_times, add_time_arrows, _parse_dual_name, condition_indices_for_task, flow_specs_for_task, _condition_colors, _reduce_marginals, _flow_panel_cache, _render_flow_panel, plot_task_flow_fields, plot_stage_stacked_flow,
)

@contextmanager
def temporary_attrs(obj, **kwargs):
    old = {k: getattr(obj, k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@torch.no_grad()
def rank2_kappa_delta_torch(model, x_t, kappa, q_slice="same"):
    """
    Compute Δκ for the rank-2 projected dynamics.

    q_slice: "same" (q = κ, slaved closure) or "zero" (q = 0).
    """
    assert model.rank == 2
    device = model.device
    kappa  = kappa.to(device)
    if x_t.ndim == 1: x_t = x_t[None, :]
    x_t = x_t.to(device)

    G, N = kappa.shape[0], model.hidden_size
    m, n = model.m.detach(), model.n.detach()

    if q_slice == "same":  q = kappa
    elif q_slice == "zero": q = torch.zeros_like(kappa)
    else: raise ValueError("q_slice must be 'same' or 'zero'.")

    beta_h = model.exp_alpha_rec.to(device)
    beta_r = model.exp_alpha.to(device)

    q_next = beta_h * q + (1.0 - beta_h) * kappa

    if model.wi is not None:
        input_drive = model.Ai * model.wi(x_t.expand(G, -1))
    else:
        input_drive = 0.0

    phi        = model.nonlinearity(model.gain * (input_drive + q_next @ m.T))
    F          = phi @ n / N
    kappa_next = beta_r * kappa + (1.0 - beta_r) * F
    return kappa_next - kappa


@torch.no_grad()
def rank2_kappa_flow(model, x_t, kappa1_lim=(-3.0, 3.0), kappa2_lim=(-3.0, 3.0),
                     grid_size=61, q_slice="same"):
    assert model.rank == 2
    device = model.device
    k1 = torch.linspace(kappa1_lim[0], kappa1_lim[1], grid_size, device=device)
    k2 = torch.linspace(kappa2_lim[0], kappa2_lim[1], grid_size, device=device)
    K1, K2 = torch.meshgrid(k1, k2, indexing="ij")
    kappa   = torch.stack([K1.reshape(-1), K2.reshape(-1)], dim=-1)

    d_kappa = rank2_kappa_delta_torch(model, x_t=x_t, kappa=kappa, q_slice=q_slice)
    U = d_kappa[:, 0].reshape(grid_size, grid_size)
    V = d_kappa[:, 1].reshape(grid_size, grid_size)
    speed = torch.sqrt(U**2 + V**2)

    return K1.cpu().numpy(), K2.cpu().numpy(), U.cpu().numpy(), V.cpu().numpy(), speed.cpu().numpy()


# ---------------------------------------------------------------------------
# Fixed points (torch-native)
# ---------------------------------------------------------------------------


def classify_fixed_point(model, x_t, kappa_star, q_slice="same", eps=1e-4):
    kappa_star = np.asarray(kappa_star, dtype=np.float64)
    J_delta    = np.zeros((2, 2), dtype=np.float64)

    for j in range(2):
        step = np.zeros(2)
        step[j] = eps
        kp = torch.tensor(kappa_star + step, dtype=torch.float32, device=model.device)[None, :]
        km = torch.tensor(kappa_star - step, dtype=torch.float32, device=model.device)[None, :]
        fp = rank2_kappa_delta_torch(model, x_t, kp, q_slice=q_slice).cpu().numpy()[0]
        fm = rank2_kappa_delta_torch(model, x_t, km, q_slice=q_slice).cpu().numpy()[0]
        J_delta[:, j] = (fp - fm) / (2.0 * eps)

    J_map   = np.eye(2) + J_delta
    eigvals = np.linalg.eigvals(J_map)
    abs_eigs= np.abs(eigvals)

    if   np.all(abs_eigs < 1.0): kind = "stable"
    elif np.all(abs_eigs > 1.0): kind = "unstable"
    else:                         kind = "saddle"

    return eigvals, kind


def find_rank2_fixed_points(model, x_t, kappa1_lim=(-3, 3), kappa2_lim=(-3, 3),
                             n_init=21, q_slice="same", root_tol=1e-8,
                             residual_tol=1e-5, duplicate_tol=1e-3, pad=0.25):
    assert model.rank == 2

    def f_np(z):
        z_t = torch.tensor(z, dtype=torch.float32, device=model.device)[None, :]
        return rank2_kappa_delta_torch(model, x_t=x_t, kappa=z_t, q_slice=q_slice).cpu().numpy()[0].astype(np.float64)

    fixed_points = []

    for k1 in np.linspace(kappa1_lim[0], kappa1_lim[1], n_init):
        for k2 in np.linspace(kappa2_lim[0], kappa2_lim[1], n_init):
            sol = root(f_np, np.array([k1, k2], dtype=np.float64), method="hybr", tol=root_tol)
            if not sol.success: continue
            z     = sol.x.astype(np.float64)
            resid = float(np.linalg.norm(f_np(z)))
            in_bounds = (
                kappa1_lim[0] - pad <= z[0] <= kappa1_lim[1] + pad and
                kappa2_lim[0] - pad <= z[1] <= kappa2_lim[1] + pad
            )
            if resid > residual_tol or not in_bounds: continue

            is_new = True
            for fp in fixed_points:
                if np.linalg.norm(z - fp["kappa"]) < duplicate_tol:
                    is_new = False
                    if resid < fp["residual"]:
                        fp["kappa"], fp["residual"] = z, resid
                    break

            if is_new:
                eigvals, kind = classify_fixed_point(model, x_t, z, q_slice=q_slice)
                fixed_points.append({"kappa": z, "residual": resid, "eigvals": eigvals, "kind": kind})

    return sorted(fixed_points, key=lambda d: (d["kappa"][0], d["kappa"][1]))


def overlay_fixed_points(ax, fixed_points, label=False, stable_kwargs=None,
                         saddle_kwargs=None, unstable_kwargs=None):
    stable_kwargs   = stable_kwargs   or dict(s=200, marker="o", facecolor="white", edgecolor="black", linewidth=1.0, zorder=40)
    saddle_kwargs   = saddle_kwargs   or dict(s=200, marker="x", color="white", linewidth=2.0, zorder=40)
    unstable_kwargs = unstable_kwargs or dict(s=200, marker="o", facecolor="none", edgecolor="red", linewidth=1.7, zorder=40)
    used_labels = set()

    for fp in fixed_points:
        k, kind = fp["kappa"], fp["kind"]
        if   kind == "stable":   kwargs, lab = stable_kwargs.copy(),   "stable FP"
        elif kind == "saddle":   kwargs, lab = saddle_kwargs.copy(),   "saddle FP"
        else:                    kwargs, lab = unstable_kwargs.copy(), "unstable FP"
        if label and lab not in used_labels:
            kwargs["label"] = lab
            used_labels.add(lab)
        ax.scatter(k[0], k[1], **kwargs)


# ---------------------------------------------------------------------------
# Heatmap + streamplot panel
# ---------------------------------------------------------------------------


def plot_kappa_heatmap_flow_panel(ax, K1, K2, U, V, heatmap=None, title=None,
                                   cmap="magma", flow_color="white", density=1.2,
                                   linewidth=0.8, arrowsize=1.0, normalize_flow=False,
                                   vmin=None, vmax=None):
    if heatmap is None:
        heatmap = np.sqrt(U**2 + V**2)

    if normalize_flow:
        speed   = np.sqrt(U**2 + V**2)
        U_plot  = U / (speed + 1e-12)
        V_plot  = V / (speed + 1e-12)
    else:
        U_plot, V_plot = U, V

    hm = ax.pcolormesh(K1.T, K2.T, heatmap.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.streamplot(K1.T, K2.T, U_plot.T, V_plot.T, color=flow_color,
                  density=density, linewidth=linewidth, arrowsize=arrowsize)
    ax.set_xlabel(r"$\kappa_0$")
    ax.set_ylabel(r"$\kappa_1$")
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)
    return hm


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_kappa_trajectories(model, ff_inputs, targets=None, disable_rwd=False):
    """Run the model and return κ_t = r_t @ n / N."""
    ff_inputs = ff_inputs.to(model.device)
    if targets is not None:
        targets = targets.to(model.device)
    attrs = {"noise": 0.0}
    if disable_rwd:
        attrs["rwd"] = False
    with temporary_attrs(model, **attrs):
        readout, rates, rec_inputs = model(ff_inputs, targets=targets, ret_rates=True)
    kappa = rates @ model.n.detach() / model.hidden_size
    return kappa.cpu().numpy(), rates, rec_inputs, readout


def infer_dpa_labels_from_inputs(ff_inputs, timing=None, sample_channels=(0, 1), test_channels=(2, 3)):
    x = ff_inputs.detach().cpu() if torch.is_tensor(ff_inputs) else torch.as_tensor(ff_inputs)

    if timing is not None:
        n_on, n_off = timing.n_stim_on, timing.n_stim_off
        s_sample = slice(int(n_on[0]), int(n_off[0]))
        s_test   = slice(int(n_on[3]), int(n_off[3])) if len(n_on) >= 4 else slice(int(n_on[1]), int(n_off[1]))
        sample_score = x[:, s_sample, list(sample_channels)].mean(dim=1)
        test_score   = x[:, s_test,   list(test_channels)].mean(dim=1)
    else:
        sample_score = x[:, :, list(sample_channels)].mean(dim=1)
        test_score   = x[:, :, list(test_channels)].mean(dim=1)

    sample_char = np.where(sample_score.argmax(dim=1).numpy() == 0, "A", "B")
    test_char   = np.where(test_score.argmax(dim=1).numpy()   == 0, "C", "D")
    return np.char.add(sample_char, test_char)


def dpa_labels_from_condition_names(condition_names):
    labels = []
    for name in condition_names:
        s = str(name)
        if   s.startswith("A"): sample = "A"
        elif s.startswith("B"): sample = "B"
        else: raise ValueError(f"Cannot parse sample from condition name: {s}")
        if   s.endswith("C"): test = "C"
        elif s.endswith("D"): test = "D"
        else: raise ValueError(f"Cannot parse test from condition name: {s}")
        labels.append(sample + test)
    return np.asarray(labels)


def overlay_kappa_trajectories(ax, kappa, trial_inds=None, color="cyan", alpha=0.25,
                                linewidth=0.9, start_marker=False, end_marker=False):
    B = kappa.shape[0]
    if trial_inds is None:
        trial_inds = range(B)
    for b in trial_inds:
        traj = kappa[b]
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=linewidth, zorder=7)
        if start_marker:
            ax.scatter(traj[0, 0], traj[0, 1], color=color, s=20, marker="o", alpha=alpha, zorder=8)
        if end_marker:
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=30, marker="x", alpha=alpha, zorder=8)


def overlay_average_kappa_trajectories(ax, kappa, labels, trial_types=("AC", "AD", "BC", "BD"),
                                        colors=None, linewidth=3.0, alpha=0.95,
                                        start_marker=True, end_marker=True, label_prefix=""):
    if colors is None:
        colors = {"AC": "tab:blue", "AD": "tab:orange", "BC": "tab:green", "BD": "tab:red"}
    labels = np.asarray(labels)
    for tt in trial_types:
        mask = labels == tt
        if not np.any(mask): continue
        mean_traj = kappa[mask].mean(axis=0)
        color     = colors.get(tt, "black")
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=color, linewidth=linewidth,
                alpha=alpha, label=f"{label_prefix}{tt} mean", zorder=12)
        if start_marker:
            ax.scatter(mean_traj[0, 0],  mean_traj[0, 1],  color=color, s=45, marker="o",
                       edgecolor="black", linewidth=0.7, zorder=13)
        if end_marker:
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], color=color, s=65, marker="X",
                       edgecolor="black", linewidth=0.7, zorder=13)


# ---------------------------------------------------------------------------
# Two-panel autonomous vs input-driven figure
# ---------------------------------------------------------------------------


def plot_autonomous_vs_input_heatmap_flow(
    model, x_autonomous, x_input,
    ff_inputs=None, targets=None, timing=None, condition_names=None,
    kappa1_lim=(-3, 3), kappa2_lim=(-3, 3), grid_size=61, q_slice="same",
    heatmap_kind="speed", cmap="magma", flow_color="white", normalize_flow=False,
    shared_color_scale=True, show_fixed_points=True, fp_n_init=21,
    overlay_average=True, overlay_individual=False, trial_inds=None,
    avg_trial_types=("AC", "AD", "BC", "BD"), avg_colors=None,
):
    K1a, K2a, Ua, Va, Sa = rank2_kappa_flow(model, x_t=x_autonomous, kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, grid_size=grid_size, q_slice=q_slice)
    K1i, K2i, Ui, Vi, Si = rank2_kappa_flow(model, x_t=x_input,      kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, grid_size=grid_size, q_slice=q_slice)

    if   heatmap_kind == "speed": Ha, Hi, cbar_label = Sa, Si, r"$\|\Delta \kappa\|$"
    elif heatmap_kind == "dK1":   Ha, Hi, cbar_label = Ua, Ui, r"$\Delta \kappa_0$"
    elif heatmap_kind == "dK2":   Ha, Hi, cbar_label = Va, Vi, r"$\Delta \kappa_1$"
    else: raise ValueError("heatmap_kind must be 'speed', 'dK1', or 'dK2'.")

    vmin = min(np.nanmin(Ha), np.nanmin(Hi)) if shared_color_scale else None
    vmax = max(np.nanmax(Ha), np.nanmax(Hi)) if shared_color_scale else None

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True, sharex=True, sharey=True)

    hm0 = plot_kappa_heatmap_flow_panel(axs[0], K1a, K2a, Ua, Va, heatmap=Ha, title="Autonomous / baseline",
                                         cmap=cmap, flow_color=flow_color, normalize_flow=normalize_flow, vmin=vmin, vmax=vmax)
    hm1 = plot_kappa_heatmap_flow_panel(axs[1], K1i, K2i, Ui, Vi, heatmap=Hi, title="Input-driven",
                                         cmap=cmap, flow_color=flow_color, normalize_flow=normalize_flow, vmin=vmin, vmax=vmax)
    for ax in axs:
        ax.set_xlim(kappa1_lim); ax.set_ylim(kappa2_lim)

    fps_aut, fps_inp = [], []
    if show_fixed_points:
        fps_aut = find_rank2_fixed_points(model, x_t=x_autonomous, kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, n_init=fp_n_init, q_slice=q_slice)
        fps_inp = find_rank2_fixed_points(model, x_t=x_input,      kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, n_init=fp_n_init, q_slice=q_slice)
        overlay_fixed_points(axs[0], fps_aut)
        overlay_fixed_points(axs[1], fps_inp)

    if ff_inputs is not None:
        kappa, rates, rec_inputs, readout = get_kappa_trajectories(model, ff_inputs=ff_inputs, targets=targets, disable_rwd=True)
        dpa_labels = (
            dpa_labels_from_condition_names(condition_names) if condition_names is not None
            else infer_dpa_labels_from_inputs(ff_inputs, timing=timing)
        )
        if overlay_individual:
            for ax in axs:
                overlay_kappa_trajectories(ax, kappa, trial_inds=trial_inds, color="cyan", alpha=0.25)
        if overlay_average:
            for ax in axs:
                overlay_average_kappa_trajectories(ax, kappa, dpa_labels, trial_types=avg_trial_types, colors=avg_colors)

    cbar = fig.colorbar(hm1, ax=axs, shrink=0.85)
    cbar.set_label(cbar_label)
    handles, labels_list = axs[1].get_legend_handles_labels()
    if handles:
        axs[1].legend(handles, labels_list, loc="upper right", frameon=True, fontsize=8)

    return fig, axs, fps_aut, fps_inp

