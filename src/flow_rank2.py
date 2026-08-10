from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.special
from scipy.optimize import root

from .tasks import TaskTiming
from .flow_field import (
    _model_device, _model_dtype, make_input, low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np, low_rank_jacobian_map_np, _phi_avgs, solve_sc_variance, _sc_a_bar, low_rank_field_sc_np, low_rank_jacobian_sc_np, trace_slow_manifold, project_rec_inputs_to_kappa, run_low_rank_with_effective_inputs, auto_limits_from_trajs, _canonical_flow_panels, sim_kappa_field, _sim_step_single, integrate_kappa_trajectories,
)
from .flow_fixedpoints import (
    merge_roots, find_all_fixed_points, classify_fixed_points, find_sim_fixed_points, classify_sim_fixed_points,
)

"""Rank-2 flow rendering: κ0–κ1 stage×condition portraits (plot_stage_stacked_flow) + panels."""

def make_vector_field_grid(model, ff_input, xlim, ylim, n_grid=151, include_beta=False):
    params       = low_rank_numpy_params(model)
    ff_input_np  = torch.as_tensor(ff_input).detach().cpu().numpy().astype(np.float64)

    k1 = np.linspace(xlim[0], xlim[1], n_grid)
    k2 = np.linspace(ylim[0], ylim[1], n_grid)
    K1, K2 = np.meshgrid(k1, k2)
    K      = np.stack([K1, K2], axis=-1)

    field  = low_rank_field_np(params, K, ff_input=ff_input_np, include_beta=include_beta)
    U, V   = field[..., 0], field[..., 1]
    speed  = np.sqrt(U**2 + V**2)

    return K1, K2, U, V, speed, params, ff_input_np


# ---------------------------------------------------------------------------
# Fixed points
# ---------------------------------------------------------------------------


def _canonical_input_mask(effective_x, dims, threshold=0.35, atol_inactive=0.35):
    x = effective_x.detach().cpu()

    if dims is None:
        return x.abs().amax(dim=-1) < atol_inactive

    dims = list(dims)
    active = torch.ones(x.shape[:2], dtype=torch.bool)
    for d in dims:
        active &= x[..., d] > threshold

    inactive_dims = [d for d in range(x.shape[-1]) if d not in dims]
    inactive = x[..., inactive_dims].abs().amax(dim=-1) < atol_inactive
    return active & inactive


def _segments_from_mask(mask_1d, min_len=2):
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    chunks = np.split(idx, breaks)
    return [slice(int(c[0]), int(c[-1]) + 1) for c in chunks if len(c) >= min_len]


# ---------------------------------------------------------------------------
# Time arrows
# ---------------------------------------------------------------------------


def default_arrow_times(T, n_arrows=4, arrow_step=3):
    if T <= arrow_step + 1:
        return np.array([], dtype=int)
    lo, hi = 1, T - arrow_step - 1
    n_arrows = min(n_arrows, max(1, hi - lo + 1))
    return np.unique(np.linspace(lo, hi, n_arrows).astype(int))


def add_time_arrows(ax, xy, times=None, color="k", arrow_step=3,
                    mutation_scale=13, lw=1.5, alpha=0.95, zorder=8, min_length=1e-4):
    xy = np.asarray(xy, dtype=np.float64)
    T  = xy.shape[0]

    if times is None:
        times = default_arrow_times(T, n_arrows=4, arrow_step=arrow_step)

    for t in times:
        t  = int(t)
        if t < 0: t += T
        if t < 0 or t >= T - 1: continue

        j0 = t
        j1 = min(t + arrow_step, T - 1)
        while j1 < T and np.linalg.norm(xy[j1] - xy[j0]) < min_length:
            j0 += 1
            j1  = min(j0 + arrow_step, T - 1)
            if j0 >= T - 1: break

        if j0 >= T - 1: continue
        p0, p1 = xy[j0], xy[j1]
        if np.linalg.norm(p1 - p0) < min_length: continue

        ax.annotate(
            "",
            xy=p1, xytext=p0,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                shrinkA=0, shrinkB=0, mutation_scale=mutation_scale,
            ),
            zorder=zorder,
        )


# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------


def _parse_dual_name(name):
    parts = str(name).split("_")
    if len(parts) == 2:
        sample, test = parts
        gng = "none"
    elif len(parts) == 3:
        sample, gng, test = parts
    else:
        raise ValueError(f"Unexpected dual condition name: {name}")
    return {"sample": sample, "gng": gng, "test": test, "pair": f"{sample}{test}"}


def condition_indices_for_task(inputs, timing, task, condition_names=None, dual_mode="conditions"):
    task = task.lower()
    X    = inputs.detach().cpu()
    n_on = timing.n_stim_on
    n_off= timing.n_stim_off

    if task == "dpa":
        sample_epoch = slice(int(n_on[0]),  int(n_off[0]))
        test_epoch   = slice(int(n_on[1]),  int(n_off[1]))
        sample_drive = X[:, sample_epoch, 0:2].mean(dim=1)
        test_drive   = X[:, test_epoch,   2:4].mean(dim=1)
        is_A = sample_drive[:, 0] > sample_drive[:, 1]
        is_C = test_drive[:,   0] > test_drive[:,   1]
        cond_idx = {
            "AC": torch.where( is_A &  is_C)[0], "AD": torch.where( is_A & ~is_C)[0],
            "BC": torch.where(~is_A &  is_C)[0], "BD": torch.where(~is_A & ~is_C)[0],
        }
        meta = {
            "AC": dict(sample="A", test="C", pair="AC"), "AD": dict(sample="A", test="D", pair="AD"),
            "BC": dict(sample="B", test="C", pair="BC"), "BD": dict(sample="B", test="D", pair="BD"),
        }
        return cond_idx, meta

    if task == "gng":
        stim_epoch  = slice(int(n_on[0]), int(n_off[0]))
        stim_drive  = X[:, stim_epoch, 4:6].mean(dim=1)
        is_go       = stim_drive[:, 0] > stim_drive[:, 1]
        cond_idx    = {"Go": torch.where(is_go)[0], "NoGo": torch.where(~is_go)[0]}
        meta        = {"Go": dict(gng="go"), "NoGo": dict(gng="nogo")}
        return cond_idx, meta

    if task == "dual":
        if condition_names is None:
            raise ValueError("For task='dual', pass condition_names from generate_dual_trials.")
        names = np.asarray(condition_names).astype(str)

        if dual_mode == "conditions":
            cond_idx = {}
            meta     = {}
            for name in sorted(np.unique(names)):
                idx = np.where(names == name)[0]
                cond_idx[name] = torch.as_tensor(idx, dtype=torch.long)
                meta[name]     = _parse_dual_name(name)
            return cond_idx, meta

        if dual_mode == "groups":
            groups = {"AC": [], "AD": [], "BC": [], "BD": [], "Go": [], "NoGo": [], "DPAOnly": []}
            for i, name in enumerate(names):
                info = _parse_dual_name(name)
                if info["pair"] in groups:
                    groups[info["pair"]].append(i)
                if info["gng"] == "go":      groups["Go"].append(i)
                elif info["gng"] == "nogo":  groups["NoGo"].append(i)
                elif info["gng"] == "none":  groups["DPAOnly"].append(i)
            cond_idx = {k: torch.as_tensor(v, dtype=torch.long) for k, v in groups.items()}
            meta = {
                "AC": dict(pair="AC"), "AD": dict(pair="AD"),
                "BC": dict(pair="BC"), "BD": dict(pair="BD"),
                "Go": dict(gng="go"), "NoGo": dict(gng="nogo"), "DPAOnly": dict(gng="none"),
            }
            return cond_idx, meta

    raise ValueError(f"Unknown task: {task}")


def flow_specs_for_task(timing, task, input_size, cond_idx, meta, dual_mode="conditions",
                        cue_on_go_input=False, cue_scale=1.0):
    task = task.lower()

    if task == "dpa":
        return [
            dict(name="Autonomous", dims=None, conds=["AC", "AD", "BC", "BD"]),
            dict(name="A", dims=[0], conds=["AC", "AD"]),
            dict(name="B", dims=[1], conds=["BC", "BD"]),
            dict(name="C", dims=[2], conds=["AC", "BC"]),
            dict(name="D", dims=[3], conds=["AD", "BD"]),
        ]

    if task == "gng":
        specs = [
            dict(name="Autonomous", dims=None, conds=["Go", "NoGo"]),
            dict(name="Go",   dims=[4], conds=["Go"]),
            dict(name="NoGo", dims=[5], conds=["NoGo"]),
        ]
        if not cue_on_go_input:
            specs.append(dict(name="Cue", dims=[6], conds=["Go", "NoGo"]))
        else:
            # cue rides on the go channel (4) at cue amplitude; show both go & nogo means
            specs.append(dict(name="Cue", dims=[4], conds=["Go", "NoGo"], value=cue_scale))
        return specs

    if task == "dual":
        labels = list(cond_idx.keys())
        if dual_mode == "conditions":
            A_conds    = [c for c in labels if meta[c]["sample"] == "A"]
            B_conds    = [c for c in labels if meta[c]["sample"] == "B"]
            C_conds    = [c for c in labels if meta[c]["test"]   == "C"]
            D_conds    = [c for c in labels if meta[c]["test"]   == "D"]
            Go_conds   = [c for c in labels if meta[c]["gng"]    == "go"]
            NoGo_conds = [c for c in labels if meta[c]["gng"]    == "nogo"]
            auto_conds = labels
        else:
            auto_conds = ["AC", "AD", "BC", "BD", "DPAOnly"]
            A_conds, B_conds = ["AC", "AD"], ["BC", "BD"]
            C_conds, D_conds = ["AC", "BC"], ["AD", "BD"]
            Go_conds, NoGo_conds = ["Go"], ["NoGo"]

        specs = [
            dict(name="Autonomous", dims=None,  conds=auto_conds),
            dict(name="A",    dims=[0], conds=A_conds),
            dict(name="B",    dims=[1], conds=B_conds),
            dict(name="Go",   dims=[4], conds=Go_conds),
            dict(name="NoGo", dims=[5], conds=NoGo_conds),
        ]
        if not cue_on_go_input:
            specs.append(dict(name="Cue", dims=[6], conds=Go_conds + NoGo_conds))
        else:
            specs.append(dict(name="Cue", dims=[4], conds=Go_conds + NoGo_conds, value=cue_scale))
        specs += [
            dict(name="C", dims=[2], conds=C_conds),
            dict(name="D", dims=[3], conds=D_conds),
        ]
        return specs

    raise ValueError(f"Unknown task: {task}")


def _condition_colors(cond_labels):
    base = {
        "AC": "tab:blue", "AD": "tab:orange", "BC": "tab:green", "BD": "tab:purple",
        "Go": "tab:red",  "NoGo": "tab:brown", "DPAOnly": "tab:gray",
    }
    cmap   = plt.get_cmap("tab20")
    colors = {}
    for i, c in enumerate(cond_labels):
        colors[c] = base.get(c, cmap(i % 20))
    return colors


# ---------------------------------------------------------------------------
# Main phase-portrait plotter
# ---------------------------------------------------------------------------


def _reduce_marginals(pts, labels, mem_thresh=0.6, keep_thresh=0.3):
    """Autonomous-field marginal cleanup. A near-line-attractor makes almost every point
    'marginal'. If the memory is resolved as a genuine BISTABLE PAIR of (slow_)attractors
    (one κ₀>mem_thresh and one κ₀<−mem_thresh), drop ALL marginal points (slow-manifold clutter).
    If a memory side is MISSING, keep a single best marginal point — the most extreme κ₀ on that
    side — as a good approximation of where the missing memory well sits."""
    pts = np.asarray(pts); labels = np.asarray(labels, dtype=object).copy()
    if len(pts) == 0:
        return pts, labels
    att = np.isin(labels, ("attractor", "slow_attractor"))
    mem_pos = bool(np.any(att & (pts[:, 0] >  mem_thresh)))
    mem_neg = bool(np.any(att & (pts[:, 0] < -mem_thresh)))
    marg = labels == "marginal"
    keep = ~marg                       # drop all marginals by default
    if not (mem_pos and mem_neg):      # a memory is missing → keep one representative per missing side,
        midx = np.where(marg)[0]       # relabeled 'slow_attractor' (it IS transversely attracting)
        if len(midx):
            if not mem_neg:
                j = midx[np.argmin(pts[midx, 0])]
                if pts[j, 0] < -keep_thresh:
                    keep[j] = True; labels[j] = "slow_attractor"
            if not mem_pos:
                j = midx[np.argmax(pts[midx, 0])]
                if pts[j, 0] >  keep_thresh:
                    keep[j] = True; labels[j] = "slow_attractor"
    return pts[keep], labels[keep]


def _flow_panel_cache(model, spec, input_size, effective_x, xlim, ylim, n_grid, *,
                      attention_input=False, attention_scale=1.0,
                      use_sim_field=False, sim_n_warmup=0,
                      field_input_noise=0.0, field_noise_K=16, field_noise_seed=0,
                      include_beta_in_field=False, n_fp_seeds=41, slow_tol=None,
                      input_threshold=0.35, inactive_atol=0.35):
    """Field + fixed points for ONE frozen-input panel `spec`. Returns (cache, finite-speed-flat).
    Factored out of plot_task_flow_fields so the single-stage plotter and the stacked multi-stage
    plotter compute panels identically."""
    device = _model_device(model)
    dtype  = _model_dtype(model)
    ff_input = make_input(input_size, active_dims=spec["dims"], value=spec.get("value", 1.0),
                          device=device, dtype=dtype)
    if attention_input:   # tonic attention (last channel) ON during the whole task
        ff_input[-1] = attention_scale
    if use_sim_field:
        K1, K2, U, V, speed = sim_kappa_field(
            model, ff_input, xlim=xlim, ylim=ylim, n_grid=n_grid, n_warmup=sim_n_warmup)
        fixed_points, fixed_residuals = find_sim_fixed_points(
            model, ff_input, xlim=xlim, ylim=ylim, n_seeds=n_fp_seeds,
            n_warmup=sim_n_warmup, residual_tol=1e-5, merge_tol=5e-2)
        fp_labels, fp_eigvals = classify_sim_fixed_points(
            model, fixed_points, ff_input, n_warmup=sim_n_warmup, slow_tol=slow_tol)
    else:
        ff_field = ff_input
        if field_input_noise and field_input_noise > 0.0:
            gen = torch.Generator(device=device).manual_seed(int(field_noise_seed))
            nz  = field_input_noise * torch.randn(int(field_noise_K), input_size,
                                                  generator=gen, device=device, dtype=dtype)
            ff_field = ff_input[None, :] + nz   # (K, input_size) → noise-averaged field/FPs
        K1, K2, U, V, speed, params, ff_input_np = make_vector_field_grid(
            model, ff_input=ff_field, xlim=xlim, ylim=ylim, n_grid=n_grid,
            include_beta=include_beta_in_field)
        fixed_points, fixed_residuals = find_all_fixed_points(
            model, xlim=xlim, ylim=ylim, ff_input=ff_field,
            n_seeds=n_fp_seeds, residual_tol=1e-8, merge_tol=5e-2)
        fp_labels, fp_eigvals = classify_fixed_points(model, fixed_points, ff_input=ff_field, slow_tol=slow_tol)
    panel_mask = _canonical_input_mask(
        effective_x, dims=spec["dims"], threshold=input_threshold, atol_inactive=inactive_atol)
    cache = dict(
        spec=spec, ff_input=ff_input,
        K1=K1, K2=K2, U=U, V=V, speed=speed,
        fixed_points=fixed_points, fixed_residuals=fixed_residuals,
        fp_labels=fp_labels, fp_eigvals=fp_eigvals,
        panel_mask=panel_mask.detach().cpu().numpy(),
    )
    return cache, speed[np.isfinite(speed)].reshape(-1)


def _render_flow_panel(ax, cache, *, speed_vmax, sim_scattered, kappa_traj, cond_idx, colors,
                       xlim, ylim, model, show_slow_manifold=False, slow_manifold_thresh=0.12,
                       show_single_trials=False, max_single_trials=12,
                       max_autonomous_conditions=None):
    """Draw ONE flow panel (speed heatmap + streamlines + fixed points + condition trajectories)
    into `ax`. Returns the pcolormesh handle (for a shared colorbar). Factored out of
    plot_task_flow_fields; behaviour is byte-identical to the original inline loop."""
    spec = cache["spec"]
    K1, K2, U, V, speed = cache["K1"], cache["K2"], cache["U"], cache["V"], cache["speed"]

    hm = ax.pcolormesh(K1, K2, speed, shading="auto", cmap="magma",
                       vmax=speed_vmax, rasterized=True, zorder=0)
    if sim_scattered:
        ax.quiver(K1, K2, U, V, color="white", alpha=0.7,
                  scale=None, scale_units="xy", angles="xy", zorder=2)
    else:
        ax.streamplot(K1[0, :], K2[:, 0], U, V, color="white", density=1.05,
                      linewidth=0.70, arrowsize=0.80, zorder=2)

    fps_draw, labels_draw = cache["fixed_points"], cache["fp_labels"]
    if spec["name"] == "Autonomous" and len(labels_draw):
        fps_draw, labels_draw = _reduce_marginals(fps_draw, labels_draw)

    for label, marker in [("attractor","o"),("slow_attractor","o"),("marginal","s"),
                          ("saddle","x"),("repeller","^"),("nonhyperbolic","D")]:
        if len(labels_draw) == 0: continue
        mask = labels_draw == label
        if not np.any(mask): continue
        pts = fps_draw[mask]
        if label == "saddle":
            ax.scatter(pts[:, 0], pts[:, 1], s=60, marker="x", color="cyan", linewidths=1.5, zorder=10, label=label)
        elif label == "marginal":
            ax.scatter(pts[:, 0], pts[:, 1], s=55, marker="s", facecolors="none",
                       edgecolors="gold", linewidths=1.6, zorder=10, label=label)
        elif label == "slow_attractor":
            ax.scatter(pts[:, 0], pts[:, 1], s=62, marker="o", facecolors="none",
                       edgecolors="orange", linewidths=1.8, zorder=10, label=label)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=60, marker=marker,
                       facecolors="cyan" if label == "attractor" else "none",
                       edgecolors="cyan", linewidths=1.1, zorder=10, label=label)

    conds      = [] if spec["dims"] is None else list(spec["conds"])
    panel_mask = cache["panel_mask"]

    if spec["name"] == "Autonomous" and max_autonomous_conditions is not None:
        conds = conds[:max_autonomous_conditions]

    for cond in conds:
        if cond not in cond_idx: continue
        idx = cond_idx[cond].detach().cpu().numpy()
        if len(idx) == 0: continue

        cond_mask = panel_mask[idx].mean(axis=0) > 0.5
        segments  = _segments_from_mask(cond_mask, min_len=2)
        if not segments: continue

        traj_mean = kappa_traj[idx].mean(axis=0)
        color     = colors.get(cond, "tab:gray")

        for seg_slice in segments:
            seg = traj_mean[seg_slice, :2]
            if seg.shape[0] < 2: continue

            if show_single_trials:
                for i in idx[:max_single_trials]:
                    seg_i = kappa_traj[i, seg_slice, :2]
                    ax.plot(seg_i[:, 0], seg_i[:, 1], color=color, lw=0.7, alpha=0.12, zorder=4)

            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2.7, alpha=0.95, zorder=6, label=cond)
            add_time_arrows(ax, seg, times=default_arrow_times(seg.shape[0], 4, 3),
                            color=color, arrow_step=3, mutation_scale=13, lw=1.5, alpha=0.95, zorder=8)
            ax.scatter(seg[0, 0],  seg[0, 1],  marker="o", s=38, color=color, edgecolors="white", linewidths=0.6, zorder=9)
            ax.scatter(seg[-1, 0], seg[-1, 1], marker="X", s=46, color=color, edgecolors="white", linewidths=0.6, zorder=9)

    if show_slow_manifold:
        mpts, mvel, _ = trace_slow_manifold(
            model, cache["ff_input"], xlim, ylim, vel_thresh=slow_manifold_thresh)
        if len(mpts) > 0:
            ax.scatter(mpts[:, 0], mpts[:, 1], c=mvel, cmap="spring", s=14,
                       vmin=0.0, vmax=slow_manifold_thresh, zorder=11,
                       edgecolors="none", label="slow manifold")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    return hm


def plot_task_flow_fields(
    model, inputs, timing, task,
    targets=None, condition_names=None, dual_mode="conditions",
    xlim=None, ylim=None, n_grid=151, n_grid_per_unit=None, n_fp_seeds=41,
    include_beta_in_field=False, qlo=1, qhi=99, pad=1.25,
    symmetric_auto=True, speed_percentile=98,
    figsize_per_panel=4.2, input_threshold=0.35, inactive_atol=0.35,
    show_single_trials=False, max_single_trials=12, max_autonomous_conditions=None,
    cue_on_go_input=False, cue_scale=1.0,
    use_sim_field=False, sim_n_warmup=0, slow_tol=None,
    show_slow_manifold=False, slow_manifold_thresh=0.12,
    attention_input=False, attention_scale=1.0,
    field_input_noise=0.0, field_noise_K=16, field_noise_seed=0,
):
    """Generic low-rank phase portrait for DPA, GNG, and Dual tasks.

    field_input_noise>0 renders the NOISE-AVERAGED field/fixed points: the frozen input for
    each panel is replicated into field_noise_K draws with N(0, field_input_noise²) added per
    channel and the field is averaged over them (E_x[Ψ(κ)]). Trajectories/slow-manifold keep
    the clean input.
    """
    task = task.lower()
    if model.m.shape[1] != 2:
        raise ValueError("This plotter assumes rank == 2.")

    model.eval()
    device     = _model_device(model)
    dtype      = _model_dtype(model)
    input_size = inputs.shape[-1]

    with torch.no_grad():
        readouts, rates, rec_inputs, effective_x = run_low_rank_with_effective_inputs(
            model, inputs, targets=targets
        )

    kappa_traj = project_rec_inputs_to_kappa(model, rec_inputs).detach().cpu().numpy()

    if xlim is None or ylim is None:
        # Use full range (qlo=0, qhi=100) so input-driven excursions in a small
        # fraction of timesteps are not clipped by percentile-based limits.
        auto_xlim, auto_ylim = auto_limits_from_trajs(
            kappa_traj[..., :2], qlo=0, qhi=100, pad=pad, symmetric=symmetric_auto
        )
        xlim = auto_xlim if xlim is None else xlim
        ylim = auto_ylim if ylim is None else ylim

    # Scale grid density to match n_grid_per_unit pts/unit if requested.
    if n_grid_per_unit is not None:
        axis_range = xlim[1] - xlim[0]
        n_grid = min(int(round(n_grid_per_unit * axis_range)), 401)

    cond_idx, meta = condition_indices_for_task(
        inputs.detach().cpu(), timing=timing, task=task,
        condition_names=condition_names, dual_mode=dual_mode,
    )
    specs = flow_specs_for_task(
        timing=timing, task=task, input_size=input_size,
        cond_idx=cond_idx, meta=meta, dual_mode=dual_mode,
        cue_on_go_input=cue_on_go_input, cue_scale=cue_scale,
    )

    max_dim = max(
        [d for spec in specs if spec["dims"] is not None for d in spec["dims"]], default=-1
    )
    if input_size <= max_dim:
        raise ValueError(f"Task {task} requires input_size > {max_dim}, got {input_size}.")

    colors = _condition_colors(cond_idx.keys())

    sim_scattered = use_sim_field and sim_n_warmup > 0   # quiver vs streamplot

    caches, all_speeds = [], []
    for spec in specs:
        cache, sp = _flow_panel_cache(
            model, spec, input_size, effective_x, xlim, ylim, n_grid,
            attention_input=attention_input, attention_scale=attention_scale,
            use_sim_field=use_sim_field, sim_n_warmup=sim_n_warmup,
            field_input_noise=field_input_noise, field_noise_K=field_noise_K,
            field_noise_seed=field_noise_seed, include_beta_in_field=include_beta_in_field,
            n_fp_seeds=n_fp_seeds, slow_tol=slow_tol,
            input_threshold=input_threshold, inactive_atol=inactive_atol,
        )
        caches.append(cache); all_speeds.append(sp)

    speed_vmax = np.percentile(np.concatenate(all_speeds), speed_percentile)

    n_panels = len(specs)
    fig = plt.figure(figsize=(n_panels * figsize_per_panel + 0.45, figsize_per_panel),
                     constrained_layout=False)
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(
        1, n_panels + 1,
        width_ratios=[1.0] * n_panels + [0.045],
        left=0.045, right=0.965, bottom=0.16, top=0.84, wspace=0.16,
    )
    axes = []
    for i in range(n_panels):
        ax_i = fig.add_subplot(gs[0, i]) if i == 0 else fig.add_subplot(gs[0, i], sharex=axes[0], sharey=axes[0])
        axes.append(ax_i)
    axes = np.asarray(axes, dtype=object)
    cax  = fig.add_subplot(gs[0, -1])

    last_hm = None

    for ax, cache in zip(axes, caches):
        last_hm = _render_flow_panel(
            ax, cache, speed_vmax=speed_vmax, sim_scattered=sim_scattered,
            kappa_traj=kappa_traj, cond_idx=cond_idx, colors=colors,
            xlim=xlim, ylim=ylim, model=model,
            show_slow_manifold=show_slow_manifold, slow_manifold_thresh=slow_manifold_thresh,
            show_single_trials=show_single_trials, max_single_trials=max_single_trials,
            max_autonomous_conditions=max_autonomous_conditions,
        )
        ax.set_title(cache["spec"]["name"])
        ax.set_xlabel(r"$\kappa_0$")

    axes[0].set_ylabel(r"$\kappa_1$")

    handles, labels_list = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels_list.extend(l)
    unique = dict(zip(labels_list, handles))
    _leg = axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=7,
                          loc="lower right", handlelength=1.1)
    for _t in _leg.get_texts():
        _t.set_color("white")

    cbar = fig.colorbar(last_hm, cax=cax)
    if use_sim_field:
        cbar.set_label(r"$\|\Delta\kappa\|$ (simulation)")
    elif include_beta_in_field:
        cbar.set_label(r"$\beta\|\Psi(\kappa;x)-\kappa\|$")
    else:
        cbar.set_label(r"$\|\Psi(\kappa;x)-\kappa\|$")
    fig.suptitle(f"{task.upper()} low-rank dynamics: autonomous and frozen-input fields", y=0.965)
    fig.set_layout_engine("none")

    return fig, axes, {
        "task": task, "kappa": kappa_traj,
        "readouts": readouts.detach().cpu(), "rates": rates.detach().cpu(),
        "rec_inputs": rec_inputs.detach().cpu(), "effective_x": effective_x.detach().cpu(),
        "condition_indices": cond_idx, "condition_meta": meta,
        "panel_specs": specs, "caches": caches,
    }


# Canonical panel columns for the stacked multi-stage portrait: the Dual 8-condition layout, used
# for EVERY stage row so all rows have the same panels. Each entry is a frozen-input clamp (which
# input channels are held on) that is valid for ANY stage's model — the field only depends on the
# model + clamp, not on which task the model was trained on. A stage overlays its OWN task
# trajectories only where that condition exists in its trials (via per-stage `conds`).


def plot_stage_stacked_flow(
    stages, *, cue_on_go_input=False, cue_scale=1.0,
    attention_input=False, attention_scale=1.0,
    xlim=None, ylim=None, n_grid=151, n_grid_per_unit=None,
    n_fp_seeds=41, slow_tol=None, use_sim_field=False, sim_n_warmup=0,
    field_input_noise=0.0, figsize_per_panel=3.6, speed_percentile=98,
    include_beta_in_field=False, show_slow_manifold=False, slow_manifold_thresh=0.12,
    dual_mode="conditions", input_threshold=0.35, inactive_atol=0.35, suptitle=None,
):
    """Stacked κ-plane flow portrait: ONE ROW per training stage (e.g. dpa / naive / expert), all
    rows sharing the SAME canonical panel columns (the Dual 8-condition layout) and the SAME κ
    limits + speed colour scale. Reading a column top→bottom shows how that field/well evolves
    across stages. Each row overlays its own stage's task trajectories where that condition exists.

    `stages` is a list of dicts, each: {label, task, timing, model, inputs, targets (opt),
    condition_names (opt)}. `inputs`/`targets` are that stage's task trials (arrays or tensors)."""
    if model_missing := [i for i, s in enumerate(stages) if s.get("model") is None]:
        raise ValueError(f"stages {model_missing} missing a model")

    CANON = _canonical_flow_panels(cue_on_go_input, cue_scale)
    n_panels = len(CANON)

    # ---- per-stage: trajectories, condition indices, per-stage specs (canonical + own conds) ----
    rows, all_traj = [], []
    for st in stages:
        model = st["model"]; model.eval()
        task  = st["task"].lower(); timing = st["timing"]
        inputs  = torch.as_tensor(st["inputs"],  dtype=torch.float32)
        targets = None if st.get("targets") is None else torch.as_tensor(st["targets"], dtype=torch.float32)
        input_size = inputs.shape[-1]
        with torch.no_grad():
            readouts, rates, rec_inputs, effective_x = run_low_rank_with_effective_inputs(
                model, inputs, targets=targets)
        kappa_traj = project_rec_inputs_to_kappa(model, rec_inputs).detach().cpu().numpy()
        all_traj.append(kappa_traj[..., :2].reshape(-1, 2))
        cond_idx, meta = condition_indices_for_task(
            inputs.detach().cpu(), timing=timing, task=task,
            condition_names=st.get("condition_names"), dual_mode=dual_mode)
        native   = flow_specs_for_task(timing=timing, task=task, input_size=input_size,
                                       cond_idx=cond_idx, meta=meta, dual_mode=dual_mode,
                                       cue_on_go_input=cue_on_go_input, cue_scale=cue_scale)
        cond_map = {s["name"]: list(s.get("conds", [])) for s in native}
        specs    = [dict(name=c["name"], dims=c["dims"], value=c["value"],
                         conds=cond_map.get(c["name"], [])) for c in CANON]
        rows.append(dict(model=model, task=task, label=st["label"], input_size=input_size,
                         kappa_traj=kappa_traj, effective_x=effective_x,
                         cond_idx=cond_idx, colors=_condition_colors(cond_idx.keys()), specs=specs))

    # ---- common κ limits across ALL stages (symmetric, full range + pad, floored at ±1.5) ----
    cat = np.concatenate(all_traj, axis=0)
    if xlim is None:
        r0 = max(float(np.nanmax(np.abs(cat[:, 0]))) * 1.25, 1.5); xlim = (-r0, r0)
    if ylim is None:
        r1 = max(float(np.nanmax(np.abs(cat[:, 1]))) * 1.25, 1.5); ylim = (-r1, r1)
    if n_grid_per_unit is not None:
        n_grid = min(int(round(n_grid_per_unit * (xlim[1] - xlim[0]))), 401)

    # ---- fields + fixed points for every (stage row × canonical panel); global speed scale ----
    all_speeds = []
    for row in rows:
        row["caches"] = []
        for spec in row["specs"]:
            cache, sp = _flow_panel_cache(
                row["model"], spec, row["input_size"], row["effective_x"], xlim, ylim, n_grid,
                attention_input=attention_input, attention_scale=attention_scale,
                use_sim_field=use_sim_field, sim_n_warmup=sim_n_warmup,
                field_input_noise=field_input_noise, include_beta_in_field=include_beta_in_field,
                n_fp_seeds=n_fp_seeds, slow_tol=slow_tol,
                input_threshold=input_threshold, inactive_atol=inactive_atol)
            row["caches"].append(cache); all_speeds.append(sp)
    speed_vmax = np.percentile(np.concatenate(all_speeds), speed_percentile)

    # ---- render the n_rows × n_panels grid (shared axes, one colourbar spanning all rows) ----
    n_rows = len(rows)
    sim_scattered = use_sim_field and sim_n_warmup > 0
    fig = plt.figure(figsize=(n_panels * figsize_per_panel + 0.6, n_rows * figsize_per_panel + 0.3),
                     constrained_layout=False)
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(n_rows, n_panels + 1, width_ratios=[1.0] * n_panels + [0.05],
                          left=0.05, right=0.95, bottom=0.06, top=0.90, wspace=0.12, hspace=0.20)
    ax00, last_hm = None, None
    for r, row in enumerate(rows):
        row_axes = []
        for c, cache in enumerate(row["caches"]):
            ax = (fig.add_subplot(gs[r, c]) if ax00 is None
                  else fig.add_subplot(gs[r, c], sharex=ax00, sharey=ax00))
            if ax00 is None: ax00 = ax
            row_axes.append(ax)
            last_hm = _render_flow_panel(
                ax, cache, speed_vmax=speed_vmax, sim_scattered=sim_scattered,
                kappa_traj=row["kappa_traj"], cond_idx=row["cond_idx"], colors=row["colors"],
                xlim=xlim, ylim=ylim, model=row["model"],
                show_slow_manifold=show_slow_manifold, slow_manifold_thresh=slow_manifold_thresh)
            if r == 0:
                ax.set_title(cache["spec"]["name"])
            if r == n_rows - 1:
                ax.set_xlabel(r"$\kappa_0$")
            if c == 0:
                ax.set_ylabel(f"{row['label']} ({row['task'].upper()})\n" + r"$\kappa_1$")
        # per-row legend (conditions differ by stage) on the row's Autonomous panel
        handles, labels_list = [], []
        for ax in row_axes:
            h, l = ax.get_legend_handles_labels(); handles += h; labels_list += l
        uniq = dict(zip(labels_list, handles))
        if uniq:
            leg = row_axes[0].legend(uniq.values(), uniq.keys(), frameon=False, fontsize=7,
                                     loc="lower right", handlelength=1.1)
            for _t in leg.get_texts(): _t.set_color("white")

    cax  = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_hm, cax=cax)
    cbar.set_label(r"$\beta\|\Psi(\kappa;x)-\kappa\|$" if include_beta_in_field
                   else (r"$\|\Delta\kappa\|$ (simulation)" if use_sim_field
                         else r"$\|\Psi(\kappa;x)-\kappa\|$"))
    if suptitle:
        fig.suptitle(suptitle, y=0.965)
    fig.set_layout_engine("none")
    return fig, rows


# ---------------------------------------------------------------------------
# Simulation-based κ-plane flow field  (exact: includes W_fixed, full dynamics)
# ---------------------------------------------------------------------------

