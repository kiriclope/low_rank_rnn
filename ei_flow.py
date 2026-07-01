"""
ei_flow.py — binned drift-field flow figures (simulation-based), for any model:
EILowRankModel **and** LowRankModel (including a non-orthogonalised fixed-weight
backbone, where the analytic κ-reduction in plot_sweep is invalid). Same output
layout as the vanilla low-rank `plot_sweep.py --plots flow`.

Method (after NeuroFlame flow_dual_alt.org `compute_binned_flow_field`):
  - initialise a grid of states ON the manifold by injecting a current
    `X·n0 + Y·n1` along the readout vectors for `set_w` steps, then release;
  - run the full EI simulation under a clamped condition input;
  - bin (position, one-step displacement) over the κ-plane, average per bin
    (the empirical drift), mask low-count bins, Gaussian-smooth;
  - fixed points via KMeans on trajectory endpoints.

Panels per stage mirror src/dynamics.flow_specs_for_task:
  dpa  : Autonomous, A, B, C, D
  gng  : Autonomous, Go, NoGo, Cue
  dual : Autonomous, A, B, Go, NoGo, Cue, C, D

Saves to <out_root>/<sweep>/individual/<run_id>/flow/fp_<stage>.pdf
(stage ∈ {dpa, naive, expert}).

Usage:
  LD_PRELOAD=... python ei_flow.py --sweep_dir results/dual/sweep_ei_v1 \
      --out_root results/figures --device cuda:0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import EILowRankModel, LowRankModel, EISTPModel
from src.dynamics import make_input

STAGE_TASK = {"dpa": "dpa", "naive": "gng", "expert": "dual"}


def build_model(cfg, device):
    """Build EILowRankModel or LowRankModel (incl. fixed-weight backbone) from a config."""
    DT = cfg["dt_base"] * cfg["tau_rec_frac"]
    a, ar = DT / cfg["tau"], DT / (cfg["tau"] * cfg["tau_rec_frac"])
    if cfg.get("model_type", "lowrank") == "eistp":
        return EISTPModel(
            n_neuron=cfg["n_neuron"], K=cfg["eistp_K"], rank=cfg["rank"], gain=cfg["gain"],
            dt=DT, input_size=cfg["input_size"],
            stp_use=cfg.get("stp_U", 0.05), stp_tau_fac=cfg.get("stp_tau_f", 1.0),
            stp_tau_rec=cfg.get("stp_tau_d", 0.2), j_stp=cfg.get("j_stp", 1.0),
            lr_ini=cfg["low_rank_scale"], lr_scale=cfg.get("eistp_lr_scale", "N"),
            lr_additive=cfg.get("eistp_lr_additive", False),
            dense_cee=cfg.get("eistp_dense_cee", False),
            r_max=cfg.get("eistp_r_max", None),
            train_inputs=False, nonlinearity=cfg["nonlinearity"], device=device,
        )
    if cfg.get("model_type", "lowrank") == "ei":
        return EILowRankModel(
            input_size=cfg["input_size"], rank=cfg["rank"], n_exc=cfg["hidden_size"],
            n_inh=cfg["n_inh"], gain=cfg["gain"], alpha=a, alpha_rec=ar, noise=0.0,
            static_radius=cfg["static_radius"], low_rank_scale=cfg["low_rank_scale"],
            low_rank_full=cfg.get("low_rank_full", False),
            use_stp=cfg.get("use_stp", False), stp_U=cfg.get("stp_U", 0.2),
            stp_tau_f=cfg.get("stp_tau_f", 1.5), stp_tau_d=cfg.get("stp_tau_d", 0.3), stp_dt=DT,
            rwd=cfg["rwd"], rwd_scale=cfg["rwd_scale"], nonlinearity=cfg["nonlinearity"],
            device=device,
        )
    return LowRankModel(
        input_size=cfg["input_size"], hidden_size=cfg["hidden_size"], output_size=0,
        rank=cfg["rank"], gain=cfg["gain"], alpha=a, alpha_rec=ar, noise=0.0,
        rwd=cfg["rwd"], rwd_scale=cfg["rwd_scale"], nonlinearity=cfg["nonlinearity"],
        nl_gamma=cfg.get("nl_gamma", 0.0),
        use_fixed_weights=cfg.get("use_fixed_weights", False),
        fixed_weight_scale=cfg.get("fixed_weight_scale", 0.8),
        fixed_weight_orthogonalize=cfg.get("fixed_weight_orthogonalize", True),
        fixed_weight_sparsity=cfg.get("fixed_weight_sparsity", 1.0),
        use_unit_bias=cfg.get("use_unit_bias", False),
        unit_bias_trainable=cfg.get("unit_bias_trainable", True),
        device=device,
    )


def make_hidden_fn(m):
    """Recurrent map rates -> hidden, generic over EI and low-rank(+fixed-weight) models."""
    if hasattr(m, "_W_rec_eff"):           # EI: full effective recurrent matrix
        Wrec = m._W_rec_eff()
        return lambda r: r @ Wrec.T
    N = m.hidden_size; wf = getattr(m, "w_fixed", None)
    def f(r):                              # low-rank: m nᵀ/N (+ frozen backbone)
        h = (r @ m.n) @ m.m.T / N
        if wf is not None:
            h = h + r @ wf.T
        return h
    return f


def panels_for(task, isz, cue_scale, cue_on_go):
    cue_dim = 4 if cue_on_go else 6
    if task == "dpa":
        specs = [("Autonomous", None, 1.0), ("A", [0], 1.0), ("B", [1], 1.0),
                 ("C", [2], 1.0), ("D", [3], 1.0)]
    elif task == "gng":
        specs = [("Autonomous", None, 1.0), ("Go", [4], 1.0), ("NoGo", [5], 1.0),
                 ("Cue", [cue_dim], cue_scale)]
    else:  # dual
        specs = [("Autonomous", None, 1.0), ("A", [0], 1.0), ("B", [1], 1.0),
                 ("Go", [4], 1.0), ("NoGo", [5], 1.0), ("Cue", [cue_dim], cue_scale),
                 ("C", [2], 1.0), ("D", [3], 1.0)]
    out = []
    for name, dims, val in specs:
        out.append((name, None if dims is None else make_input(isz, dims, val)))
    return out


@torch.no_grad()
def _run_grid_eistp(m, cond_x, R, gsize, set_w, T, I0, device):
    """Grid flow for EISTPModel: two-timescale + Markram STP via model.update_dynamics.
    Inject I0·(X·n0 + Y·n1) into E (on top of the Ja0 baseline) for set_w steps to place
    the state on the κ-plane, then release to the condition (baseline + stimulus)."""
    ne, N = m.n_exc, m.hidden_size
    # Inject along the UNIT, orthogonalised readout vectors v0, v1. The trained ‖n‖ is large
    # and the net is near-critical, so bscale·n overdrives it → calibrate the per-axis scale
    # so the injected (held) state spans κ ≈ ±R along each axis (R is the κ-range to cover).
    v0 = m.n[:, 0] / m.n[:, 0].norm()
    v1 = m.n[:, 1] - (m.n[:, 1] @ v0) * v0; v1 = v1 / v1.norm()
    base1 = m.ext_base.view(1, -1)
    W_ee = m.W_EE_eff()

    @torch.no_grad()
    def _held_kappa(vhat, S):                              # κ after holding S·vhat for set_w steps
        r = torch.zeros(1, N, device=device); syn = torch.zeros(1, N, device=device)
        u, x = m.init_stp(1)
        inj = torch.zeros(1, N, device=device); inj[:, :ne] = S * vhat
        for _ in range(set_w):
            r, syn, u, x = m.update_dynamics(base1 + inj, r, syn, u, x, W_ee)
        k = (r[:, :ne] @ m.n / ne)[0]
        return float(k[0]), float(k[1])

    def _scale(vhat, axis):                                # S s.t. X=1 → κ_axis ≈ R
        chi = abs(_held_kappa(vhat, 50.0)[axis]) / 50.0    # susceptibility from a safe probe
        return float(min(R / max(chi, 1e-3), 1000.0))      # cap to avoid runaway
    S0, S1 = _scale(v0, 0), _scale(v1, 1)

    gx = torch.linspace(-1.0, 1.0, gsize, device=device)
    Xg, Yg = torch.meshgrid(gx, gx, indexing="ij")
    Xf, Yf = Xg.reshape(-1), Yg.reshape(-1); B = Xf.shape[0]
    grid_inj = torch.zeros(B, N, device=device)
    grid_inj[:, :ne] = Xf[:, None] * (S0 * v0)[None] + Yf[:, None] * (S1 * v1)[None]
    base = m.ext_base.view(1, -1).expand(B, N).clone()      # Ja0 baseline (E and I)
    cond = base.clone()
    if cond_x is not None:                                   # + stimulus current to E
        cond[:, :ne] = cond[:, :ne] + m.bscale * m.wi(cond_x.to(device).repeat(B, 1))
    rates = torch.zeros(B, N, device=device); syn = torch.zeros(B, N, device=device)
    u, x = m.init_stp(B)
    W_ee = m.W_EE_eff()
    traj = []
    for tt in range(T):
        drive = (base + grid_inj) if tt < set_w else cond
        rates, syn, u, x = m.update_dynamics(drive, rates, syn, u, x, W_ee)
        traj.append(rates[:, :ne] @ m.n / ne)
    return torch.stack(traj, 1).cpu().numpy(), set_w


@torch.no_grad()
def run_grid(m, cond_x, R, gsize, set_w, T, I0, device):
    if m.__class__.__name__ == "EISTPModel":
        return _run_grid_eistp(m, cond_x, R, gsize, set_w, T, I0, device)
    N = m.hidden_size
    ne = m.n.shape[0]                       # readout population (=N for low-rank, n_exc for EI)
    n_out = m.wi.out_features               # input pathway width (E units for EI)
    n0 = m.n[:, 0] / m.n[:, 0].norm(); n1 = m.n[:, 1] / m.n[:, 1].norm()
    gx = torch.linspace(-R, R, gsize, device=device)
    Xg, Yg = torch.meshgrid(gx, gx, indexing="ij")
    Xf, Yf = Xg.reshape(-1), Yg.reshape(-1); B = Xf.shape[0]
    grid_cur = torch.zeros(B, N, device=device)
    grid_cur[:, :ne] = I0 * (Xf[:, None] * n0[None] + Yf[:, None] * n1[None])
    cond = torch.zeros(B, N, device=device)
    if cond_x is not None:
        cond[:, :n_out] = m.wi(cond_x.to(device).repeat(B, 1))
    rec = torch.zeros(B, N, device=device); rates = torch.zeros(B, N, device=device)
    hidden_fn = make_hidden_fn(m); gain = m.gain
    ub = getattr(m, "unit_bias", None)      # low-rank per-unit bias (None for EI)
    use_stp = getattr(m, "use_stp", False)  # EI short-term plasticity on E presynapse
    ne_exc = getattr(m, "n_exc", None)
    u, x = m.init_stp(B) if use_stp else (None, None)
    Wrec = m._W_rec_eff() if (use_stp and hasattr(m, "_W_rec_eff")) else None
    traj = []
    for tt in range(T):
        drive = grid_cur if tt < set_w else cond
        if use_stp:                         # gate E presynaptic rates by (u·x)/U, evolve u,x
            gate = (u * x) / m.stp_U
            r_eff = torch.cat([rates[:, :ne_exc] * gate, rates[:, ne_exc:]], dim=1)
            h = r_eff @ Wrec.T
            u, x = m.stp_update(rates[:, :ne_exc], u, x)
        else:
            h = hidden_fn(rates)
        rec = m.exp_alpha_rec * rec + (1 - m.exp_alpha_rec) * h
        pre = gain * (drive + rec)
        if ub is not None:
            pre = pre + ub
        rates = m.exp_alpha * rates + (1 - m.exp_alpha) * m.nonlinearity(pre)
        traj.append(rates[:, :ne] @ m.n / ne)
    return torch.stack(traj, 1).cpu().numpy(), set_w


def binned_field(traj, window, size=60, sigma=1.0, min_count=1):
    x = traj[:, window:, 0]; y = traj[:, window:, 1]
    dx = np.diff(x, 1); dy = np.diff(y, 1)
    x0 = x[:, :-1].ravel(); y0 = y[:, :-1].ravel(); du = dx.ravel(); dv = dy.ravel()
    ok = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(du) & np.isfinite(dv)
    x0, y0, du, dv = x0[ok], y0[ok], du[ok], dv[ok]
    if len(x0) == 0:
        return None
    xlo, xhi = np.percentile(x0, [1, 99]); ylo, yhi = np.percentile(y0, [1, 99])
    rad = max(abs(xlo), abs(xhi), abs(ylo), abs(yhi), 1e-3) * 1.1
    xe = np.linspace(-rad, rad, size); ye = np.linspace(-rad, rad, size)
    su, _, _ = np.histogram2d(y0, x0, bins=[ye, xe], weights=du)
    sv, _, _ = np.histogram2d(y0, x0, bins=[ye, xe], weights=dv)
    cnt, _, _ = np.histogram2d(y0, x0, bins=[ye, xe])
    with np.errstate(invalid="ignore", divide="ignore"):
        ui = su / cnt; vi = sv / cnt
    bad = cnt < min_count; ui[bad] = np.nan; vi[bad] = np.nan
    u0 = np.nan_to_num(ui); v0 = np.nan_to_num(vi)
    if sigma:
        u0 = gaussian_filter(u0, sigma); v0 = gaussian_filter(v0, sigma)
        cnt_s = gaussian_filter(cnt.astype(float), sigma)
    else:
        cnt_s = cnt.astype(float)
    sup = cnt_s > 0
    ui = np.where(sup, u0, np.nan); vi = np.where(sup, v0, np.nan)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    return xc, yc, ui, vi


def _components(pts, link_tol):
    """Single-linkage connected components: index arrays of points within link_tol chains."""
    n = len(pts)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.where(d < link_tol)[0]:
            parent[find(i)] = find(j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return [np.array(v) for v in comps.values()]


def fixed_points(traj, vel_tol=0.003, recent=12, link_tol=0.5, point_tol=0.9, min_pts=10,
                 k=None, merge_tol=None):
    """Honest attractor detection — distinguishes isolated point attractors from a
    *continuous* (ring / slow-manifold) attractor instead of dropping arbitrary dots on it.

    Steps:
      1. Keep only *converged* endpoints (recent per-step speed
         |κ(T) − κ(T−recent)|/recent < vel_tol), so diverged / fast-moving points drop out.
      2. Cluster the converged cloud by single-linkage (link_tol); discard components
         with < min_pts points (noise / saddle stragglers).
      3. Classify each component by spatial *extent* (max radius from its centroid):
         extent < point_tol  → genuine **point attractor**  (return centroid as a dot);
         extent ≥ point_tol  → **continuous manifold** (ring/arc) — return its member
         points to be drawn as a locus, NOT a single arbitrary dot.

    Returns (point_fps [p,2], manifold_pts [q,2] | None, all_endpoints [n,2]).
    `k`/`merge_tol` are accepted for backward-compatible call sites and ignored.
    """
    ep_all = traj[:, -1, :]
    r = min(recent, traj.shape[1] - 1)
    speed = np.linalg.norm(traj[:, -1, :] - traj[:, -1 - r, :], axis=1) / r
    keep = (np.isfinite(ep_all).all(1) & (np.abs(ep_all) < 20).all(1) & (speed < vel_tol))
    ep = ep_all[keep]
    if len(ep) < min_pts:
        return np.empty((0, 2)), None, ep_all
    point_fps, manifold = [], []
    for c in _components(ep, link_tol):
        if len(c) < min_pts:
            continue
        p = ep[c]
        ctr = p.mean(0)
        extent = np.linalg.norm(p - ctr, axis=1).max()
        if extent < point_tol:
            point_fps.append(ctr)
        else:
            manifold.append(p)
    pts = np.array(point_fps) if point_fps else np.empty((0, 2))
    man = np.concatenate(manifold, 0) if manifold else None
    return pts, man, ep_all


def make_stage_figure(m, task, run_id, stage, out_dir, device,
                      R=8.0, gsize=28, set_w=40, T=1333, I0=1.0,
                      cue_scale=2.0, cue_on_go=True, style="magma"):
    isz = m.input_size
    # EISTPModel lives on a much larger κ range (~±10) and uses its own dynamics → wider
    # grid, shorter T, and fixed-point tolerances scaled up ~10× to the κ scale.
    is_eistp = m.__class__.__name__ == "EISTPModel"
    if is_eistp:
        R, T = 15.0, 600
        fp_kw = dict(vel_tol=0.03, recent=12, link_tol=2.0, point_tol=4.0, min_pts=10)
    else:
        fp_kw = {}
    specs = panels_for(task, isz, cue_scale, cue_on_go)
    npan = len(specs)
    fig, axes = plt.subplots(1, npan, figsize=(npan * 3.2, 3.4))
    if npan == 1:
        axes = [axes]
    for ax, (nm, x) in zip(axes, specs):
        traj, w = run_grid(m, x, R, gsize, set_w, T, I0, device)
        bf = binned_field(traj, w)
        if bf is not None:
            xc, yc, ui, vi = bf
            sp = np.sqrt(ui ** 2 + vi ** 2)
            fin = np.isfinite(sp)
            if style == "binned":           # original: coolwarm z-scored speed, black streams
                spz = (sp - np.nanmean(sp)) / (np.nanstd(sp) + 1e-6)
                vmin, vmax = (np.nanpercentile(spz[np.isfinite(spz)], [5, 95]) if fin.any() else (-1, 1))
                ax.pcolormesh(xc, yc, spz, cmap="coolwarm", shading="auto",
                              norm=mpl.colors.Normalize(vmin, vmax))
                stream_color = (0, 0, 0, 0.5)
            else:                           # "magma": raw speed (vmax=98th pct), white streams
                vmax = np.nanpercentile(sp[fin], 98) if fin.any() else 1.0
                ax.pcolormesh(xc, yc, sp, cmap="magma", shading="auto",
                              vmax=vmax, rasterized=True)
                stream_color = "white"
            ax.streamplot(xc, yc, np.nan_to_num(ui), np.nan_to_num(vi),
                          color=stream_color, density=1.05, linewidth=0.6, arrowsize=0.7)
        fp, man, ep = fixed_points(traj, **fp_kw)
        cloud_c = "0.7" if style != "binned" else "k"   # light cloud on dark magma background
        if len(ep):
            ax.scatter(ep[:, 0], ep[:, 1], s=5, color=cloud_c, alpha=0.12, zorder=3)
        if man is not None and len(man):          # continuous attractor → draw the locus
            ax.scatter(man[:, 0], man[:, 1], s=14, color="orange",
                       edgecolors="k", linewidths=0.2, alpha=0.85, zorder=4)
        if len(fp):                               # isolated point attractors → dots
            ax.plot(fp[:, 0], fp[:, 1], "o", color="lime", ms=10, mec="k", zorder=5)
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_title(nm); ax.set_xlabel(r"$\kappa_0$"); ax.set_aspect("equal")
    axes[0].set_ylabel(r"$\kappa_1$")
    fig.suptitle(f"{run_id} — {stage} ({task.upper()}) — EI binned flow", y=1.02)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fp_{stage}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out_root", default="results/figures")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--style", choices=["magma", "binned"], default="magma",
                    help="magma: vanilla-look (magma speed + white streams); binned: coolwarm z-score")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    sweep = os.path.basename(args.sweep_dir.rstrip("/"))

    entries = [json.loads(l) for l in open(os.path.join(args.sweep_dir, "results.jsonl"))
               if l.strip() and json.loads(l).get("status") == "ok"]
    for e in entries:
        rid = e["run_id"]
        if args.run_ids and rid not in args.run_ids:
            continue
        cfg = e["config"]
        cue_on_go = bool(cfg.get("cue_on_go_input", True))
        cue_scale = float(cfg.get("cue_scale", 2.0))
        for stage, task in STAGE_TASK.items():
            ckpt = os.path.join(args.sweep_dir, rid, f"{stage}_{rid}.pth")
            if not os.path.exists(ckpt):
                print(f"  [skip] {rid}/{stage}: no checkpoint")
                continue
            m = build_model(cfg, device)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
            m.eval()
            out_dir = os.path.join(args.out_root, sweep, "individual", rid, "flow")
            path = make_stage_figure(m, task, rid, stage, out_dir, device,
                                     cue_scale=cue_scale, cue_on_go=cue_on_go, style=args.style)
            print(f"  saved {path}", flush=True)
            del m
            if "cuda" in str(device):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
