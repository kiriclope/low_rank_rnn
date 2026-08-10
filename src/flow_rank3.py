"""Rank-3 flow rendering — the three pairwise κ-plane slices (κ0κ1, κ0κ2, κ1κ2) with the rank-2 look
(magma speed + white streamlines + cyan fixed points), one row per input condition. Uses the shared
rank-general FP finder (`find_fixed_points`, brainpy backend) and jax field (`build_jax_field`) from
flow_fixedpoints. CLI wrapper: `rank3_flow.py`."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bifurcation_probe import load_run, discover_run_ids
from .flow_field import low_rank_numpy_params, low_rank_field_np
from .flow_fixedpoints import build_jax_field, find_fixed_points

plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
KLAB   = {0: r"$\kappa_0$ (sample)", 1: r"$\kappa_1$ (gng)", 2: r"$\kappa_2$ (action)"}
PLANES = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]        # (x_dim, y_dim, fixed_dim)


def _draw_fps(ax, fps, labs, dx, dy):
    """Project the real 3-D fixed points onto the (dx,dy) plane using the rank-2 marker convention
    (cyan filled-o attractor / cyan x saddle / open-cyan-^ repeller / open-gold-s marginal)."""
    for lab in ("attractor", "slow_attractor", "saddle", "repeller", "marginal"):
        if len(labs) == 0:
            continue
        pts = fps[labs == lab]
        if not len(pts):
            continue
        X, Y = pts[:, dx], pts[:, dy]
        if lab == "saddle":
            ax.scatter(X, Y, s=60, marker="x", color="cyan", linewidths=1.5, zorder=10, label="saddle")
        elif lab == "marginal":
            ax.scatter(X, Y, s=55, marker="s", facecolors="none", edgecolors="gold",
                       linewidths=1.6, zorder=10, label="marginal")
        elif lab == "repeller":
            ax.scatter(X, Y, s=60, marker="^", facecolors="none", edgecolors="cyan",
                       linewidths=1.1, zorder=10, label="repeller")
        else:  # attractor / slow_attractor
            ax.scatter(X, Y, s=60, marker="o", facecolors="cyan", edgecolors="cyan",
                       linewidths=1.1, zorder=10, label="attractor")


COND_CHANNEL = {"Autonomous": None, "A": 0, "B": 1, "Go": 4, "NoGo": 5, "Cue": 4, "C": 2, "D": 3}
ALL_CONDS    = ["Autonomous", "A", "B", "Go", "NoGo", "Cue", "C", "D"]


def conditions_for(cfg, names):
    isz, sc, cs = cfg["input_size"], cfg.get("input_scale", 1.0), cfg.get("cue_scale", 1.0)
    out = []
    for nm in names:
        if nm not in COND_CHANNEL:
            continue
        ff = np.zeros(isz, dtype=float)
        if cfg.get("attention_input", False):
            ff[-1] = 1.0
        ch = COND_CHANNEL[nm]
        if ch is not None:
            ff[ch] += (cs * sc if nm == "Cue" else sc)
        out.append((nm, ff))
    return out


def plane_field(field_fn, dx, dy, df, GX, GY, fps, iters=250, step=0.5):
    """Adiabatic 2-D projection (jax, jit): at every grid point relax the off-plane coord df to its
    nullcline (F_df=0), seeded from the NEAREST fixed point's df-value so each attractor's basin
    follows its own branch. The surface passes THROUGH every FP, so streamlines converge onto the
    projected markers. Runs on jax → fast even at high grid resolution."""
    import jax, jax.numpy as jnp
    n = GX.shape[0]
    gx, gy = GX.ravel(), GY.ravel()
    K = jnp.zeros((n * n, 3)).at[:, dx].set(jnp.asarray(gx)).at[:, dy].set(jnp.asarray(gy))
    if len(fps):
        d2 = (gx[:, None] - fps[:, dx]) ** 2 + (gy[:, None] - fps[:, dy]) ** 2
        K = K.at[:, df].set(jnp.asarray(fps[:, df][np.argmin(d2, axis=1)]))
    relax = jax.jit(lambda K: jax.lax.fori_loop(
        0, iters, lambda i, K: K.at[:, df].add(step * field_fn(K)[:, df]), K))
    F = np.asarray(field_fn(relax(K)))
    return F[:, dx].reshape(n, n), F[:, dy].reshape(n, n)


def render_run(sweep_dir, rid, out_path, conditions, xlim=3.0, n_seeds=11, stage="expert",
               slice_mode="adiabatic", slow_tol=1e-7, marg=0.04, use_run_noise=False):
    m, cfg = load_run(sweep_dir, rid, stage=stage)
    p = low_rank_numpy_params(m)
    # input-noise σ (exact Gaussian resummation): σ_eff = noise·√(β(2−β)), β = 1−e^{−α} (from params)
    sigma = 0.0
    if use_run_noise:
        beta  = float(p["beta"])
        sigma = float(cfg.get("noise", 0.0)) * float(np.sqrt(beta * (2.0 - beta)))
    conds = conditions_for(cfg, conditions)
    n = 121
    ax_ = np.linspace(-xlim, xlim, n)
    GX, GY = np.meshgrid(ax_, ax_)

    phi_name = cfg.get("nonlinearity", "tanh")
    cells, all_speed, total_fps = {}, [], 0
    for r, (cname, ff) in enumerate(conds):
        field_fn  = build_jax_field(p, phi_name, ff, noise_sigma=sigma)          # jax field for slices
        fps, labs = find_fixed_points(p, ff, phi_name=phi_name, rank=3, backend="brainpy",
                                      box=xlim, n_seeds=n_seeds, noise_sigma=sigma,
                                      slow_tol=slow_tol, marg=marg)               # SHARED finder
        att        = fps[labs == "attractor"] if len(fps) else np.zeros((0, 3))
        total_fps += len(fps)
        for c, (dx, dy, df) in enumerate(PLANES):
            if slice_mode == "adiabatic":
                dX, dY = plane_field(field_fn, dx, dy, df, GX, GY, fps)
                c_slice = None
            else:
                c_slice = 0.0 if slice_mode == "zero" else (
                    float(np.median(att[:, df])) if len(att) else
                    (float(np.median(fps[:, df])) if len(fps) else 0.0))
                K = np.zeros((n, n, 3))
                K[..., dx], K[..., dy], K[..., df] = GX, GY, c_slice
                F = low_rank_field_np(p, K, ff_input=ff, noise_sigma=sigma)
                dX, dY = F[..., dx], F[..., dy]
            speed = np.hypot(dX, dY)
            cells[(r, c)] = dict(cname=cname, dx=dx, dy=dy, df=df, c_slice=c_slice,
                                 dX=dX, dY=dY, speed=speed, fps=fps, labs=labs)
            all_speed.append(speed.reshape(-1))
    speed_vmax = float(np.percentile(np.concatenate(all_speed), 98))

    nrow, ncol = len(conds), 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol + 0.5, 4.2 * nrow), squeeze=False)
    hm = None
    for (r, c), cell in cells.items():
        ax = axes[r][c]
        hm = ax.pcolormesh(GX, GY, cell["speed"], shading="auto", cmap="magma",
                           vmax=speed_vmax, rasterized=True, zorder=0)
        ax.streamplot(ax_, ax_, cell["dX"], cell["dY"], color="white", density=1.05,
                      linewidth=0.70, arrowsize=0.80, zorder=2)
        _draw_fps(ax, cell["fps"], cell["labs"], cell["dx"], cell["dy"])
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(KLAB[cell["dx"]])
        ax.set_ylabel((f"{cell['cname']}\n" if c == 0 else "") + KLAB[cell["dy"]])
        ttl = (r"$\kappa_%d$ adiabatic" % cell["df"]) if cell["c_slice"] is None else \
              (r"slice $\kappa_%d = %+.2f$" % (cell["df"], cell["c_slice"]))
        ax.set_title(ttl, fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.09, top=0.9, wspace=0.28, hspace=0.32)
    cax = fig.add_axes([0.915, 0.12, 0.014, 0.72])
    fig.colorbar(hm, cax=cax, label=r"speed $\|F\|$")
    handles = [Line2D([0], [0], ls="", marker="o", mfc="cyan", mec="cyan", ms=8, label="attractor"),
               Line2D([0], [0], ls="", marker="x", color="cyan", ms=8, mew=1.5, label="saddle"),
               Line2D([0], [0], ls="", marker="^", mfc="none", mec="cyan", ms=8, label="repeller"),
               Line2D([0], [0], ls="", marker="s", mfc="none", mec="gold", ms=8, label="marginal")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.48, 0.99))
    noise_tag = (f"  ·  NOISE-corrected mean field σ={sigma:.2f} (input-noise gain compression)"
                 if sigma > 0 else "")
    fig.suptitle(f"{os.path.basename(os.path.normpath(sweep_dir))} · {rid} · {stage}  —  rank-3 flow "
                 f"(rows = input condition, cols = κ-plane; real 3-D fixed points projected){noise_tag}",
                 fontsize=11, y=1.005)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out_path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return total_fps
