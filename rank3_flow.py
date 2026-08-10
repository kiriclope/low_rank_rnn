#!/usr/bin/env python
"""
rank3_flow.py — 3D fixed-point flow portraits for RANK-3 low-rank RNNs.

The role split is κ0 = sample memory, κ1 = gng memory, κ2 = action (lick / pairing). The reduced
field F(κ) = Ψ(κ) − κ is genuinely 3-D, so a 2-D streamplot needs a projection. This tool does the
CORRECT thing (option B): a real 3-D fixed-point search (grid + scipy.root on 3-vectors, reusing the
rank-general `low_rank_field_np` / `low_rank_jacobian_flow_np`), classification by the 3×3 Jacobian
eigenvalues, then renders the three pairwise-plane slices (κ0κ1, κ0κ2, κ1κ2) with the third coord
fixed at the dominant-attractor value and every real fixed point PROJECTED onto the plane.

One row per input condition. The **κ1–κ2 panel under the cue** is the nogo diagnosis: it shows
whether the negative-κ1 (nogo) memory actually holds κ2 (action) down, or the cue drives κ2 to lick
regardless — the signature of the collapsed nogo we see in the rank-3 sweep.

Standalone (does NOT touch the rank-2-locked `plot_task_flow_fields`).

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python rank3_flow.py \
      --sweep_dir results/dual/sweep_rank3 --out_root results/figures \
      [--run_ids s0_rank3] [--xlim 2.5] [--stage expert] [--conditions autonomous cue]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation_probe import load_run, discover_run_ids
from src.dynamics import low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np

plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
KLAB   = {0: r"$\kappa_0$ (sample)", 1: r"$\kappa_1$ (gng)", 2: r"$\kappa_2$ (action)"}
PLANES = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]        # (x_dim, y_dim, fixed_dim)


def _draw_fps(ax, fps, labs, dx, dy):
    """Project the real 3-D fixed points onto the (dx,dy) plane using the rank-2 marker convention
    (cyan filled-o attractor / cyan x saddle / open-cyan-^ repeller / open-gold-s marginal)."""
    for lab in ("attractor", "saddle", "repeller", "marginal"):
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
        else:
            ax.scatter(X, Y, s=60, marker="o", facecolors="cyan", edgecolors="cyan",
                       linewidths=1.1, zorder=10, label="attractor")


# The canonical DUAL input-driven conditions (mirrors dynamics.flow_specs_for_task for the dual
# task): each drives one input channel at input_scale, except Cue which rides the go channel (4) at
# cue_scale. Attention (last channel) is always on if the run uses it. Autonomous = attention only.
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


def build_field(p, phi_name, ff, noise_sigma=0.0):
    """jax reduced field  F(κ)=Ψ(κ)−κ  for a single input condition ff, as a (batch,rank)→(batch,rank)
    callable — shared by the brainpy fixed-point search AND the jax flow relaxation.

    noise_sigma>0 applies the EXACT input-noise mean field for a Gaussian-CDF φ: ⟨φ(a+η)⟩=φ(a/√(1+c·s²)),
    sᵢ²=g²Aᵢ²σ²‖wᵢ‖², c=1(lif)/2(erf)/2π(lif_sc) — i.e. noise compresses the drive (effective-gain drop).
    Non-Gaussian φ: noise ignored here (matches low_rank_field_np's Taylor fallback being off in jax)."""
    import jax, jax.numpy as jnp
    PHI = {"tanh": jnp.tanh, "relu": lambda x: jnp.maximum(x, 0.0),
           "erf": jax.scipy.special.erf, "softplus": jax.nn.softplus, "elu": jax.nn.elu,
           # match src/models.py: lif = Gaussian CDF ½(1+erf(x/√2)); lif_sc rescaled by √π
           "lif":    lambda x: 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0))),
           "lif_sc": lambda x: 0.5 * (1.0 + jax.scipy.special.erf(x * jnp.sqrt(jnp.pi)))}
    NC  = {"lif": 1.0, "erf": 2.0, "lif_sc": 2.0 * np.pi}
    M  = jnp.asarray(p["M"]); Nv = jnp.asarray(p["Nvec"]); g = float(p["gain"]); N = p["M"].shape[0]
    drive = jnp.asarray(p["Ai"] * (np.asarray(ff, float) @ p["Wi"].T + p["bi"]))
    phi = PHI.get(phi_name, jnp.tanh)
    c = NC.get(phi_name)
    denom = None
    if noise_sigma and noise_sigma > 0.0 and c is not None:
        s2 = (g ** 2) * (np.asarray(p["Ai"], float) ** 2) * (float(noise_sigma) ** 2) \
             * np.sum(np.asarray(p["Wi"], float) ** 2, axis=1)
        denom = jnp.asarray(np.sqrt(1.0 + c * s2))                # (N,)
    def field(k):
        a = g * (drive[None, :] + k @ M.T)
        if denom is not None:
            a = a / denom[None, :]
        return phi(a) @ Nv / N - k
    return field


def find_fps_brainpy(field_fn, p, ff, xlim=3.0, grid=9, slow_tol=1e-7, marg=0.04, num_opt=1500, noise_sigma=0.0):
    """3-D fixed points via brainpy SlowPointFinder (Adam GD on ½‖F‖² over a grid³ of candidates,
    on jax) — same finder as the rank-2 tools. Classified by the analytic 3×3 flow Jacobian."""
    import brainpy as bp, brainpy.math as bm
    gx = np.linspace(-xlim, xlim, grid)
    cand = np.array(np.meshgrid(gx, gx, gx)).reshape(3, -1).T.astype(np.float32)
    fdr = bp.analysis.SlowPointFinder(f_cell=field_fn, f_type="continuous")
    fdr.find_fps_with_gd_method(candidates=bm.asarray(cand), tolerance=1e-9, num_batch=400,
        num_opt=num_opt, optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.05, 1, 0.9999)))
    fdr.filter_loss(slow_tol); fdr.keep_unique(tolerance=8e-2)
    fps = np.asarray(fdr.fixed_points).reshape(-1, 3)
    labs = []
    for fp in fps:
        ev = np.sort(np.linalg.eigvals(low_rank_jacobian_flow_np(p, fp, ff_input=ff, noise_sigma=noise_sigma)).real)
        lo, hi = ev[0], ev[-1]
        labs.append("attractor" if hi < -marg else "repeller" if lo > marg else
                    "saddle" if (hi > marg and lo < -marg) else "marginal")
    return fps, np.array(labs)


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

    # pass 1 — compute the real 3-D fixed points + the sliced field/speed for every (row, plane).
    # The streamplot is a 2-D SLICE of the true 3-D field: the off-plane coord (df) is held fixed.
    # slice_mode "attractor" → hold df at the median of the attractor FPs' df-coord, so the slice
    # plane passes THROUGH the dominant attractor(s) and the streamlines converge onto the projected
    # markers (fallback: median of all FPs, else 0). slice_mode "zero" → hold df at 0 (resting).
    phi_name = cfg.get("nonlinearity", "tanh")
    cells, all_speed, total_fps = {}, [], 0
    for r, (cname, ff) in enumerate(conds):
        field_fn   = build_field(p, phi_name, ff, noise_sigma=sigma)   # jax field (noise-corrected if σ>0)
        fps, labs  = find_fps_brainpy(field_fn, p, ff, xlim=xlim, grid=n_seeds,
                                      slow_tol=slow_tol, marg=marg, noise_sigma=sigma)
        att        = fps[labs == "attractor"] if len(fps) else np.zeros((0, 3))
        total_fps += len(fps)
        for c, (dx, dy, df) in enumerate(PLANES):
            if slice_mode == "adiabatic":
                dX, dY = plane_field(field_fn, dx, dy, df, GX, GY, fps)
                c_slice = None                                # off-plane coord varies over the grid
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
    speed_vmax = float(np.percentile(np.concatenate(all_speed), 98))   # shared vmax (rank-2 convention)

    # pass 2 — draw with the rank-2 look: magma speed pcolormesh + white streamlines + cyan FPs.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out_root", default="results/figures")
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDS, choices=ALL_CONDS,
                    help="dual input-driven conditions (rows); default = all 8")
    ap.add_argument("--xlim", type=float, default=3.0)
    ap.add_argument("--n_seeds", type=int, default=11, help="brainpy candidate grid per axis (n_seeds³)")
    ap.add_argument("--stage", default="expert", choices=["dpa", "naive", "expert"])
    ap.add_argument("--slice", dest="slice_mode", default="adiabatic",
                    choices=["adiabatic", "attractor", "zero"],
                    help="off-plane coord: 'adiabatic' (relax to its nullcline per grid point so "
                         "streamlines converge onto every FP — default), or a FLAT slice at the "
                         "median-attractor value ('attractor') or 0 ('zero')")
    ap.add_argument("--slow_tol", type=float, default=1e-7,
                    help="max squared speed ‖F‖² kept by the brainpy finder. 1e-7 = exact FPs; raise "
                         "(e.g. 1e-3) to keep SLOW points that trace near-marginal / ring / slow-manifold "
                         "structure a root-finder misses")
    ap.add_argument("--marg", type=float, default=0.04,
                    help="|Re eigenvalue| below this → 'marginal' (a slow-manifold direction)")
    ap.add_argument("--noise", action="store_true",
                    help="render the INPUT-NOISE mean field at the run's own σ (exact Gaussian "
                         "resummation φ(a/√(1+c·s²))) — shows which fixed points the noise destabilizes")
    args = ap.parse_args()

    sweep = os.path.basename(os.path.normpath(args.sweep_dir))
    rids = args.run_ids or discover_run_ids(args.sweep_dir)
    suffix = "_noise" if args.noise else ""
    print(f"rank3_flow: {sweep}  stage={args.stage}  conditions={args.conditions}  noise={args.noise}  runs={rids}")
    for rid in rids:
        out = os.path.join(args.out_root, sweep, "individual", rid, "flow", f"rank3_{args.stage}{suffix}")
        try:
            k = render_run(args.sweep_dir, rid, out, args.conditions, args.xlim, args.n_seeds,
                           args.stage, args.slice_mode, args.slow_tol, args.marg, use_run_noise=args.noise)
            print(f"  {rid}: {k} fixed points  ->  {out}.png")
        except Exception as e:
            print(f"  {rid}: ERROR {e}")


if __name__ == "__main__":
    main()
