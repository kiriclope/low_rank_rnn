#!/usr/bin/env python
"""
bifurcation_flows.py — real reduced-field κ-plane flow figures for low-rank sweep runs.

For each run it plots the AUTONOMOUS reduced vector field  F(κ) = Ψ(κ) − κ
(computed from the trained populations m, n and the run's attention input) as a magma
streamplot, overlays the classified fixed points (attractor / saddle / repeller) and the
F₁=0 decision nullcline, shades the no-lick half-plane (κ₁<0), and labels g·λ₁.

This is the per-run flow that shows the well geometry the probe tabulates: two isolated
attractors at κ₁<0 (isolated low wells) vs a 3-well/U ring. Deterministic clean-input field
by default; --field_noise renders the noise-averaged field E_x[Ψ] instead.

Complements plot_sweep.py's analytic flow (same field) — this one is standalone, labels
g·λ₁, and draws the nullcline, so it's the right tool for a focused bifurcation read.

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python bifurcation_flows.py \
      --sweep_dir results/dual/sweep_gainscan --out_root results/figures \
      [--run_ids s0_g05 s0_g10 s0_g15] [--xlim 2.0] [--field_noise]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dynamics import low_rank_numpy_params, low_rank_field_np
from bifurcation_probe import (load_run, self_gains, autonomous_ff, find_wells,
                               find_wells_brainpy, discover_run_ids)


def _wells(m, cfg, xlim, finder, slow_tol=1e-7, marg_tol=0.04):
    """Fixed points via the chosen finder. brainpy (default) uses SlowPointFinder + analytic
    Jacobian; falls back to scipy if jax/brainpy is unavailable or errors."""
    if finder == "brainpy":
        try:
            return find_wells_brainpy(m, cfg, xlim=max(xlim, 2.5), slow_tol=slow_tol, marg_tol=marg_tol)
        except Exception as e:
            print(f"    [brainpy finder unavailable ({e}); falling back to scipy]")
    return find_wells(m, cfg, xlim=max(xlim, 2.5))

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})

STYLE = {"attractor": dict(mfc="lime", mec="k", marker="o", ms=11),
         "saddle":    dict(mfc="orange", mec="k", marker="s", ms=8),
         "repeller":  dict(mfc="white", mec="crimson", marker="o", ms=8, mew=1.5),
         "marginal":  dict(mfc="0.6", mec="0.35", marker="o", ms=4.5, mew=0.6)}


def _ff_field(cfg, noise_sigma, K=16):
    """Frozen field input; if noise_sigma>0 return K noisy draws for the noise-averaged field."""
    ff = autonomous_ff(cfg)
    if not noise_sigma:
        return ff
    return ff[None, :] + noise_sigma * np.random.default_rng(0).standard_normal((K, ff.size))


def render_run(sweep_dir, rid, out_path, xlim=2.0, field_noise=False, finder="brainpy",
               slow_tol=1e-7, marg_tol=0.04, density=1.6, stage="expert"):
    m, cfg = load_run(sweep_dir, rid, stage=stage)
    p = low_rank_numpy_params(m)
    gl1 = float(self_gains(m)[1, 1])
    sig = cfg["noise"] * np.sqrt(1 - np.exp(-2 * 0.075)) if field_noise else 0.0
    ff = _ff_field(cfg, sig)

    n = 120
    ax_ = np.linspace(-xlim, xlim, n)
    K0, K1 = np.meshgrid(ax_, ax_)
    F = low_rank_field_np(p, np.stack([K0, K1], axis=-1), ff_input=ff)
    d0, d1 = F[..., 0], F[..., 1]
    speed = np.hypot(d0, d1)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.axhspan(-xlim, 0, color="0.85", alpha=0.30, zorder=0)
    # faint SLOW-field background: dark where the field is slow (near fixed points / manifolds),
    # so weak nodes and saddles are legible regardless of where streamlines seed.
    ax.imshow(np.log1p(speed), extent=[-xlim, xlim, -xlim, xlim], origin="lower",
              cmap="Greys", alpha=0.28, zorder=0.5, aspect="auto")
    ax.streamplot(ax_, ax_, d0, d1, color=speed, cmap="magma",
                  density=density, linewidth=0.7, arrowsize=0.7)
    ax.axhline(0, color="0.5", lw=0.7, ls=":")
    ax.axvline(0, color="0.5", lw=0.7, ls=":")
    ax.contour(K0, K1, d1, levels=[0], colors="deepskyblue", linewidths=1.8)
    wells = [(f, k) for f, k in _wells(m, cfg, xlim, finder, slow_tol, marg_tol)
             if abs(f[0]) <= xlim and abs(f[1]) <= xlim]
    # once the memory is resolved as a genuine BISTABLE PAIR of point attractors — one at κ0>0 and
    # one at κ0<0 (the A/B sample states), not the nogo pole — the marginal slow-manifold samples
    # are redundant clutter → drop them. If only one side is a point attractor (the other memory
    # state is still a slow manifold), keep the marginals so that memory is visible.
    mem_pos = any(k == "attractor" and f[0] >  0.6 for f, k in wells)
    mem_neg = any(k == "attractor" and f[0] < -0.6 for f, k in wells)
    if mem_pos and mem_neg:
        wells = [(f, k) for f, k in wells if k != "marginal"]
    for f, kind in wells:
        ax.plot(f[0], f[1], zorder=6, **STYLE.get(kind, STYLE["saddle"]))
    ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim); ax.set_aspect("equal")
    ax.set_xlabel(r"$\kappa_0$  (memory)"); ax.set_ylabel(r"$\kappa_1$  (decision)")
    ttl = f"{rid}" + (f" · {stage}" if stage != "expert" else "") + \
          f"   $g\\lambda_1={gl1:.2f}$" + ("   (noise-avg)" if field_noise else "")
    ax.set_title(ttl, loc="left", fontsize=11.5)
    ax.legend(handles=[Line2D([0], [0], **{**STYLE[k], "ls": ""}, label=k)
                       for k in ("attractor", "saddle", "repeller", "marginal")]
                      + [Line2D([0], [0], color="deepskyblue", lw=2, label=r"$F_1=0$")],
              loc="upper right", fontsize=7.5, framealpha=0.9)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return gl1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out_root", default="results/figures")
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--xlim", type=float, default=2.0)
    ap.add_argument("--field_noise", action="store_true",
                    help="render the noise-averaged field E_x[Ψ] (K=16) instead of the clean field")
    ap.add_argument("--finder", choices=["brainpy", "scipy"], default="brainpy",
                    help="fixed-point finder: brainpy SlowPointFinder (default) or scipy.root grid")
    ap.add_argument("--slow_tol", type=float, default=1e-7,
                    help="max squared speed ‖F‖² kept as a (slow) fixed point; raise to reveal slow manifolds")
    ap.add_argument("--marg", type=float, default=0.04,
                    help="|Re eigenvalue| below this → 'marginal' (slow-manifold direction)")
    ap.add_argument("--density", type=float, default=1.6, help="streamplot line density")
    ap.add_argument("--stages", nargs="*", default=None,
                    choices=["dpa", "naive", "expert"],
                    help="render per-STAGE individual flows (fp_<stage>) into individual/<rid>/flow/ "
                         "instead of the single after-Dual bifurcation flow")
    args = ap.parse_args()

    sweep_name = os.path.basename(os.path.normpath(args.sweep_dir))
    rids = args.run_ids or discover_run_ids(args.sweep_dir)
    print(f"fixed-point finder: {args.finder}   stages: {args.stages or ['expert (bifurcation)']}")
    for rid in rids:
        try:
            if args.stages:                       # per-stage individual flows
                for st in args.stages:
                    out_path = os.path.join(args.out_root, sweep_name, "individual", rid,
                                            "flow", f"fp_{st}")
                    render_run(args.sweep_dir, rid, out_path, args.xlim, args.field_noise,
                               args.finder, args.slow_tol, args.marg, args.density, stage=st)
                print(f"  {rid}: fp_{{{','.join(args.stages)}}} -> individual/{rid}/flow/")
            else:                                 # single after-Dual bifurcation flow
                out_path = os.path.join(args.out_root, sweep_name, "bifurcation", rid)
                gl1 = render_run(args.sweep_dir, rid, out_path, args.xlim, args.field_noise,
                                 args.finder, args.slow_tol, args.marg, args.density)
                print(f"  {rid}: gλ1={gl1:.2f}  ->  {out_path}.pdf")
        except Exception as e:
            print(f"  {rid}: ERROR {e}")


if __name__ == "__main__":
    main()
