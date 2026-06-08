"""
plot_ring.py — visualise the ring of attractors that emerges after GNG training.

Collects autonomous attractor fixed points from all runs at each stage,
overlays a KDE to show the ring, and fits a circle to the attractor cloud.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from scipy.optimize import least_squares
import seaborn as sns

from src.models   import LowRankModel
from src.dynamics import find_all_fixed_points, classify_fixed_points

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

GOLDEN  = (5 ** 0.5 - 1) / 2
STAGES  = ["dpa", "naive", "expert"]
STAGE_TITLE = {"dpa": "After DPA", "naive": "After GNG", "expert": "After Dual"}
DEFAULTS = dict(hidden_size=512, rank=2, gain=2.0, input_size=8,
                tau=0.3, dt_base=0.03, tau_rec_frac=0.75)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_and_load(ckpt_path, device, gain=2.0):
    DT = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    model = LowRankModel(
        input_size=DEFAULTS["input_size"], hidden_size=DEFAULTS["hidden_size"],
        output_size=0, rank=DEFAULTS["rank"], gain=gain,
        alpha=DT / DEFAULTS["tau"],
        alpha_rec=DT / (DEFAULTS["tau"] * DEFAULTS["tau_rec_frac"]),
        noise=0.0, rwd=True, device=device,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_attractors(model, xlim, ylim, n_seeds):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    ff     = torch.zeros(DEFAULTS["input_size"], device=device, dtype=dtype)
    fps, _ = find_all_fixed_points(model, xlim=xlim, ylim=ylim, ff_input=ff,
                                   n_seeds=n_seeds, residual_tol=1e-8, merge_tol=5e-2)
    if len(fps) == 0:
        return np.empty((0, 2))
    labels, _ = classify_fixed_points(model, fps, ff_input=ff)
    return fps[labels == "attractor"]


def fit_circle(pts):
    """Algebraic least-squares circle fit. Returns (cx, cy, r)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2

    def residuals(p):
        cx, cy, r2 = p
        return (x - cx)**2 + (y - cy)**2 - r2

    # linear init
    p0_lin, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx0, cy0 = p0_lin[0], p0_lin[1]
    r0 = np.sqrt(np.mean((x - cx0)**2 + (y - cy0)**2))

    sol = least_squares(residuals, [cx0, cy0, r0**2], method="lm")
    cx, cy, r2 = sol.x
    return cx, cy, float(np.sqrt(abs(r2)))


def kde_ring(attractors, xlim, ylim, n_grid=300):
    if len(attractors) < 4:
        return None, None, None
    kde  = gaussian_kde(attractors.T, bw_method=0.12)
    k1   = np.linspace(xlim[0], xlim[1], n_grid)
    k2   = np.linspace(ylim[0], ylim[1], n_grid)
    K1, K2 = np.meshgrid(k1, k2)
    Z    = kde(np.vstack([K1.ravel(), K2.ravel()])).reshape(K1.shape)
    return K1, K2, Z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/home/leon/results/dual/sweep1")
    parser.add_argument("--results",  type=str, default=None)
    parser.add_argument("--out_dir",  type=str, default=None)
    parser.add_argument("--xlim",     type=float, nargs=2, default=[-1.6, 1.6])
    parser.add_argument("--ylim",     type=float, nargs=2, default=[-1.6, 1.6])
    parser.add_argument("--n_seeds",  type=int, default=21)
    parser.add_argument("--device",   type=str, default=None)
    args = parser.parse_args()

    device   = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.ckpt_dir
    out_dir  = args.out_dir or ckpt_dir
    results  = args.results or os.path.join(ckpt_dir, "results.jsonl")
    xlim, ylim = tuple(args.xlim), tuple(args.ylim)
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    with open(results) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                runs.append(r)
    print(f"Loaded {len(runs)} runs")

    # Collect attractor positions per stage, keeping memory_lambda per point
    # Each entry: dict(pts=array(K,2), memory_lambda=float, init_style=str)
    records = {s: [] for s in STAGES}

    for i, run in enumerate(runs):
        run_id = run["run_id"]
        style  = run["config"]["init_style"]
        ml     = run["config"].get("memory_lambda", float("nan"))
        gain   = run["config"].get("gain", 2.0)
        print(f"  [{i+1}/{len(runs)}] {run_id}", end="", flush=True)

        for stage in STAGES:
            ckpt = os.path.join(ckpt_dir, f"{stage}_{run_id}.pth")
            if not os.path.exists(ckpt):
                continue
            model = _build_and_load(ckpt, device, gain=gain)
            att   = get_attractors(model, xlim, ylim, args.n_seeds)
            if len(att):
                records[stage].append(dict(pts=att, memory_lambda=ml, init_style=style))
            print(f"  {stage}:{len(att)}", end="", flush=True)
        print()

    # Color map for memory_lambda
    ml_values  = sorted({0.6, 0.8, 0.95})
    ml_cmap    = plt.get_cmap("plasma")
    ml_norm    = plt.Normalize(vmin=min(ml_values), vmax=max(ml_values))
    def ml_color(v): return ml_cmap(ml_norm(v))

    # ------------------------------------------------------------------
    # Figure: 3 panels, one per stage
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    circle_rs = {}

    for ax, stage in zip(axes, STAGES):
        recs = records[stage]

        # All attractor points for circle fit
        all_pts = np.concatenate([r["pts"] for r in recs], axis=0) if recs else np.empty((0,2))

        # Fitted circle
        if len(all_pts) >= 4:
            cx, cy, r = fit_circle(all_pts)
            circle_rs[stage] = r
            theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                    color="0.4", lw=1.4, ls="--", zorder=3, label=f"fit r={r:.2f}")
            ax.scatter([cx], [cy], marker="+", s=60, color="0.4",
                       linewidths=1.5, zorder=4)

        # Scatter structured runs, colored by memory_lambda
        for rec in recs:
            pts   = rec["pts"]
            style = rec["init_style"]
            ml    = rec["memory_lambda"]

            if style == "structured":
                color = ml_color(ml)
                ec    = "black"
                alpha = 0.85
                s     = 35
                zo    = 6
            else:
                color = "0.75"
                ec    = "white"
                alpha = 0.6
                s     = 25
                zo    = 5

            ax.scatter(pts[:, 0], pts[:, 1],
                       s=s, color=color, edgecolors=ec,
                       linewidths=0.5, alpha=alpha, zorder=zo)

        ax.axhline(0, color="0.88", lw=0.5)
        ax.axvline(0, color="0.88", lw=0.5)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\kappa_1$")
        ax.set_title(STAGE_TITLE[stage])

    axes[0].set_ylabel(r"$\kappa_2$")

    # Colorbar for memory_lambda
    sm = plt.cm.ScalarMappable(cmap=ml_cmap, norm=ml_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label(r"memory $\lambda$ (structured)", fontsize=10)
    cbar.set_ticks(ml_values)

    # Print fitted radii
    for stage, r in circle_rs.items():
        print(f"  {STAGE_TITLE[stage]}: r = {r:.3f}")

    # Legend entry for random
    import matplotlib.lines as mlines
    rand_handle = mlines.Line2D([], [], marker="o", color="w", markerfacecolor="0.75",
                                markeredgecolor="white", markersize=7, label="random")
    axes[0].legend(handles=[rand_handle], frameon=False, fontsize=9, loc="lower left")

    fig.suptitle("Attractor ring — autonomous fixed points across all runs, coloured by memory λ")
    out_path = os.path.join(out_dir, "attractor_ring_memory_lambda.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
