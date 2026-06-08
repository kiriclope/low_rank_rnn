"""
plot_all_fixed_points.py — overlay fixed points from all runs on one κ-plane.

For each run × stage, loads the checkpoint, finds autonomous fixed points,
and scatters them all on the same axes.  One panel per stage (dpa / naive /
expert), coloured by fixed-point type, marker shape by init_style.

Usage
-----
    python plot_all_fixed_points.py
    python plot_all_fixed_points.py --ckpt_dir /home/leon/results/dual/sweep1 \
                                    --out_dir  /home/leon/results/dual/sweep1
    python plot_all_fixed_points.py --input A   # use a specific frozen input
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from src.models   import LowRankModel
from src.dynamics import (
    low_rank_numpy_params,
    find_all_fixed_points,
    classify_fixed_points,
)

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

GOLDEN = (5 ** 0.5 - 1) / 2

# Fixed-point visual encoding
FP_STYLE = {
    "attractor":      dict(marker="o", s=40,  linewidths=0.5),
    "saddle":         dict(marker="X", s=40,  linewidths=0.5),
    "repeller":       dict(marker="^", s=35,  linewidths=0.5),
    "nonhyperbolic":  dict(marker="D", s=30,  linewidths=0.5),
}
FP_COLORS = {
    "attractor":     "#2196F3",
    "saddle":        "#FF9800",
    "repeller":      "#F44336",
    "nonhyperbolic": "#9C27B0",
}
INIT_EDGE = {
    "structured": "black",
    "random":     "white",
}
INIT_ALPHA = {
    "structured": 0.55,
    "random":     0.55,
}

STAGES    = ["dpa", "naive", "expert"]
STAGE_TITLE = {"dpa": "After DPA", "naive": "After GNG", "expert": "After Dual"}

# Input presets (index into the 8-dim input vector)
INPUT_PRESETS = {
    "none": None,
    "A":    [0],
    "B":    [1],
    "C":    [2],
    "D":    [3],
    "Go":   [4],
    "NoGo": [5],
    "Cue":  [6],
}


# ---------------------------------------------------------------------------
# Model builder (matches sweep.py defaults)
# ---------------------------------------------------------------------------

DEFAULTS = dict(hidden_size=512, rank=2, gain=2.0, input_size=8,
                tau=0.3, dt_base=0.03, tau_rec_frac=0.75)


def _load_model(ckpt_path: str, device: str, gain: float = 2.0) -> LowRankModel:
    DT = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    model = LowRankModel(
        input_size  = DEFAULTS["input_size"],
        hidden_size = DEFAULTS["hidden_size"],
        output_size = 0,
        rank        = DEFAULTS["rank"],
        gain        = gain,
        alpha       = DT / DEFAULTS["tau"],
        alpha_rec   = DT / (DEFAULTS["tau"] * DEFAULTS["tau_rec_frac"]),
        noise       = 0.0,
        rwd         = True,
        device      = device,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Fixed-point extraction for one checkpoint
# ---------------------------------------------------------------------------

def get_fixed_points(model, ff_input_dims, xlim, ylim, n_seeds):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    if ff_input_dims is None:
        ff_input = torch.zeros(DEFAULTS["input_size"], device=device, dtype=dtype)
    else:
        ff_input = torch.zeros(DEFAULTS["input_size"], device=device, dtype=dtype)
        ff_input[ff_input_dims] = 1.0

    fps, _ = find_all_fixed_points(
        model, xlim=xlim, ylim=ylim, ff_input=ff_input,
        n_seeds=n_seeds, residual_tol=1e-8, merge_tol=5e-2,
    )
    labels, _ = classify_fixed_points(model, fps, ff_input=ff_input)
    return fps, labels          # arrays of shape (K,2) and (K,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/home/leon/results/dual/sweep1")
    parser.add_argument("--results",  type=str, default=None,
                        help="Path to results.jsonl (default: ckpt_dir/results.jsonl)")
    parser.add_argument("--out_dir",  type=str, default=None)
    parser.add_argument("--input",    type=str, default="none",
                        choices=list(INPUT_PRESETS.keys()),
                        help="Input condition for fixed-point search (default: none = autonomous)")
    parser.add_argument("--xlim",     type=float, nargs=2, default=[-1.5, 1.5])
    parser.add_argument("--ylim",     type=float, nargs=2, default=[-1.5, 1.5])
    parser.add_argument("--n_seeds",  type=int, default=21,
                        help="Grid seeds per axis for root-finding (default: 21)")
    parser.add_argument("--device",   type=str, default=None)
    parser.add_argument("--stages",   type=str, nargs="+", default=STAGES,
                        choices=STAGES)
    args = parser.parse_args()

    device   = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.ckpt_dir
    out_dir  = args.out_dir or ckpt_dir
    results  = args.results or os.path.join(ckpt_dir, "results.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    ff_dims  = INPUT_PRESETS[args.input]
    xlim     = tuple(args.xlim)
    ylim     = tuple(args.ylim)

    # Load run metadata
    runs = []
    with open(results) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                runs.append(r)
    print(f"Found {len(runs)} runs")

    # Collect fixed points per stage
    # data[stage] = list of dicts {fp:(K,2), labels:(K,), init_style, run_id}
    data = {s: [] for s in args.stages}

    for i, run in enumerate(runs):
        run_id     = run["run_id"]
        init_style = run["config"]["init_style"]
        gain       = run["config"].get("gain", 2.0)
        print(f"  [{i+1}/{len(runs)}] {run_id}", end="", flush=True)

        for stage in args.stages:
            ckpt = os.path.join(ckpt_dir, "models", f"{stage}_{run_id}.pth")
            if not os.path.exists(ckpt):
                print(f"  (missing {stage})", end="")
                continue

            model = _load_model(ckpt, device, gain=gain)
            fps, labels = get_fixed_points(model, ff_dims, xlim, ylim, args.n_seeds)

            data[stage].append(dict(
                fps=fps, labels=labels,
                init_style=init_style, run_id=run_id,
            ))
            print(f"  {stage}:{len(fps)}fps", end="", flush=True)

        print()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n_panels = len(args.stages)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(4.5 * n_panels, 4.5),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for ax, stage in zip(axes, args.stages):
        for entry in data[stage]:
            fps    = entry["fps"]
            labels = entry["labels"]
            style  = entry["init_style"]

            if len(fps) == 0:
                continue

            for fp_type in ["attractor", "saddle", "repeller", "nonhyperbolic"]:
                mask = labels == fp_type
                if not np.any(mask):
                    continue
                pts = fps[mask]
                kw  = FP_STYLE[fp_type].copy()
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    color        = FP_COLORS[fp_type],
                    edgecolors   = INIT_EDGE[style],
                    alpha        = INIT_ALPHA[style],
                    zorder       = 5,
                    **kw,
                )

        ax.axhline(0, color="0.8", lw=0.6, zorder=1)
        ax.axvline(0, color="0.8", lw=0.6, zorder=1)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\kappa_1$")
        ax.set_title(STAGE_TITLE[stage])

    axes[0].set_ylabel(r"$\kappa_2$")

    # Legend — fixed-point type
    fp_handles = [
        mlines.Line2D([], [], marker=FP_STYLE[t]["marker"], color="w",
                      markerfacecolor=FP_COLORS[t], markeredgecolor="gray",
                      markersize=8, label=t)
        for t in ["attractor", "saddle", "repeller"]
    ]
    # init_style edge
    style_handles = [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor="gray", markeredgecolor=INIT_EDGE[s],
                      markersize=8, label=s)
        for s in ["structured", "random"]
    ]
    axes[-1].legend(handles=fp_handles + style_handles,
                    frameon=False, fontsize=8, loc="upper right")

    input_label = f"input={args.input}" if args.input != "none" else "autonomous"
    fig.suptitle(f"Fixed points across all runs — {input_label}  (n={len(runs)})")

    out_path = os.path.join(out_dir, f"all_fp_{args.input}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
