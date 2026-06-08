"""
plot_fp_scatter.py — fixed-point scatter in κ-space.

Marker shape  = input condition (what stimulus is active)
Fill          = stability  (filled = attractor, open = saddle, ✕ = repeller)
Color         = input condition (redundant with shape for readability)

Outputs
-------
  fp_scatter_{run_id}.pdf   — 1 row × 3 cols (dpa / naive / expert)  per run
  fp_scatter_expert_all.pdf — all runs at the expert stage in one grid
"""

import json, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models   import LowRankModel
from src.dynamics import find_all_fixed_points, classify_fixed_points, make_input

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_DIR   = "/home/leon/results/dual/sweep_g5"
INPUT_SIZE = 8
N_FP_SEEDS = 41
XLIM = YLIM = (-2.5, 2.5)

STAGE_TASK = {"dpa": "dpa", "naive": "gng", "expert": "dual"}

# (label, active dims, color, marker)
INPUT_CONDITIONS = {
    "dpa": [
        ("Autonomous", None, "black",      "o"),
        ("Sample A",   [0],  "tab:blue",   "s"),
        ("Sample B",   [1],  "tab:orange", "D"),
        ("Test C",     [2],  "tab:green",  "^"),
        ("Test D",     [3],  "tab:purple", "v"),
    ],
    "gng": [
        ("Autonomous", None, "black",      "o"),
        ("Go",         [4],  "tab:red",    "P"),
        ("NoGo",       [5],  "tab:brown",  "h"),
        ("Cue",        [6],  "tab:pink",   "*"),
    ],
    "dual": [
        ("Autonomous", None, "black",      "o"),
        ("Sample A",   [0],  "tab:blue",   "s"),
        ("Sample B",   [1],  "tab:orange", "D"),
        ("Test C",     [2],  "tab:green",  "^"),
        ("Test D",     [3],  "tab:purple", "v"),
        ("Go",         [4],  "tab:red",    "P"),
        ("NoGo",       [5],  "tab:brown",  "h"),
        ("Cue",        [6],  "tab:pink",   "*"),
    ],
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_gain(run_id):
    for line in open(os.path.join(CKPT_DIR, "results.jsonl")):
        row = json.loads(line)
        if row["run_id"] == run_id:
            return float(row["config"]["gain"])
    raise ValueError(run_id)


def build_model(gain, device):
    tau = 0.3; dt = 0.03 * 0.75
    return LowRankModel(
        input_size=INPUT_SIZE, hidden_size=512, output_size=0,
        rank=2, gain=gain,
        alpha=dt / tau, alpha_rec=dt / (tau * 0.75),
        noise=0.0, rwd=True, device=device,
    )


def fps_for_stage(model, task, device):
    dtype = next(model.parameters()).dtype
    out = []
    for label, dims, color, marker in INPUT_CONDITIONS[task]:
        ff = make_input(INPUT_SIZE, active_dims=dims, value=1.0, device=device, dtype=dtype)
        fps, _ = find_all_fixed_points(
            model, xlim=XLIM, ylim=YLIM, ff_input=ff,
            n_seeds=N_FP_SEEDS, residual_tol=1e-8, merge_tol=5e-2,
        )
        labels, _ = classify_fixed_points(model, fps, ff_input=ff)
        out.append((label, color, marker, fps, labels))
    return out


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def draw_panel(ax, fp_data, title, acc_str=""):
    ax.axhline(0, color="lightgray", lw=0.7, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.7, zorder=0)

    for label, color, marker, fps, stabilities in fp_data:
        for fp, stab in zip(fps, stabilities):
            if stab == "attractor":
                fc, ec, sz, lw, zorder = color, color,   110, 1.2, 6
            elif stab == "saddle":
                fc, ec, sz, lw, zorder = "white", color,  90, 1.5, 5
            else:  # repeller / nonhyperbolic
                fc, ec, sz, lw, zorder = "white", color,  70, 1.0, 4

            ax.scatter(fp[0], fp[1],
                       marker=marker, s=sz,
                       facecolors=fc, edgecolors=ec,
                       linewidths=lw, zorder=zorder)

    title_full = title if not acc_str else f"{title}\n{acc_str}"
    ax.set_title(title_full, fontsize=8)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\kappa_1$", fontsize=8)


def make_legend_handles():
    cond_handles = [
        mpatches.Patch(color=color, label=label)
        for label, _, color, _ in INPUT_CONDITIONS["dual"]
    ]
    stab_handles = [
        plt.scatter([], [], marker="o", s=70, facecolors="gray",  edgecolors="gray",  label="attractor (filled)"),
        plt.scatter([], [], marker="o", s=70, facecolors="white", edgecolors="gray",  linewidths=1.5, label="saddle (open)"),
        plt.scatter([], [], marker="o", s=50, facecolors="white", edgecolors="gray",  linewidths=1.0, label="repeller (open, small)"),
    ]
    return cond_handles, stab_handles


# ---------------------------------------------------------------------------
# Per-run figure
# ---------------------------------------------------------------------------

def make_run_figure(run_id, model, device, acc_lookup):
    stages = ["dpa", "naive", "expert"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)

    for ax, stage in zip(axes, stages):
        ckpt = os.path.join(CKPT_DIR, "models", f"{stage}_{run_id}.pth")
        if not os.path.exists(ckpt):
            ax.set_visible(False)
            continue
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        task    = STAGE_TASK[stage]
        fp_data = fps_for_stage(model, task, device)
        acc     = acc_lookup.get(run_id, {}).get(stage, "")
        draw_panel(ax, fp_data, f"{stage} ({task.upper()})", acc)

    axes[0].set_ylabel(r"$\kappa_2$", fontsize=8)

    cond_h, stab_h = make_legend_handles()
    fig.legend(handles=cond_h + stab_h, loc="lower center",
               ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.06), frameon=False)

    # Add accuracy to suptitle
    a = acc_lookup.get(run_id, {})
    dual = a.get("expert", "")
    fig.suptitle(f"{run_id}   {dual}", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Summary grid (expert stage only)
# ---------------------------------------------------------------------------

def make_summary_figure(run_ids, models_data, device, acc_lookup, ncols=4):
    nrows = int(np.ceil(len(run_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.5 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for i, run_id in enumerate(run_ids):
        ax = axes_flat[i]
        gain = load_gain(run_id)
        model = build_model(gain, device)
        ckpt  = os.path.join(CKPT_DIR, "models", f"expert_{run_id}.pth")
        if not os.path.exists(ckpt):
            ax.set_visible(False)
            continue
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        fp_data = fps_for_stage(model, "dual", device)
        acc     = acc_lookup.get(run_id, {}).get("expert", "")
        draw_panel(ax, fp_data, run_id.replace("_random_g5", ""), acc)
        ax.set_ylabel(r"$\kappa_2$" if i % ncols == 0 else "", fontsize=7)

    for j in range(len(run_ids), len(axes_flat)):
        axes_flat[j].set_visible(False)

    cond_h, stab_h = make_legend_handles()
    fig.legend(handles=cond_h + stab_h, loc="lower center",
               ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.suptitle("Expert stage (after dual training) — all runs, gain=5.0", fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load all run results
    all_rows = [json.loads(l) for l in open(os.path.join(CKPT_DIR, "results.jsonl"))]
    run_ids  = [r["run_id"] for r in all_rows]

    # Build accuracy lookup: run_id -> stage -> short string
    acc_lookup = {}
    for r in all_rows:
        a  = r["accuracy"]
        au = a["after_dual"]
        acc_lookup[r["run_id"]] = {
            "dpa":    f"dpa={a['after_dpa']['dpa']:.2f}",
            "naive":  f"dpa={a['after_gng']['dpa']:.2f}  gng={a['after_gng']['gng']:.2f}",
            "expert": f"dual_dpa={au['dual_dpa']:.2f}  dual_gng={au['dual_gng']:.2f}",
        }

    # Sort by dual_dpa descending
    run_ids.sort(key=lambda rid: next(
        r["accuracy"]["after_dual"]["dual_dpa"]
        for r in all_rows if r["run_id"] == rid
    ), reverse=True)

    scatter_dir = os.path.join(CKPT_DIR, "figures", "scatters")
    os.makedirs(scatter_dir, exist_ok=True)

    # Per-run figures
    for run_id in run_ids:
        gain  = load_gain(run_id)
        model = build_model(gain, device)
        fig   = make_run_figure(run_id, model, device, acc_lookup)
        out   = os.path.join(scatter_dir, f"fp_scatter_{run_id}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # Summary grid
    fig = make_summary_figure(run_ids, None, device, acc_lookup, ncols=4)
    out = os.path.join(scatter_dir, "fp_scatter_expert_all.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
