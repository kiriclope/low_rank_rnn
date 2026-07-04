"""
plot.py — figures from results.jsonl.

Usage
-----
    python plot.py --results ../results/dual/vanilla/results.jsonl
    python plot.py --results path/to/results.jsonl --out_dir ./figures

Figures produced
----------------
    accuracy_stages.pdf   — DPA and GNG accuracy at each training stage
    dpa_retention.pdf     — DPA accuracy across stages, coloured by init_style
    loss_curves.pdf       — train/val loss curves per stage (requires loss_curves key)
"""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

from analyze import load_results

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

GOLDEN = (5 ** 0.5 - 1) / 2
W = 6

STAGE_LABELS  = ["After DPA", "After GNG", "After Dual"]
STAGE_KEYS    = ["after_dpa", "after_gng", "after_dual"]
STAGE_X       = np.arange(len(STAGE_LABELS))

STYLE_COLORS  = {"structured": "#4C72B0", "random": "#DD8452"}
STAGE_COLORS  = {"dpa": "#4C72B0", "gng": "#DD8452"}


# ---------------------------------------------------------------------------
# Figure 1 — accuracy at each stage, two panels (DPA task / GNG task)
# ---------------------------------------------------------------------------

def plot_accuracy_stages(df, group_by="init_style", out_path=None):
    """
    Dots + mean ± SEM for DPA accuracy and GNG accuracy across the three
    training stages, grouped by `group_by`.
    """
    groups  = sorted(df[group_by].unique()) if group_by in df.columns else ["all"]
    n_groups= len(groups)
    offsets = np.linspace(-0.15 * (n_groups - 1) / 2,
                           0.15 * (n_groups - 1) / 2, n_groups)

    cmap   = plt.get_cmap("tab10")
    colors = {g: STYLE_COLORS.get(g, cmap(i)) for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, 2, figsize=(2.2 * W, W * GOLDEN), sharey=True)

    # At the DUAL checkpoint, use the DUAL-TASK metrics (dual_dpa / dual_gng) — NOT the
    # standalone-task scores (after_dual/gng is a dual-specialised net scored on the
    # isolated GNG task ≈ chance, which does NOT reflect dual go/nogo and mismatches the
    # accuracy-by-trialtype figure). Fall back to the standalone column if dual_* is absent.
    def _stage_cols(metric):
        cols = []
        for s in STAGE_KEYS:
            dual = f"{s}/dual_{metric}"
            cols.append(dual if dual in df.columns else f"{s}/{metric}")
        return cols

    task_meta = [
        ("DPA accuracy",  _stage_cols("dpa")),
        ("GNG accuracy",  _stage_cols("gng")),
    ]

    rng = np.random.default_rng(0)

    for ax, (title, col_keys) in zip(axes, task_meta):
        for i, g in enumerate(groups):
            sub = df[df[group_by] == g] if group_by in df.columns else df
            x   = STAGE_X + offsets[i]
            c   = colors[g]

            means, sems = [], []
            for col in col_keys:
                if col not in sub.columns:
                    means.append(np.nan); sems.append(0.0); continue
                vals = sub[col].dropna().values
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

            means, sems = np.array(means), np.array(sems)

            # individual dots
            for j, col in enumerate(col_keys):
                if col not in sub.columns: continue
                vals   = sub[col].dropna().values
                jitter = rng.normal(0, 0.03, len(vals))
                ax.scatter(np.full(len(vals), x[j]) + jitter, vals,
                           s=22, color=c, alpha=0.3, linewidths=0, zorder=2)

            # mean line
            ax.plot(x, means, color=c, lw=2.0, alpha=0.85, zorder=3)

            # mean ± SEM
            ax.errorbar(x, means, yerr=sems, fmt="o", ms=9,
                        mfc=c, mec="white", mew=1.6,
                        ecolor=c, elinewidth=2.0, capsize=5, capthick=2.0,
                        zorder=4, label=str(g))

        ax.axhline(0.5, color="0.6", ls="--", lw=1.2, zorder=1)
        ax.set_title(title, pad=10)
        ax.set_xticks(STAGE_X)
        ax.set_xticklabels(STAGE_LABELS)
        ax.set_xlim(-0.5, len(STAGE_LABELS) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Checkpoint")
        ax.grid(axis="y", color="0.92", lw=1.0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Accuracy")
    axes[1].legend(title=group_by, frameon=False, fontsize=9)
    sns.despine(fig=fig, trim=True)
    fig.tight_layout(w_pad=2.0)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — DPA retention: trajectory across stages per run
# ---------------------------------------------------------------------------

def plot_dpa_retention(df, group_by="init_style", out_path=None):
    """
    Each line is one run; colour = init_style.
    Shows how DPA accuracy evolves across the three stages.
    """
    groups = sorted(df[group_by].unique()) if group_by in df.columns else ["all"]
    cmap   = plt.get_cmap("tab10")
    colors = {g: STYLE_COLORS.get(g, cmap(i)) for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(W, W * GOLDEN))

    dpa_cols = [f"{s}/dpa" for s in STAGE_KEYS]

    for _, row in df.iterrows():
        vals = [row.get(c, np.nan) for c in dpa_cols]
        if all(np.isnan(v) for v in vals): continue
        g     = row.get(group_by, "all")
        color = colors.get(g, "gray")
        ax.plot(STAGE_X, vals, color=color, lw=1.2, alpha=0.35, zorder=2)

    # group means
    for g in groups:
        sub  = df[df[group_by] == g] if group_by in df.columns else df
        means= [sub[c].mean() if c in sub.columns else np.nan for c in dpa_cols]
        ax.plot(STAGE_X, means, color=colors[g], lw=2.8, alpha=0.95,
                label=f"{g} (mean)", zorder=4)
        ax.scatter(STAGE_X, means, color=colors[g], s=60, zorder=5,
                   edgecolors="white", linewidths=1.2)

    ax.axhline(0.5, color="0.6", ls="--", lw=1.2)
    ax.set_xticks(STAGE_X)
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("DPA accuracy")
    ax.set_title("DPA retention across training stages")
    ax.legend(title=group_by, frameon=False, fontsize=9)
    ax.grid(axis="y", color="0.92", lw=1.0)
    ax.set_axisbelow(True)
    sns.despine(fig=fig, trim=True)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — loss curves per stage
# ---------------------------------------------------------------------------

def plot_loss_curves(df_raw, group_by="init_style", out_path=None):
    """
    Mean train/val loss curves for each stage, one panel per stage.
    Requires loss_curves key in results.jsonl (added in sweep.py).
    """
    import json

    # Re-read raw JSON to get loss curves (not stored in flat DataFrame)
    groups  = sorted(df_raw[group_by].unique()) if group_by in df_raw.columns else ["all"]
    cmap    = plt.get_cmap("tab10")
    colors  = {g: STYLE_COLORS.get(g, cmap(i)) for i, g in enumerate(groups)}
    stages  = ["dpa", "gng", "dual"]

    fig, axes = plt.subplots(1, 3, figsize=(3 * W * 0.85, W * GOLDEN),
                             sharey=False, constrained_layout=True)

    for ax, stage in zip(axes, stages):
        for g in groups:
            sub_ids = set(df_raw[df_raw[group_by] == g]["run_id"]) if group_by in df_raw.columns else set(df_raw["run_id"])
            all_train, all_val = [], []

            for run_id in sub_ids:
                curves = _loss_curves_cache.get(run_id, {})
                if stage not in curves: continue
                all_train.append(curves[stage]["train"])
                all_val.append(curves[stage]["val"])

            if not all_train: continue

            # Pad to same length then average
            max_len  = max(len(x) for x in all_train)
            def pad(lst): return lst + [lst[-1]] * (max_len - len(lst))
            t_arr = np.array([pad(x) for x in all_train])
            v_arr = np.array([pad(x) for x in all_val])

            ep    = np.arange(1, max_len + 1)
            c     = colors[g]
            ax.plot(ep, t_arr.mean(0), color=c,   lw=2.0, label=f"{g} train")
            ax.plot(ep, v_arr.mean(0), color=c,   lw=2.0, ls="--", label=f"{g} val")
            ax.fill_between(ep,
                            t_arr.mean(0) - t_arr.std(0),
                            t_arr.mean(0) + t_arr.std(0),
                            color=c, alpha=0.12)

        ax.set_title(f"Stage: {stage.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(color="0.92", lw=1.0)
        ax.set_axisbelow(True)

    # shared legend on last axis
    handles = []
    for g in groups:
        c = colors[g]
        handles += [
            mlines.Line2D([], [], color=c, lw=2.0, label=f"{g} train"),
            mlines.Line2D([], [], color=c, lw=2.0, ls="--", label=f"{g} val"),
        ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8)
    sns.despine(fig=fig)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Load loss curves from jsonl (separate from flat DataFrame)
# ---------------------------------------------------------------------------

_loss_curves_cache: dict = {}

def _load_loss_curves(path):
    import json
    global _loss_curves_cache
    _loss_curves_cache = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("status") == "ok" and "loss_curves" in r:
                _loss_curves_cache[r["run_id"]] = r["loss_curves"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  type=str, default="../results/dual/vanilla/results.jsonl")
    parser.add_argument("--out_dir",  type=str, default=None)
    parser.add_argument("--group_by", type=str, default="init_style")
    parser.add_argument("--show",     action="store_true", default=False)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    df = load_results(args.results)
    _load_loss_curves(args.results)
    print(f"Loaded {len(df)} runs  |  group_by={args.group_by}")

    plot_accuracy_stages(df, group_by=args.group_by,
                         out_path=os.path.join(out_dir, "accuracy_stages.pdf"))

    plot_dpa_retention(df, group_by=args.group_by,
                       out_path=os.path.join(out_dir, "dpa_retention.pdf"))

    if _loss_curves_cache:
        plot_loss_curves(df, group_by=args.group_by,
                         out_path=os.path.join(out_dir, "loss_curves.pdf"))
    else:
        print("No loss_curves in results.jsonl — skipping loss curve figure.")
        print("(Re-run sweep with current sweep.py to get them.)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
