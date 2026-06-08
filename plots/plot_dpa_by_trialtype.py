"""
plot_dpa_by_trialtype.py — DPA accuracy broken down by trial type (DPA-only / Go / NoGo)
evaluated post-hoc from saved checkpoints.

Each checkpoint (dpa, naive, expert) is evaluated on dual trials and DPA
accuracy is computed separately for:
  - DPA-only  (no GNG stimulus present)
  - Go        (GNG=go + DPA)
  - NoGo      (GNG=nogo + DPA)

Usage
-----
    python plot_dpa_by_trialtype.py --ckpt_dir /home/leon/results/dual/sweep_random
    python plot_dpa_by_trialtype.py --ckpt_dir /home/leon/results/dual/sweep1
    python plot_dpa_by_trialtype.py \
        --ckpt_dir /home/leon/results/dual/sweep1 \
                   /home/leon/results/dual/sweep_random \
        --labels sweep1 sweep_random
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
import seaborn as sns

from src.tasks  import TaskTiming, generate_dual_trials
from src.models import LowRankModel

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

GOLDEN = (5 ** 0.5 - 1) / 2
W      = 5

STAGES      = ["dpa", "naive", "expert"]
STAGE_LABEL = {"dpa": "After DPA", "naive": "After GNG", "expert": "After Dual"}
STAGE_X     = np.arange(len(STAGES))

TRIAL_TYPES  = ["DPA-only", "Go", "NoGo"]
TRIAL_COLORS = {"DPA-only": "#4C72B0", "Go": "#55A868", "NoGo": "#C44E52"}
TRIAL_MARKER = {"DPA-only": "o", "Go": "s", "NoGo": "^"}

DEFAULTS = dict(hidden_size=512, rank=2, input_size=8,
                tau=0.3, dt_base=0.03, tau_rec_frac=0.75)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_and_load(ckpt_path, device, gain=2.0):
    DT = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    model = LowRankModel(
        input_size=DEFAULTS["input_size"], hidden_size=DEFAULTS["hidden_size"],
        output_size=0, rank=2, gain=gain,
        alpha=DT / DEFAULTS["tau"],
        alpha_rec=DT / (DEFAULTS["tau"] * DEFAULTS["tau_rec_frac"]),
        noise=0.0, rwd=True, device=device,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def dpa_accuracy_by_trialtype(model, dual_timing, device, n_trials=2048, noise=0.1):
    """Returns dict: {DPA-only: float, Go: float, NoGo: float}"""
    X, y, _, cond_names = generate_dual_trials(
        n_trials, timing=dual_timing, noise=noise
    )
    X, y = X.to(device), y.to(device)
    pred = model(X, y)[..., -1].cpu()

    names     = np.asarray(cond_names).astype(str)
    is_dpaonly= torch.as_tensor(["_go_" not in n and "_nogo_" not in n for n in names])
    is_go     = torch.as_tensor(["_go_"   in n for n in names])
    is_nogo   = torch.as_tensor(["_nogo_" in n for n in names])

    dpa_start = int(dual_timing.n_stim_off[3])
    pred_dpa  = pred[:, dpa_start:].mean(1)
    label_dpa = (y[:, -1, -1] > 0).cpu()

    correct = (pred_dpa > 0) == label_dpa

    result = {}
    for mask, name in [(is_dpaonly, "DPA-only"), (is_go, "Go"), (is_nogo, "NoGo")]:
        result[name] = correct[mask].float().mean().item() if mask.any() else float("nan")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, nargs="+",
                        default=["/home/leon/results/dual/sweep_random"])
    parser.add_argument("--labels",   type=str, nargs="+", default=None)
    parser.add_argument("--out_dir",  type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=2048)
    parser.add_argument("--device",   type=str, default=None)
    args = parser.parse_args()

    device  = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    labels  = args.labels or [os.path.basename(d) for d in args.ckpt_dir]
    out_dir = args.out_dir or args.ckpt_dir[0]
    os.makedirs(out_dir, exist_ok=True)

    DT          = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    dual_timing = TaskTiming([2.0, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0], 10.0, DT)
    noise       = float(0.5 * torch.sqrt(1.0 - torch.exp(torch.tensor(-DT / DEFAULTS["tau"])) ** 2))

    # Collect per-run accuracy per stage per trial type
    # all_data[sweep_label][stage][trial_type] = list of accuracy values
    all_data = {}

    for ckpt_dir, sweep_label in zip(args.ckpt_dir, labels):
        results_path = os.path.join(ckpt_dir, "results.jsonl")
        with open(results_path) as f:
            runs = [json.loads(l) for l in f if l.strip()]
        runs = [r for r in runs if r.get("status") == "ok"]
        print(f"\n{sweep_label}: {len(runs)} runs")

        data = {stage: {tt: [] for tt in TRIAL_TYPES} for stage in STAGES}

        for i, run in enumerate(runs):
            run_id = run["run_id"]
            gain   = run["config"].get("gain", 2.0)
            print(f"  [{i+1}/{len(runs)}] {run_id}", end="", flush=True)

            for stage in STAGES:
                ckpt = os.path.join(ckpt_dir, "models", f"{stage}_{run_id}.pth")
                if not os.path.exists(ckpt):
                    print(f"  (missing {stage})", end="")
                    continue
                model  = _build_and_load(ckpt, device, gain=gain)
                result = dpa_accuracy_by_trialtype(model, dual_timing, device,
                                                   n_trials=args.n_trials, noise=noise)
                for tt in TRIAL_TYPES:
                    data[stage][tt].append(result[tt])
                print(f"  {stage}:ok", end="", flush=True)
            print()

        all_data[sweep_label] = data

    # ------------------------------------------------------------------
    # Plot: one panel per sweep, x=stage, lines=trial type
    # ------------------------------------------------------------------
    n_sweeps = len(labels)
    fig, axes = plt.subplots(1, n_sweeps,
                             figsize=(W * n_sweeps, W * GOLDEN),
                             sharey=True, constrained_layout=True)
    if n_sweeps == 1:
        axes = [axes]

    rng = np.random.default_rng(0)

    for ax, sweep_label in zip(axes, labels):
        data = all_data[sweep_label]

        for tt in TRIAL_TYPES:
            c      = TRIAL_COLORS[tt]
            marker = TRIAL_MARKER[tt]
            means, sems = [], []

            for stage in STAGES:
                vals = [v for v in data[stage][tt] if not np.isnan(v)]
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                            if len(vals) > 1 else 0.0)

                # individual dots with jitter
                jitter = rng.normal(0, 0.04, len(vals))
                ax.scatter(np.full(len(vals), STAGE_X[STAGES.index(stage)]) + jitter,
                           vals, s=18, color=c, alpha=0.25, linewidths=0, zorder=2)

            means, sems = np.array(means), np.array(sems)
            ax.plot(STAGE_X, means, color=c, lw=2.2, zorder=4, label=tt)
            ax.errorbar(STAGE_X, means, yerr=sems,
                        fmt=marker, ms=9, mfc=c, mec="white", mew=1.5,
                        ecolor=c, elinewidth=2.0, capsize=4, capthick=1.8,
                        zorder=5)

        ax.axhline(0.5, color="0.6", ls="--", lw=1.2)
        ax.set_xticks(STAGE_X)
        ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
        ax.set_xlim(-0.4, len(STAGES) - 0.6)
        ax.set_ylim(0.4, 1.05)
        ax.set_xlabel("Checkpoint")
        ax.set_title(sweep_label)
        ax.grid(axis="y", color="0.92", lw=1.0)
        ax.set_axisbelow(True)
        ax.legend(title="Trial type", frameon=False, fontsize=9)

    axes[0].set_ylabel("DPA accuracy")
    sns.despine(fig=fig, trim=True)

    out_path = os.path.join(out_dir, "dpa_by_trialtype.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
