"""
plot_trajectories.py — κ vs time for each training stage (dual trials).

For each of {dpa, naive, expert} checkpoints, runs dual trials (target_rank=2)
and produces 3 figures per stage following plot_dual_kappa_targets from lr_dual.org:

    traj_{stage}_dpa_only.pdf  — DPA-only trials
    traj_{stage}_go.pdf        — Go trials
    traj_{stage}_nogo.pdf      — NoGo trials

Each figure: 2 rows (paired / unpaired) × 2 cols (κ₁ / κ₂).
κ = model readout (rates @ n / N), shape (trials, time, 2).

Usage
-----
    python plot_trajectories.py \
        --run_id test_cue_go \
        --ckpt_dir results/test_go_cue/simulations \
        --out_dir  results/test_go_cue/figures \
        --gain 2.0 --cue_on_go_input
"""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from src.tasks  import TaskTiming, generate_dual_trials
from src.models import LowRankModel

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)


DEFAULTS = dict(
    hidden_size  = 512,
    rank         = 2,
    gain         = 2.0,
    input_size   = 8,
    tau          = 0.3,
    dt_base      = 0.03,
    tau_rec_frac = 0.75,
)

CONDITION_COLORS = ("tab:blue", "tab:orange")  # sample A, sample B

GNG_SPECS = {
    "dpa_only": "none",
    "go":       "go",
    "nogo":     "nogo",
}


def build_model(device, gain=None, input_size=8):
    DT = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    alpha     = DT / DEFAULTS["tau"]
    alpha_rec = DT / (DEFAULTS["tau"] * DEFAULTS["tau_rec_frac"])
    return LowRankModel(
        input_size  = input_size,
        hidden_size = DEFAULTS["hidden_size"],
        output_size = 0,
        rank        = DEFAULTS["rank"],
        gain        = gain if gain is not None else DEFAULTS["gain"],
        alpha       = alpha,
        alpha_rec   = alpha_rec,
        noise       = 0.0,
        rwd         = True,
        device      = device,
    )


def make_dual_timing():
    DT = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    return TaskTiming([2.0, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0], 10.0, DT)


def _load_run_config(ckpt_dir: str, run_id: str) -> dict | None:
    """
    Return the saved config dict for run_id from <ckpt_dir>/results.jsonl, or None.

    Note: config["input_size"] is already post-RunConfig.__post_init__ (decremented
    when cue_on_go_input is True), so use it as-is — do NOT subtract again.
    """
    import json
    jsonl = os.path.join(ckpt_dir, "results.jsonl")
    if not os.path.exists(jsonl):
        return None
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("run_id") == run_id:
            return row.get("config", {})
    return None


def get_readout(model, inputs, targets):
    """Return readout (= κ, shape trials×time×rank)."""
    inputs  = inputs.to(model.device)
    targets = targets.to(model.device)
    with torch.no_grad():
        readout, _, _ = model(inputs, targets=targets, ret_rates=True)
    return readout.detach().cpu().numpy()


def concrete_conditions(pair_status: str, gng_status: str) -> list[str]:
    """Return the 2 condition name strings for the given pair/gng combination."""
    if pair_status == "pair":
        base = [("A", "C"), ("B", "D")]
    else:
        base = [("A", "D"), ("B", "C")]
    if gng_status == "none":
        return [f"{s}_{t}" for s, t in base]
    return [f"{s}_{gng_status}_{t}" for s, t in base]


def plot_gng_condition(kappa_np, targets_np, condition_names, timing,
                       gng_label: str, gng_status: str) -> plt.Figure:
    """
    2×2 figure for one GNG condition:
        rows = pair | unpair
        cols = κ₁  | κ₂
    Colors: tab:blue = sample A,  tab:orange = sample B.
    Black line = target (mean over matching trials).
    """
    n_dims = kappa_np.shape[-1]
    cnames = np.asarray(condition_names).astype(str)
    t = np.arange(kappa_np.shape[1]) * timing.dt

    width, height = 5.0, 3.1
    fig, axes = plt.subplots(2, n_dims, figsize=(n_dims * width, 2 * height),
                              sharex=True, sharey="col", constrained_layout=True)
    fig.suptitle(f"{gng_label} trials", fontsize=13)

    for row, (row_title, pair_status) in enumerate([("Paired", "pair"), ("Unpaired", "unpair")]):
        conds = concrete_conditions(pair_status, gng_status)

        for col in range(n_dims):
            ax = axes[row, col]

            # y-limits from this dim's full data
            k_lim = np.nanpercentile(np.abs(kappa_np[:, :, col]), 99)
            y_lim = 1.1 * max(float(k_lim), 1.5, 1e-6)
            ax.set_ylim(-y_lim, y_lim)

            # stim shading
            for on, off in zip(timing.stim_on, timing.stim_off):
                ax.axvspan(on, off, alpha=0.12, color="tab:blue", lw=0, zorder=0)

            for cond_i, cond_name in enumerate(conds):
                idxs = np.where(cnames == cond_name)[0]
                if len(idxs) == 0:
                    ax.text(0.5, 0.5 - 0.12 * cond_i, f"missing {cond_name}",
                            ha="center", va="center", transform=ax.transAxes,
                            color="crimson", fontsize=8)
                    continue

                color = CONDITION_COLORS[cond_i]
                k_mean = kappa_np[idxs, :, col].mean(axis=0)

                # target: use same dim when in range, else last channel
                tgt_dim = col if col < targets_np.shape[-1] else -1
                tgt_mean = np.nanmean(targets_np[idxs, :, tgt_dim], axis=0)

                ax.plot(t, tgt_mean, color="k", lw=2.4, alpha=0.85, zorder=9)
                ax.plot(t, k_mean,   color=color, lw=3.0, alpha=0.9, zorder=10)

            if col == 0:
                ax.set_ylabel(f"κ{col + 1}\n{row_title}", fontsize=10)
            else:
                ax.set_ylabel(f"κ{col + 1}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Time (s)")
            if row == 0:
                ax.set_title(f"κ{col + 1}")

    legend_handles = [
        mlines.Line2D([], [], color="k",                lw=2.4, label="Target"),
        mlines.Line2D([], [], color=CONDITION_COLORS[0], lw=3.0, label="A"),
        mlines.Line2D([], [], color=CONDITION_COLORS[1], lw=3.0, label="B"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.06))

    return fig


def run_stage(ckpt_path, device, timing, inputs_t, targets_t, condition_names,
              gain, input_size):
    model = build_model(device, gain=gain, input_size=input_size)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    kappa = get_readout(model, inputs_t, targets_t)
    return kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",          type=str,   default="test_cue_go")
    parser.add_argument("--ckpt_dir",        type=str,   default="results/test_go_cue/simulations")
    parser.add_argument("--out_dir",         type=str,   default=None)
    parser.add_argument("--stage",           type=str,   default=None,
                        help="One of: dpa, naive, expert (default: all)")
    parser.add_argument("--n_batch",         type=int,   default=516)
    parser.add_argument("--gain",            type=float, default=None)
    parser.add_argument("--noise",           type=float, default=0.5,
                        help="Noise prefactor used during training (default: 0.5)")
    parser.add_argument("--input_size",      type=int,   default=None)
    parser.add_argument("--cue_on_go_input", action="store_true")
    parser.add_argument("--device",          type=str,   default=None)
    parser.add_argument("--show",            action="store_true")
    args = parser.parse_args()

    device  = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Pull gain / input_size / cue_on_go_input from the saved config when available;
    # explicit CLI flags override.  config["input_size"] is already resized.
    cfg = _load_run_config(args.ckpt_dir, args.run_id)
    if cfg is not None:
        print(f"Read config for {args.run_id} from results.jsonl")

    gain = args.gain
    if gain is None and cfg is not None:
        gain = float(cfg["gain"])

    if args.cue_on_go_input:
        cue_on_go_input = True
    elif cfg is not None:
        cue_on_go_input = bool(cfg.get("cue_on_go_input", False))
    else:
        cue_on_go_input = False

    if args.input_size is not None:
        input_size = args.input_size                 # explicit override, used as-is
    elif cfg is not None:
        input_size = int(cfg["input_size"])          # already post-__post_init__
    else:
        input_size = DEFAULTS["input_size"]
        if cue_on_go_input:
            input_size -= 1

    timing = make_dual_timing()

    # generate one shared batch of dual trials (target_rank=2)
    inputs, targets, _, condition_names = generate_dual_trials(
        args.n_batch, timing,
        input_size      = input_size,
        target_rank     = 2,
        noise           = args.noise,
        cue_on_go_input = cue_on_go_input,
    )
    inputs_t  = torch.as_tensor(inputs,  dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    targets_np = targets_t.numpy()

    stages = [args.stage] if args.stage else ["dpa", "naive", "expert"]

    for stage in stages:
        ckpt = os.path.join(args.ckpt_dir, "models", f"{stage}_{args.run_id}.pth")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found, skipping: {ckpt}")
            continue

        print(f"[{stage}]  input_size={input_size}  gain={gain}")
        kappa = run_stage(ckpt, device, timing, inputs_t, targets_t,
                          condition_names, gain, input_size)

        for fig_name, gng_status in GNG_SPECS.items():
            gng_label = {"dpa_only": "DPA-only", "go": "Go", "nogo": "NoGo"}[fig_name]
            fig = plot_gng_condition(kappa, targets_np, condition_names, timing,
                                     gng_label, gng_status)
            out_path = os.path.join(out_dir, f"traj_{stage}_{fig_name}.pdf")
            fig.savefig(out_path, bbox_inches="tight")
            print(f"  Saved {out_path}")
            plt.close(fig)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
