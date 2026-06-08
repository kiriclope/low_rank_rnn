"""
plot_trials.py — visualise inputs and targets for GNG and dual tasks.
One row per representative trial type, columns = time.
Top panel: inputs (one line per active channel). Bottom panel: targets (one line per rank).
"""
import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tasks import TaskTiming, generate_gng_trials, generate_dual_trials

sns.set_context("notebook"); sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

DT   = 0.03 * 0.75
OUT  = "/home/leon/rnn/results/trial_targets"
os.makedirs(OUT, exist_ok=True)

# ── colour maps ──────────────────────────────────────────────────────────────
INPUT_COLORS = {
    0: ("tab:blue",   "A sample"),
    1: ("tab:orange", "B sample"),
    2: ("tab:green",  "C test"),
    3: ("tab:purple", "D test"),
    4: ("tab:red",    "Go / Cue"),
    5: ("tab:brown",  "NoGo"),
    6: ("tab:pink",   "Cue (sep)"),
}
TARGET_COLORS = ["steelblue", "crimson"]


def _plot_trial(ax_in, ax_tgt, t, inputs_np, targets_np, stim_on, stim_off):
    """Plot one trial's inputs (top) and targets (bottom)."""
    # stim shading
    for ax in (ax_in, ax_tgt):
        for on, off in zip(stim_on, stim_off):
            ax.axvspan(on, off, alpha=0.10, color="gray", lw=0)

    # inputs — only channels that are ever nonzero in this trial
    active = np.where(np.abs(inputs_np).max(0) > 0.05)[0]
    for ch in active:
        col, lbl = INPUT_COLORS.get(ch, ("black", f"ch{ch}"))
        ax_in.plot(t, inputs_np[:, ch], color=col, lw=1.8, label=lbl)
    ax_in.axhline(0, color="0.8", lw=0.7)
    ax_in.set_ylabel("Input", fontsize=8)

    # targets — one line per rank; nan shown as gap
    n_ranks = targets_np.shape[-1]
    for rk in range(n_ranks):
        tgt = targets_np[:, rk].astype(float)
        col = TARGET_COLORS[rk % len(TARGET_COLORS)]
        ax_tgt.plot(t, np.where(np.isfinite(tgt), tgt, np.nan),
                    color=col, lw=2.2, label=f"target ch{rk}")
        # mark nan regions as light fill
        nan_mask = ~np.isfinite(tgt)
        if nan_mask.any():
            ax_tgt.fill_between(t, -1.6, 1.6, where=nan_mask,
                                alpha=0.08, color="gray", lw=0)
    ax_tgt.axhline(0,   color="0.7", lw=0.7, ls="--")
    ax_tgt.axhline(0.5, color="0.5", lw=0.7, ls=":")   # lick threshold
    ax_tgt.set_ylim(-1.7, 1.7)
    ax_tgt.set_ylabel("Target", fontsize=8)
    ax_tgt.set_xlabel("Time (s)", fontsize=8)


def plot_gng(cue_on_go_input=True):
    timing = TaskTiming([2.0, 4.0], [3.0, 5.0], 6.0, DT)
    input_size = 7 if cue_on_go_input else 8
    torch.manual_seed(0)
    X, y = generate_gng_trials(200, timing, input_size=input_size,
                                target_rank=1, noise=0.0,
                                cue_on_go_input=cue_on_go_input)
    X, y = X.numpy(), y.numpy()
    t = np.arange(X.shape[1]) * DT

    # pick one go trial and one nogo trial (noise-free so easy to identify)
    stim_epoch = slice(int(timing.n_stim_on[0]), int(timing.n_stim_off[0]))
    is_go = X[:, stim_epoch, 4].mean(1) > 0.5
    idx_go   = np.where( is_go)[0][0]
    idx_nogo = np.where(~is_go)[0][0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.4]})
    fig.suptitle(f"GNG trials  (cue_on_go_input={cue_on_go_input})", fontsize=11)

    for col, (idx, lbl) in enumerate([(idx_go, "Go"), (idx_nogo, "NoGo")]):
        axes[0, col].set_title(lbl, fontsize=10)
        _plot_trial(axes[0, col], axes[1, col],
                    t, X[idx], y[idx],
                    timing.stim_on, timing.stim_off)

    # shared legend
    handles = ([mpatches.Patch(color=INPUT_COLORS[ch][0], label=INPUT_COLORS[ch][1])
                for ch in INPUT_COLORS if ch < input_size] +
               [plt.Line2D([], [], color=TARGET_COLORS[0], lw=2, label="target ch0")])
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    path = os.path.join(OUT, f"gng_trials_cue{'on' if cue_on_go_input else 'off'}.pdf")
    fig.savefig(path, bbox_inches="tight"); print(f"Saved {path}")
    plt.close(fig)


def plot_dual(cue_on_go_input=True):
    timing = TaskTiming([2.0, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0], 10.0, DT)
    input_size = 7 if cue_on_go_input else 8
    torch.manual_seed(0)
    X, y, _, cnames = generate_dual_trials(500, timing, input_size=input_size,
                                            target_rank=2, noise=0.0,
                                            cue_on_go_input=cue_on_go_input)
    X, y, cnames = X.numpy(), y.numpy(), np.array(cnames)

    # pick one representative of each of the 6 main condition types
    pick = [
        ("A_C",      "DPA-only paired"),
        ("A_D",      "DPA-only unpaired"),
        ("A_go_C",   "Go paired"),
        ("A_go_D",   "Go unpaired"),
        ("A_nogo_C", "NoGo paired"),
        ("A_nogo_D", "NoGo unpaired"),
    ]

    fig, axes = plt.subplots(2, len(pick), figsize=(3.8 * len(pick), 5.5),
                              sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.4]})
    fig.suptitle(f"Dual trials  (cue_on_go_input={cue_on_go_input})", fontsize=11)
    t = np.arange(X.shape[1]) * DT

    for col, (cname, lbl) in enumerate(pick):
        idxs = np.where(cnames == cname)[0]
        if not len(idxs):
            print(f"  condition {cname!r} not found"); continue
        idx = idxs[0]
        axes[0, col].set_title(lbl, fontsize=8)
        _plot_trial(axes[0, col], axes[1, col],
                    t, X[idx], y[idx],
                    timing.stim_on, timing.stim_off)

    # legend
    handles = ([mpatches.Patch(color=INPUT_COLORS[ch][0], label=INPUT_COLORS[ch][1])
                for ch in INPUT_COLORS if ch < input_size] +
               [plt.Line2D([], [], color=TARGET_COLORS[r], lw=2,
                           label=f"target ch{r}  ({'memory' if r==0 else 'decision'})")
                for r in range(2)])
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    path = os.path.join(OUT, f"dual_trials_cue{'on' if cue_on_go_input else 'off'}.pdf")
    fig.savefig(path, bbox_inches="tight"); print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_gng(cue_on_go_input=True)
    plot_dual(cue_on_go_input=True)
    print("Done →", OUT)
