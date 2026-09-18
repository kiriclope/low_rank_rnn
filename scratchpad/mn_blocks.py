"""Does the orbit relation hold? For every unit i of block 0 and every group element sigma, plot the
value the constraint predicts for the image unit (x, = D_sigma applied to block 0) against the value
the network actually has there (y). Exact equivariance = every point on the identity line.
Usage: mn_blocks.py <sweep_dir> <out.png> <run_id>:<kind> ...    kind = pair | klein
"""
import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
from bifurcation_probe import load_run
from src.flow_field import low_rank_numpy_params
sw, out = sys.argv[1], sys.argv[2]
specs = [s.split(":") for s in sys.argv[3:]]
STAGES = ["dpa", "expert"]
NAMES = ["m₀", "m₁", "n₀", "n₁"]
cols = sns.color_palette("deep")
SA = [+1, +1, -1, -1]; SB = [+1, -1, +1, -1]
LBL = {"klein": ["1", "σ₃", "σ₁", "σ₂"], "pair": ["1", "σ₁"]}

nrow = len(specs) * len(STAGES)
fig, axes = plt.subplots(nrow, 4, figsize=(12.0, 2.45 * nrow), dpi=150, squeeze=False)
r = 0
for rid, kind in specs:
    for stage in STAGES:
        m, cfg = load_run(sw, rid, stage=stage, device="cpu")
        P = low_rank_numpy_params(m); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); N = M.shape[0]
        J = cfg["gain"] * (Nv.T @ M) / N
        V = np.c_[M[:, 0], M[:, 1], Nv[:, 0], Nv[:, 1]]
        nb = 4 if kind == "klein" else 2; B = N // nb
        bl = [V[k * B:(k + 1) * B] for k in range(nb)]
        rng = np.random.default_rng(0); sub = rng.choice(B, min(B, 350), replace=False)
        dev = 0.0
        for c in range(4):
            ax = axes[r][c]
            for k in range(1, nb):
                if kind == "klein": s = [SA[k], SB[k], SA[k], SB[k]][c]
                else:               s = [-1, 1, -1, 1][c]
                pred = s * bl[0][:, c]; got = bl[k][:, c]
                dev = max(dev, float(np.abs(got - pred).max()))
                ax.scatter(pred[sub], got[sub], s=5, alpha=0.5, color=cols[k], linewidths=0,
                           label=LBL[kind][k] if c == 0 else None)
            lo, hi = ax.get_xlim(); lim = max(abs(lo), abs(hi))
            ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=0.8, ls="--", zorder=0)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel(f"predicted  D$_σ$·{NAMES[c]}", fontsize=9)
            if c == 0: ax.set_ylabel(f"actual", fontsize=9)
            ax.tick_params(labelsize=7)
        held = "HELD" if dev < 1e-9 else "released"
        axes[r][0].legend(fontsize=8, frameon=False, markerscale=3, loc="lower right", title="σ", title_fontsize=8)
        axes[r][1].set_title(f"{rid} — {stage.upper()} checkpoint, constraint {held}      "
                             f"max |actual − predicted| = {dev:.2g}      "
                             f"J off-diagonal  {J[0,1]:+.3f} / {J[1,0]:+.3f}",
                             fontsize=9.5, loc="left")
        r += 1
fig.suptitle("The constraint, checked unit by unit — every image unit against the value its group element predicts",
             fontsize=11.5, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.2); fig.savefig(out, bbox_inches="tight"); print("wrote", out, "dev", dev)
