"""Population structure with the grid layout: rows = conditions, first column = the predicted cluster
lattice in the (m0, m1) plane (scheme), then the three informative scatters of seed 0: m0-m1 (the lattice),
m0-n0 and m1-n1 (write and read share a sign within a mode). Colored by block for tied nets, by a
4-component mixture for free ones. Usage: mn_grid_fig.py <out.png> <stage> <title> <row>...  row = label|sweep|arm|scheme|seed"""
import sys, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
out, stage, title = sys.argv[1:4]; rows = [r.split("|") for r in sys.argv[4:]]
teal, plum, ember = "#0d818b", "#6b2f74", "#b0460e"; cols = sns.color_palette("deep")
NB = {"pair": 2, "inv": 2, "test": 2, "klein": 4}
QC = {(1, 1): cols[0], (-1, 1): cols[1], (-1, -1): cols[2], (1, -1): cols[3]}     # quadrant of (m0, m1) -> color
def scheme(ax, kind):
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("m₀", fontsize=8, labelpad=1); ax.set_ylabel("m₁", fontsize=8)
    g = (0.8, 0.5)
    imgs = {"free": [], "pair": [(-0.8, 0.5)], "inv": [(-0.8, -0.5)], "test": [(0.8, -0.5)], "klein": [(-0.8, 0.5), (0.8, -0.5), (-0.8, -0.5)]}[kind]
    if kind == "free": ax.text(0.5, 0.5, "?", transform=ax.transAxes, ha="center", va="center", fontsize=24, color="0.6")
    else:
        ax.plot(*g, "o", color=QC[(1, 1)], ms=10, zorder=4)
        for (x, y) in imgs:
            ax.plot(x, y, "o", ms=10, mfc="none", mec=QC[(int(np.sign(x)), int(np.sign(y)))], mew=1.8, zorder=4)
            ax.annotate("", (x, y), g, arrowprops=dict(arrowstyle="->", color=plum, lw=0.9, linestyle=":", shrinkA=6, shrinkB=6))
        # the decision-axis pair (m₀ = 0): its own image under σ₁ and σ₃, antipodal under σ₂ and the group
        for y in (1.0, -1.0): ax.plot(0, y, "o", ms=8, mfc="none", mec="0.45", mew=1.4, zorder=3)
        if kind in ("inv", "klein"): ax.annotate("", (0, -1.0), (0, 1.0), arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8, linestyle=":", shrinkA=6, shrinkB=6))
    note = {"free": "no constraint", "pair": "σ₁: every cluster has its\nmirror image across the m₁ axis;\nan axis cluster is its own image", "inv": "σ₂: every cluster has\nits antipode; the axis pair too", "test": "σ₃: every cluster has its\nmirror image across the m₀ axis;\nthe axis pair is exchanged", "klein": "whole group: every cluster\nhas three images; the axis\nclusters come as a pair"}[kind]
    ax.set_title("predicted cluster lattice", fontsize=8.5, color="0.35"); ax.text(0.5, -0.12, note, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
fig, axes = plt.subplots(len(rows), 4, figsize=(13.2, 3.1 * len(rows)), dpi=140, squeeze=False, gridspec_kw=dict(width_ratios=[0.85, 1, 1, 1]))
PAIRS = [(0, 1, "m₀", "m₁", "the lattice"), (0, 2, "m₀", "n₀", "write and read, memory mode"), (1, 3, "m₁", "n₁", "write and read, decision mode")]
for r, (label, sw, arm, kind, seed) in enumerate(rows):
    scheme(axes[r][0], kind); axes[r][0].text(-0.3, 0.5, label, transform=axes[r][0].transAxes, rotation=90, ha="center", va="center", fontsize=9.5)
    rid = f"s{seed}_{arm}"
    if not os.path.exists(f"{sw}/{rid}/{stage}_{rid}.pth"):
        for c in range(1, 4): axes[r][c].text(0.5, 0.5, f"{arm}\n(not run yet)", ha="center", va="center", transform=axes[r][c].transAxes); axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        continue
    m, cfg = load_run(sw, rid, stage=stage, device="cpu"); P = low_rank_numpy_params(m); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); N = M.shape[0]
    V = np.c_[M[:, 0], M[:, 1], Nv[:, 0], Nv[:, 1]]; J = cfg["gain"] * (Nv.T @ M) / N
    lab = np.array([[(1, 1), (-1, 1), (-1, -1), (1, -1)].index((int(np.sign(a)) or 1, int(np.sign(b)) or 1)) for a, b in zip(V[:, 0], V[:, 1])]); how = "colored by the sign quadrant of (m₀, m₁)"
    rng = np.random.default_rng(0); sub = rng.choice(N, min(N, 900), replace=False)
    for c, (i, j, ni, nj, what) in enumerate(PAIRS):
        ax = axes[r][c + 1]
        for k in range(lab.max() + 1):
            s_ = sub[lab[sub] == k]; ax.scatter(V[s_, i], V[s_, j], s=4, alpha=0.55, color=cols[k], linewidths=0)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0); ax.axvline(0, color="0.85", lw=0.6, zorder=0); ax.set_xlabel(ni, fontsize=9); ax.set_ylabel(nj, fontsize=9); ax.tick_params(labelsize=7)
        ax.set_title(what if r == 0 else "", fontsize=9)
    axes[r][1].text(0.02, 0.97, f"seed {seed} · J₀₁ {J[0,1]:+.2f}  J₁₀ {J[1,0]:+.2f}", transform=axes[r][1].transAxes, va="top", fontsize=7.5, color="0.3")
    print(rid, how, "J", np.round(J, 3).tolist(), flush=True)
fig.text(0.5, 0.003, "Points colored by the sign quadrant of (m₀, m₁): blue (+,+), orange (−,+), green (−,−), red (+,−). A tie predicts which quadrants are images of which; the middle and right panels show that n follows the sign of m within each mode.", ha="center", fontsize=8.3, color="0.3"); fig.suptitle(title, fontsize=11, y=0.995); fig.tight_layout(rect=[0, 0.012, 1, 0.98]); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
