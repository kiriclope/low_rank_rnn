"""The symmetry ledger: field-equivariance violation for sigma_1, sigma_2, sigma_3 across the four
checkpoints (init, DPA, GNG, Dual), one line per network. Free recipe in gray (the two seeds that end
one-up-one-down in ember, dashed), sigma_1-tied in teal, whole-group-tied in plum.
Usage: sym_ledger_fig.py <symviol2.tsv> <out.png>"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
STAGES = ["init", "dpa", "naive", "expert"]; XL = ["init", "after DPA", "after GNG", "after Dual"]
data = {}
for r in rows:
    if len(r) < 5 or r[1] not in STAGES: continue
    data.setdefault(r[0], {})[r[1]] = [float(r[2]), float(r[3]), float(r[4])]
FAIL = {"s0_recipe7", "s3_recipe7"}
teal, plum, ember, gray = "#0d818b", "#6b2f74", "#b0460e", "0.55"
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), dpi=150, sharey=True)
titles = ["σ₁  pair exchange   (−κ₀, +κ₁)", "σ₂  inversion   (−κ₀, −κ₁)", "σ₃  test exchange   (+κ₀, −κ₁)"]
for j, ax in enumerate(axes):
    for rid, d in data.items():
        if not all(s in d for s in STAGES) or rid.startswith("s1_dualscan"): continue
        y = [d[s][j] for s in STAGES]
        if "recipe7" in rid:
            c, ls, lw, z = (ember, "--", 1.6, 3) if rid in FAIL else (gray, "-", 1.1, 1)
        elif "pair" in rid: c, ls, lw, z = teal, "-", 1.6, 2
        else: c, ls, lw, z = plum, "-", 1.6, 2
        ax.plot(range(4), y, color=c, ls=ls, lw=lw, marker="o", ms=3.5, zorder=z, alpha=0.9)
    ax.set_xticks(range(4)); ax.set_xticklabels(XL, fontsize=9); ax.set_title(titles[j], fontsize=10.5)
    ax.axhline(0, color="0.85", lw=0.7, zorder=0); ax.set_ylim(-0.03, 1.15)
    if j == 0: ax.set_ylabel("‖F(Dκ) − D F(κ)‖ / ‖F‖   (disk |κ| ≤ 1.5)", fontsize=9.5)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=gray, lw=1.1, marker="o", ms=3.5, label="free curriculum (8 seeds)"),
     Line2D([], [], color=ember, lw=1.6, ls="--", marker="o", ms=3.5, label="free, ends one-up-one-down (s0, s3)"),
     Line2D([], [], color=teal, lw=1.6, marker="o", ms=3.5, label="σ₁ tied through DPA, then released"),
     Line2D([], [], color=plum, lw=1.6, marker="o", ms=3.5, label="whole group tied through DPA, then released")]
axes[2].legend(handles=h, fontsize=7.8, frameon=False, loc="upper left")
fig.suptitle("Which symmetry each stage keeps — equivariance of the autonomous field, measured", fontsize=11.5, y=1.0)
fig.tight_layout(); fig.savefig(sys.argv[2], bbox_inches="tight"); print("wrote", sys.argv[2])
