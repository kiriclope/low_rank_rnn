"""The symmetry ledger, small multiples: rows = group element, columns = curriculum (free / sigma_1 held in
DPA / whole group held in DPA); thin lines per seed, thick line the median; the two free seeds that end
one-up-one-down dashed in ember.  Usage: sym_ledger_fig.py <symviol2.tsv> <out.png>"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
ST = ["init", "dpa", "naive", "expert"]; XL = ["init", "DPA", "GNG", "Dual"]
data = {}
for r in rows:
    if len(r) < 5 or r[1] not in ST or r[0].startswith("s1_dualscan"): continue
    data.setdefault(r[0], {})[r[1]] = [float(r[2]), float(r[3]), float(r[4])]
FAIL = {"s0_recipe7", "s3_recipe7"}
COLS = [("recipe7", "free curriculum (8 seeds)", "0.55"), ("symdpa_pair", "σ₁ held through DPA, released", "#0d818b"), ("symdpa_klein", "whole group held through DPA, released", "#6b2f74")]
ROWS = ["σ₁  pair exchange", "σ₂  inversion", "σ₃  test exchange"]
fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.4), dpi=150, sharex=True, sharey=True)
for i in range(3):
    for j, (key, lab, c) in enumerate(COLS):
        ax = axes[i][j]; ys = []
        for rid, d in data.items():
            if key not in rid or not all(s in d for s in ST): continue
            y = [d[s][i] for s in ST]; ys.append(y)
            if rid in FAIL: ax.plot(range(4), y, color="#b0460e", ls="--", lw=1.5, marker="o", ms=3.2, zorder=3)
            else: ax.plot(range(4), y, color=c, lw=0.9, alpha=0.55, marker="o", ms=2.6)
        if ys: ax.plot(range(4), np.median(ys, 0), color=c if key != "recipe7" else "0.25", lw=2.6, zorder=4)
        ax.axhline(0, color="0.85", lw=0.7, zorder=0); ax.set_ylim(-0.03, 1.15); ax.set_xticks(range(4)); ax.set_xticklabels(XL, fontsize=9)
        if i == 0: ax.set_title(lab, fontsize=10)
        if j == 0: ax.set_ylabel(ROWS[i] + "\nresidual", fontsize=10)
        ax.text(0.03, 0.9, f"after Dual: {np.median([y[-1] for y in ys]):.2f}" if ys else "", transform=ax.transAxes, fontsize=8, color="0.3")
axes[2][0].text(0.03, 0.75, "dashed: the two seeds that end\none-up-one-down", transform=axes[2][0].transAxes, fontsize=7.5, color="#b0460e")
fig.suptitle("Which symmetry each stage keeps — equivariance residual ‖F(Dκ) − D F(κ)‖ / ‖F‖ of the autonomous field (disk |κ| ≤ 1.5); thin = seeds, thick = median", fontsize=10.5)
fig.tight_layout(); fig.savefig(sys.argv[2], bbox_inches="tight"); print("wrote", sys.argv[2])
