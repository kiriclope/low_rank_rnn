"""How training breaks the inversion: (A) |<n>| per stage, (B) bias rms per stage, (C) the even part of the
field split into its two terms, (D) exactness of the identity E(k) = <n>/2 + bias term on a trained net.
Usage: inversion_breaking_fig.py <symviol2.tsv> <sweep> <rid_pattern_with_{s}> <out.png>"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
tsv, sw, pat, out = sys.argv[1:5]
rows = [l.rstrip("\n").split("\t") for l in open(tsv)][1:]
ST = ["init", "dpa", "naive", "expert"]; XL = ["init", "DPA", "GNG", "Dual"]
d = {}
for r in rows:
    if len(r) < 13 or "recipe7" not in r[0]: continue
    d.setdefault(r[0], {})[r[1]] = dict(v1=float(r[2]), v2=float(r[3]), n0=float(r[5]), n1=float(r[6]), E=float(r[7]), Ec=float(r[8]), Er=float(r[9]), b=float(r[10]))
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1), dpi=150, gridspec_kw=dict(wspace=0.42))
gray, ember, teal, plum = "0.5", "#b0460e", "#0d818b", "#6b2f74"
for rid, dd in d.items():
    if not all(s in dd for s in ST): continue
    c = ember if rid in ("s0_recipe7", "s3_recipe7") else gray
    axes[0].plot(range(4), [np.hypot(dd[s]["n0"], dd[s]["n1"]) for s in ST], color=c, marker="o", ms=3, lw=1.1)
    axes[1].plot(range(4), [dd[s]["b"] for s in ST], color=c, marker="o", ms=3, lw=1.1)
    axes[2].plot(range(4), [dd[s]["Ec"] for s in ST], color=teal, marker="o", ms=3, lw=1.0, alpha=0.8)
    axes[2].plot(range(4), [dd[s]["Er"] for s in ST], color=plum, marker="s", ms=3, lw=1.0, alpha=0.8)
for ax, t, yl in [(axes[0], "A  ‖⟨n⟩‖, the mean of the read vectors", "‖⟨n⟩‖"), (axes[1], "B  bias, rms over units", "rms(b)"), (axes[2], "C  even part of the field (rms)", "rms")]:
    ax.set_xticks(range(4)); ax.set_xticklabels(XL); ax.set_title(t, fontsize=9.5); ax.set_ylabel(yl, fontsize=9); ax.set_ylim(bottom=0)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([], [], color=gray, label="free recipe, 8 seeds"), Line2D([], [], color=ember, label="ends one-up-one-down")], frameon=False, fontsize=8)
axes[2].legend(handles=[Line2D([], [], color=teal, marker="o", label="constant term  ½⟨n⟩"), Line2D([], [], color=plum, marker="s", label="bias term")], frameon=False, fontsize=8)
# D: exactness of the identity on one trained net (noise-free field)
m, cfg = load_run(sw, pat.format(s=1), stage="expert", device="cpu"); P = low_rank_numpy_params(m)
M, Nv, b, g = np.asarray(P["M"]), np.asarray(P["Nvec"]), np.asarray(P["bi"]).ravel(), float(P["gain"])
pts = disk(25); phi = PHI["lif"]; F = raw_field(M, Nv, b, g, phi, pts); E = 0.5 * (F + raw_field(M, Nv, b, g, phi, -pts))
rhs = np.stack([0.5 * Nv.mean(0) + 0.5 * (Nv.T @ (phi(g * (M @ k + b)) - phi(g * (M @ k - b)))) / M.shape[0] for k in pts])
ax = axes[3]; ax.scatter(rhs[:, 0], E[:, 0], s=6, color=teal, label="κ₀ component"); ax.scatter(rhs[:, 1], E[:, 1], s=6, color=plum, label="κ₁ component")
lim = np.abs(np.r_[E, rhs]).max() * 1.1; ax.plot([-lim, lim], [-lim, lim], color="0.4", ls="--", lw=0.8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("from the parameters: ½⟨n⟩ + bias term"); ax.set_ylabel("from the field: ½[F(κ)+F(−κ)]", fontsize=9); ax.legend(frameon=False, fontsize=8)
ax.set_title(f"D  the identity, point by point\n({pat.format(s=1)} after Dual, max error {np.abs(E - rhs).max():.0e})", fontsize=9)
fig.suptitle("Training breaks the inversion through the read-vector mean and the bias, and through nothing else", fontsize=10.5)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
