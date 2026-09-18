"""Pairs-plot of the per-neuron low-rank vector (m₀, m₁, n₀, n₁): all 6 pairwise scatters + marginals,
coloured by POPULATION. Populations are fitted with a Gaussian mixture (BIC over k = 1..4 reported, so
the figure also answers "is this ensemble unimodal or a mixture?"); for a mirror-tied net the two halves
are marked as well, since they are the populations by construction.
Usage: mn_scatter.py <sweep_dir> <run_id> <stage> <out.png>        env: K (force k), NMAX (subsample)
"""
import sys, os, json, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
from bifurcation_probe import load_run
from src.flow_field import low_rank_numpy_params
sw, rid, stage, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m, cfg = load_run(sw, rid, stage=stage, device="cpu")
P = low_rank_numpy_params(m); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); N = M.shape[0]
V = np.c_[M[:, 0], M[:, 1], Nv[:, 0], Nv[:, 1]]; names = ["m₀", "m₁", "n₀", "n₁"]
J = cfg["gain"] * (Nv.T @ M) / N
try:
    from sklearn.mixture import GaussianMixture
    bic = {k: GaussianMixture(k, covariance_type="full", random_state=0, n_init=3).fit(V).bic(V) for k in (1, 2, 3, 4)}
    K = int(os.environ.get("K", min(bic, key=bic.get)))
    gm = GaussianMixture(K, covariance_type="full", random_state=0, n_init=5).fit(V); lab = gm.predict(V)
    bic_txt = "BIC " + "  ".join(f"k={k}:{v:,.0f}" for k, v in bic.items()) + f"   → best k = {min(bic, key=bic.get)}"
except Exception as e:
    K = int(os.environ.get("K", 2)); lab = (V[:, 0] > np.median(V[:, 0])).astype(int); bic_txt = f"(sklearn unavailable: {e})"
tied = bool(cfg.get("mirror_tying", False)); h = N // 2
sub = np.random.default_rng(0).choice(N, min(N, int(os.environ.get("NMAX", 1200))), replace=False)
cols = sns.color_palette("deep")   # seaborn deep, the project palette
fig, axes = plt.subplots(4, 4, figsize=(10.5, 10), dpi=150)
for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        if i == j:
            for k in range(K):
                ax.hist(V[lab == k, i], bins=40, alpha=0.55, color=cols[k], density=True)
            ax.set_yticks([]); ax.text(0.03, 0.85, names[i], transform=ax.transAxes, fontsize=11)
        else:
            for k in range(K):
                s = sub[lab[sub] == k]
                ax.scatter(V[s, j], V[s, i], s=3, alpha=0.5, color=cols[k], linewidths=0,
                           label=f"pop {k} (n={int((lab==k).sum())})" if (i, j) == (1, 0) else None)
            if tied:   # mark the two mirrored halves with faint edges
                ax.scatter(V[sub[sub < h], j], V[sub[sub < h], i], s=14, facecolors="none",
                           edgecolors="k", linewidths=0.2, alpha=0.25)
            ax.axhline(0, color="0.85", lw=0.6, zorder=0); ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.tick_params(labelsize=6)
        if i == 3: ax.set_xlabel(names[j], fontsize=9)
        if j == 0 and i != 0: ax.set_ylabel(names[i], fontsize=9)
axes[1][0].legend(fontsize=7, frameon=False, markerscale=3, loc="upper left")
mu = np.array([V[lab == k].mean(0) for k in range(K)])
sub_t = "   ".join(f"pop{k} mean ({mu[k,0]:+.2f},{mu[k,1]:+.2f},{mu[k,2]:+.2f},{mu[k,3]:+.2f})" for k in range(K))
fig.suptitle(f"{rid} — {stage.upper()} ckpt   |   J = [[{J[0,0]:.2f}, {J[0,1]:+.2f}], [{J[1,0]:+.2f}, {J[1,1]:.2f}]]   "
             f"σ = ({V[:,0].std():.2f}, {V[:,1].std():.2f}, {V[:,2].std():.2f}, {V[:,3].std():.2f})"
             f"{'   [mirror-tied: black rings = half 1]' if tied else ''}\n{bic_txt}\n{sub_t}", fontsize=9)
sns.despine(fig=fig)
fig.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs(os.path.dirname(out), exist_ok=True)
stem = os.path.splitext(out)[0]                     # save_fig convention: PNG (gallery) + SVG (vector)
fig.savefig(stem + ".png", bbox_inches="tight", dpi=150); fig.savefig(stem + ".svg", bbox_inches="tight")
print("saved", stem + ".png", "| K =", K)
