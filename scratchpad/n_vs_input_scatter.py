"""Rows = the low-rank vectors (n₀, n₁, m₀, m₁), columns = the INPUT weight columns (A, B, C, D, go/cue,
nogo): per-neuron scatter of each low-rank component against each input weight, coloured by the same
population labels as mn_scatter.py (a Gaussian mixture fitted on (m₀, m₁, n₀, n₁), k chosen by BIC).
Each panel prints Pearson r. Project matplotlib conventions; PNG + SVG via the save_fig convention.
Usage: n_vs_input_scatter.py <sweep_dir> <run_id> <stage> <out.png>     env: K, NMAX
"""
import sys, os, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
from bifurcation_probe import load_run
from src.flow_field import low_rank_numpy_params
sw, rid, stage, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m, cfg = load_run(sw, rid, stage=stage, device="cpu")
P = low_rank_numpy_params(m); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); N = M.shape[0]
Wi = np.asarray(P["Wi"]) if "Wi" in P else m.wi.weight.detach().numpy()
if Wi.shape[0] != N: Wi = Wi.T
V = np.c_[M[:, 0], M[:, 1], Nv[:, 0], Nv[:, 1]]
ROWS = [("n₀", Nv[:, 0]), ("n₁", Nv[:, 1]), ("m₀", M[:, 0]), ("m₁", M[:, 1])]
lbl = ["A", "B", "C", "D", "go / cue", "nogo", "rwd"][:Wi.shape[1]]
COLS = [(lbl[c], Wi[:, c]) for c in range(Wi.shape[1])]
try:
    from sklearn.mixture import GaussianMixture
    bic = {k: GaussianMixture(k, covariance_type="full", random_state=0, n_init=3).fit(V).bic(V) for k in (1, 2, 3, 4)}
    K = int(os.environ.get("K", min(bic, key=bic.get)))
    lab = GaussianMixture(K, covariance_type="full", random_state=0, n_init=5).fit(V).predict(V)
    bic_txt = "populations from a GMM on (m₀, m₁, n₀, n₁): " + "  ".join(f"k={k}:{v:,.0f}" for k, v in bic.items()) + f"  → k = {K}"
except Exception as e:
    K = 2; lab = (V[:, 0] > 0).astype(int); bic_txt = f"(sklearn unavailable: {e})"
sub = np.random.default_rng(0).choice(N, min(N, int(os.environ.get("NMAX", 1200))), replace=False)
cols = sns.color_palette("deep")
fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(2.3 * len(COLS), 2.25 * len(ROWS)), dpi=150, squeeze=False)
for i, (rn, rv) in enumerate(ROWS):
    for j, (cn, cv) in enumerate(COLS):
        ax = axes[i][j]
        for k in range(K):
            s = sub[lab[sub] == k]
            ax.scatter(cv[s], rv[s], s=3, alpha=0.5, color=cols[k], linewidths=0,
                       label=f"pop {k} (n={int((lab==k).sum())})" if (i, j) == (0, 0) else None)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0); ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        r = float(np.corrcoef(cv, rv)[0, 1])
        ax.text(0.04, 0.92, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=8, va="top")
        ax.tick_params(labelsize=6)
        if i == 0: ax.set_title(f"w$_{{{cn}}}$", fontsize=10)
        if j == 0: ax.set_ylabel(rn, fontsize=11)
        if i == len(ROWS) - 1: ax.set_xlabel("input weight", fontsize=8)
axes[0][0].legend(fontsize=7, frameon=False, markerscale=3, loc="lower right")
fig.suptitle(f"{rid} — {stage.upper()} ckpt: low-rank vectors vs INPUT weights\n{bic_txt}", fontsize=10)
sns.despine(fig=fig); fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(out), exist_ok=True)
stem = os.path.splitext(out)[0]
fig.savefig(stem + ".png", bbox_inches="tight", dpi=150); fig.savefig(stem + ".svg", bbox_inches="tight")
print("saved", stem + ".png", "| K =", K)
