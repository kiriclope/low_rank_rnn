"""Finite-N residuals at initialization: v_sigma vs N for sigma_1, sigma_2, sigma_3 (4 seeds each), with a
1/sqrt(N) guide, and the residual map |F(D1 k) - D1 F(k)| on the disk for one init.
Usage: init_scaling_fig.py <sweep> <rid> <out.png>"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
sw, rid, out = sys.argv[1:4]; NS = [256, 512, 1024, 2048, 4096, 8192]; SEEDS = [0, 1, 2, 3]; pts = disk(31)
rows = []
for N in NS:
    for s in SEEDS:
        m, cfg = build_init(sw, rid, N=N, seed=s, symmetry=""); F_of, _ = field_fn(m, sig_of(cfg)); r = residuals(F_of, pts)
        rows.append((N, s, r["s1"], r["s2"], r["s3"])); print(N, s, r, flush=True)
A = np.array(rows)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), dpi=150, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axes[0]; cols = {"σ₁": "#0d818b", "σ₂": "#b0460e", "σ₃": "#6b2f74"}
for j, (lab, c) in enumerate(cols.items()):
    y = A[:, 2 + j]; ax.scatter(A[:, 0] * (1 + 0.04 * (j - 1)), np.maximum(y, 1e-17), s=18, color=c, label=lab, zorder=3)
    med = [np.median(A[A[:, 0] == N, 2 + j]) for N in NS]; ax.plot(NS, np.maximum(med, 1e-17), color=c, lw=1.2, alpha=0.7)
ref = A[A[:, 0] == 1024, 2].mean(); ax.plot(NS, ref * np.sqrt(1024 / np.array(NS)), color="0.5", ls="--", lw=1, label="∝ N^(−1/2)")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("N (units)"); ax.set_ylabel("residual ‖F(Dκ) − D F(κ)‖ / ‖F‖")
ax.set_xticks(NS); ax.set_xticklabels(NS); ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title("A  at initialization, 4 seeds per N: σ₂ is exactly zero, σ₁ and σ₃ shrink as N^(−1/2)", fontsize=9.5)
ax.set_ylim(1e-9, 1)
m, cfg = build_init(sw, rid, N=1024, seed=0, symmetry=""); F_of, _ = field_fn(m, sig_of(cfg))
g = np.linspace(-1.5, 1.5, 61); X, Y = np.meshgrid(g, g); P = np.stack([X.ravel(), Y.ravel()], 1); msk = np.hypot(*P.T) <= 1.5
F = F_of(P); R = F_of(P @ D["s1"].T) - F @ D["s1"].T
for ax, Z, ttl in [(axes[1], np.linalg.norm(F, axis=1), "B  ‖F(κ)‖, N = 1024 init"), (axes[2], np.linalg.norm(R, axis=1), "C  ‖F(D₁κ) − D₁F(κ)‖, same init")]:
    Z = np.where(msk, Z, np.nan).reshape(X.shape); im = ax.imshow(Z, extent=[-1.5, 1.5, -1.5, 1.5], origin="lower", cmap="magma", vmin=0, vmax=np.nanmax(np.linalg.norm(F, axis=1)) if ax is axes[1] else None)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title(ttl, fontsize=9.5); ax.set_xlabel("κ₀"); ax.set_ylabel("κ₁"); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
fig.suptitle("What the initialization provides: exact inversion, finite-N residual for the pair and test exchanges", fontsize=10.5)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); np.savetxt(out.replace(".png", ".tsv"), A, fmt="%.5g", delimiter="\t", header="N seed v_s1 v_s2 v_s3"); print("wrote", out)
