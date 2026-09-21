"""The accidental O(2) at initialization: (A,B) empirical covariance of (m0,m1,n0,n1) for the isotropic
recipe init vs an anisotropic one; (C,D) their autonomous flows; (E) radial and tangential field on the
ring; (F) angular harmonics of the tangential field (odd ones vanish because the field is odd).
Usage: o2_init_fig.py <sweep> <rid> <out.png>"""
import sys, os, math, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
sw, rid, out = sys.argv[1:4]
iso, cfg = build_init(sw, rid, symmetry="")                       # rho = 1, readout_scale = sqrt(lambda): isotropic
ani, _ = build_init(sw, rid, symmetry="", readout_scale=1.0)       # sigma(n1) = 1, sigma(m1) = lambda: anisotropic
sig = sig_of(cfg); XLIM = (-1.5, 1.5)
fig = plt.figure(figsize=(13.5, 8.2), dpi=150); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
names = ["m₀", "m₁", "n₀", "n₁"]
def cov_panel(ax, model, title):
    P = low_rank_numpy_params(model); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"])
    V = np.c_[M[:, 0], M[:, 1], Nv[:, 0], Nv[:, 1]]; C = np.cov(V.T)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-abs(C).max(), vmax=abs(C).max())
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=8, color="k")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_title(title, fontsize=10); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    return C
Ciso = cov_panel(fig.add_subplot(gs[0, 0]), iso, "A  isotropic init (ρ = 1, σ(n₁) = √λ): covariance of the four numbers per unit")
Cani = cov_panel(fig.add_subplot(gs[1, 0]), ani, "B  anisotropic init (σ(n₁) = 1, σ(m₁) = λ)")
def flow_panel(ax, model, title):
    cache, _ = _flow_panel_cache(model, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"],
                                 torch.zeros(1, 1, cfg["input_size"]), XLIM, XLIM, 101, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
    hm = _render_flow_panel(ax, cache, speed_vmax=0.9, sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XLIM, ylim=XLIM, model=model)
    ax.set_title(title, fontsize=10); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel("κ₀"); ax.set_ylabel("κ₁")
flow_panel(fig.add_subplot(gs[0, 1]), iso, "C  isotropic: a ring of slow states")
flow_panel(fig.add_subplot(gs[1, 1]), ani, "D  anisotropic: the discrete group only")
# E: radial / tangential profiles on the ring of the isotropic init
F_iso, _ = field_fn(iso, sig); F_ani, _ = field_fn(ani, sig)
th = np.linspace(0, 2 * np.pi, 360, endpoint=False); e_r = np.stack([np.cos(th), np.sin(th)], 1); e_t = np.stack([-np.sin(th), np.cos(th)], 1)
def ring_radius(F_of):
    rs = np.linspace(0.2, 1.5, 53); mr = [np.mean(np.sum(F_of(r * e_r) * e_r, 1)) for r in rs]
    for r0, r1, a, b in zip(rs[:-1], rs[1:], mr[:-1], mr[1:]):
        if a > 0 and b <= 0: return r0 + (r1 - r0) * a / (a - b)
    return float("nan")
rstar = ring_radius(F_iso); Fr = F_iso(rstar * e_r)
ax = fig.add_subplot(gs[0, 2]); ax.plot(np.degrees(th), np.sum(Fr * e_r, 1), color="#6b2f74", lw=1.4, label="radial  F·κ̂")
ax.plot(np.degrees(th), np.sum(Fr * e_t, 1), color="#0d818b", lw=1.4, label="tangential  F·κ̂⊥")
ax.axhline(0, color="0.8", lw=0.7); ax.set_xlabel("angle θ on the ring (°)"); ax.set_ylabel("field component")
ax.set_xticks([0, 90, 180, 270, 360]); ax.legend(frameon=False, fontsize=9)
ax.set_title(f"E  isotropic init on its ring, r* = {rstar:.2f}: both components ≈ 0, the tangential one is the corrugation", fontsize=9.5)
# F: harmonics of the tangential component
ax = fig.add_subplot(gs[1, 2]); Ft = np.sum(Fr * e_t, 1); c = np.fft.rfft(Ft) / len(Ft); k = np.arange(len(c))
amp = 2 * np.abs(c); ax.bar(k[1:9], amp[1:9], color=["#b0460e" if kk % 2 else "#0d818b" for kk in k[1:9]])
ax.set_xlabel("angular harmonic k"); ax.set_ylabel("amplitude of cos/sin kθ in F·κ̂⊥"); ax.set_xticks(range(1, 9))
odd_max = amp[1:9:2].max(); ax.set_title(f"F  even harmonics only: odd amplitudes ≤ {odd_max:.1e} (field exactly odd); k = 2 leads", fontsize=9.5)
ax.set_yscale("log"); ax.set_ylim(max(odd_max * 0.3, 1e-18), amp[1:9].max() * 3)
res_iso = residuals(F_iso, disk()); res_ani = residuals(F_ani, disk())
fig.suptitle(f"The accidental O(2) — isotropic covariance gives a ring; residuals on the disk: isotropic σ₁ {res_iso['s1']:.3f} σ₂ {res_iso['s2']:.3f} · anisotropic σ₁ {res_ani['s1']:.3f} σ₂ {res_ani['s2']:.3f}   ({rid} init)", fontsize=10.5)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("wrote", out, "r*", rstar, res_iso, res_ani)
