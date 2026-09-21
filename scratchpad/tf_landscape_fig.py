"""Untrained only. How the transfer function and the two inversion-breaking parameters shape the landscape:
rows = transfer function (lif = Gaussian CDF, tanh, relu), columns = the SAME structured init (recipe, lambda 7,
rho 1) with (i) b = 0, <n> = 0; (ii) <n1> = -0.3; (iii) random bias b ~ N(0,1); (iv) both. Each panel: flow,
attractors, the inversion residual. Top strip: the residual for random (m, n) from four asymmetric laws.
Usage: tf_landscape_fig.py <sweep> <rid> <out.png>"""
import sys, math, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from bifurcation_probe import find_wells
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
sw, rid, out = sys.argv[1:4]; rng = np.random.default_rng(0); N = 1024; g = 1.0
TFS = [("lif", "lif  (Gaussian cumulative: ½ + odd)", 1.5), ("tanh", "tanh  (odd)", 3.5), ("relu", "relu  (neither)", 1.5)]
ROWS = TFS[:2]   # relu has no attractor at this overlap (a runaway, §28): strip only, no landscape row
PHI3 = {"lif": PHI["lif"], "tanh": np.tanh, "relu": lambda u: np.maximum(u, 0.0)}
# ---------- strip A: random asymmetric laws, b = 0 ----------
pts = disk(21)
fams = {"Gaussian, mean 1": lambda: rng.normal(1.0, 2.0, (N, 2)), "uniform on [0, 4]": lambda: rng.uniform(0, 4, (N, 2)),
        "lognormal": lambda: rng.lognormal(0.5, 0.6, (N, 2)), "two clusters, 1 : 3": lambda: np.where(rng.random((N, 1)) < 0.25, rng.normal(3, 0.5, (N, 2)), rng.normal(-1, 0.5, (N, 2)))}
res = {k: {t: [] for t, _, _ in TFS} for k in fams}
for name, draw in fams.items():
    for rep in range(10):
        M = draw() / np.sqrt(N) * 6; Nv = draw() / np.sqrt(N) * 6
        for t, _, _ in TFS: res[name][t].append(raw_residual(M, Nv, np.zeros(N), g, PHI3[t], pts))
fig = plt.figure(figsize=(17, 10.4), dpi=140); gs = fig.add_gridspec(3, 4, height_ratios=[0.75, 1, 1])
ax = fig.add_subplot(gs[0, :2]); cols = {"lif": "#b0460e", "tanh": "#0d818b", "relu": "#6b2f74"}
for i, name in enumerate(fams):
    for j, (t, _, _) in enumerate(TFS):
        y = np.maximum(res[name][t], 1e-17); ax.scatter(np.full(len(y), i + (j - 1) * 0.22) + rng.normal(0, 0.03, len(y)), y, s=12, color=cols[t], label=t if i == 0 else None)
ax.set_yscale("log"); ax.set_xticks(range(len(fams))); ax.set_xticklabels(list(fams), fontsize=8); ax.set_ylabel("inversion residual\n‖F(−κ)+F(κ)‖ / ‖F‖", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="center right", ncol=3); ax.set_ylim(1e-17, 3)
ax.set_title("A  random (m, n) from asymmetric laws, b = 0, N = 1024, 10 draws per law — odd for tanh only", fontsize=9.5)
ax = fig.add_subplot(gs[0, 2:]); ax.axis("off")
ax.text(0, 0.95, "Theorem 6.1 (derivations):   Ψ(κ) + Ψ(−κ) = 2c⟨n⟩ + (1/N) Σᵢ nᵢ [ψ(uᵢ+βᵢ) − ψ(uᵢ−βᵢ)]   for φ = c + ψ, ψ odd", fontsize=9.5, va="top", family="serif")
ax.text(0, 0.72, "lif:   c = ½  → the readout mean ⟨n⟩ shifts the whole field rigidly by ½⟨n⟩; the bias deforms it.\n"
                 "tanh:  c = 0  → ⟨n⟩ cannot break the inversion at all; only the bias can.\n"
                 "relu:  not of this form → no identity; oddness needs a sign-symmetric population (Theorem 6.3), which a finite sample is not (strip only: relu has no attractor at this overlap).",
        fontsize=9, va="top", family="serif", linespacing=1.6)
ax.text(0, 0.18, "Rows below: the same untrained network (recipe init, λ = 7, ρ = 1) under lif and under tanh.\nColumns: b = 0 and ⟨n⟩ = 0  |  ⟨n₁⟩ = −0.3  |  random bias, rms 1  |  both.", fontsize=9, va="top", family="serif", linespacing=1.5)
# ---------- grid B..M ----------
letters = "BCDEFGHIJKLM"; k = 0
conds = [("b = 0, ⟨n⟩ = 0", dict(dec=0.0, bias=False)), ("⟨n₁⟩ = −0.3", dict(dec=-0.3, bias=False)), ("random bias b, rms 1", dict(dec=0.0, bias=True)), ("both", dict(dec=-0.3, bias=True))]
for r, (tf, tflab, L) in enumerate(ROWS):
    for c, (clab, cd) in enumerate(conds):
        m, cfg = build_init(sw, rid, symmetry="", decision_readout_mean=cd["dec"])
        if cd["bias"]:
            with torch.no_grad(): m.wi.bias.copy_(torch.tensor(np.random.default_rng(1).normal(0, 1.0, N), dtype=m.wi.bias.dtype))
        mdl, cf = swap_phi(m, cfg, tf, zero_bias=False); sig = sig_of(cf); XL = (-L, L)
        ax = fig.add_subplot(gs[r + 1, c])
        cache, spd = _flow_panel_cache(mdl, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"], torch.zeros(1, 1, cfg["input_size"]), XL, XL, 81, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
        _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=mdl)
        ax.axhline(0, color="w", lw=0.9, ls=(0, (5, 4)), alpha=0.9)
        F_of, _ = field_fn(mdl, sig); rr = residuals(F_of, disk(21, L))
        try: w = [f for f, kd, t in find_wells(mdl, cf, xlim=L + 0.5, n_seeds=41, noise_sigma=sig, with_eigs=True) if str(kd).lower().startswith(("stable", "attract"))]
        except Exception: w = []
        w = [f for f in w if np.linalg.norm(f) > 0.15]
        ax.set_title(f"{letters[k]}  {tf}: {clab}\nσ₂ residual {rr['s2']:.1e} · {len(w)} attractors" + (f", mean κ₁ {np.mean([f[1] for f in w]):+.2f}" if w else ""), fontsize=8.2)
        tk = [-1, 0, 1] if L < 2 else [-3, 0, 3]; ax.set_xticks(tk); ax.set_yticks(tk)
        ax.set_xlabel("κ₀" if r == len(ROWS) - 1 else ""); ax.set_ylabel(("κ₁\n" + tflab) if c == 0 else "", fontsize=9)
        print(letters[k], tf, clab, rr, [(round(float(f[0]), 2), round(float(f[1]), 2)) for f in w], flush=True); k += 1
fig.suptitle("How the transfer function and the two inversion-breaking parameters shape the landscape — untrained networks only", fontsize=11.5, y=0.995)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
