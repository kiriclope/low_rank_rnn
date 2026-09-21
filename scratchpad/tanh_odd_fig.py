"""tanh: the autonomous field is odd for ANY m, n when b = 0. (A) inversion residual for random (m, n) from
four asymmetric laws, with tanh / lif / tanh + random bias; (B-D) the trained target network's parameters
under lif as trained, under tanh with b = 0, under tanh with the trained bias.
Usage: tanh_odd_fig.py <sweep> <rid> <out.png>"""
import sys, math, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from bifurcation_probe import find_wells
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
sw, rid, out = sys.argv[1:4]; rng = np.random.default_rng(0); N = 1024; pts = disk(21); g = 1.0
fams = {"Gaussian, mean 1": lambda: rng.normal(1.0, 2.0, (N, 2)), "uniform on [0, 4]": lambda: rng.uniform(0, 4, (N, 2)),
        "lognormal": lambda: rng.lognormal(0.5, 0.6, (N, 2)), "two clusters, 1 : 3": lambda: np.where(rng.random((N, 1)) < 0.25, rng.normal(3, 0.5, (N, 2)), rng.normal(-1, 0.5, (N, 2)))}
res = {k: {"tanh, b = 0": [], "lif, b = 0": [], "tanh, random b": []} for k in fams}
for name, draw in fams.items():
    for rep in range(12):
        M = draw(); Nv = draw() * rng.choice([-1, 1], (N, 1)) * 0 + draw()      # independent draws for m and n
        Nv = Nv / np.sqrt(N) * 6; M = M / np.sqrt(N) * 6                      # bring lambda to O(5-10)
        b = rng.normal(0, 1.0, N)
        res[name]["tanh, b = 0"].append(raw_residual(M, Nv, np.zeros(N), g, np.tanh, pts))
        res[name]["lif, b = 0"].append(raw_residual(M, Nv, np.zeros(N), g, PHI["lif"], pts))
        res[name]["tanh, random b"].append(raw_residual(M, Nv, b, g, np.tanh, pts))
fig = plt.figure(figsize=(18, 4.6), dpi=150); gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 1, 1, 1, 1])
ax = fig.add_subplot(gs[0, 0]); cols = {"tanh, b = 0": "#0d818b", "lif, b = 0": "#b0460e", "tanh, random b": "#6b2f74"}
for i, name in enumerate(fams):
    for j, (cond, c) in enumerate(cols.items()):
        y = np.maximum(res[name][cond], 1e-17); ax.scatter(np.full(len(y), i + (j - 1) * 0.22) + rng.normal(0, 0.03, len(y)), y, s=12, color=c, label=cond if i == 0 else None)
ax.set_yscale("log"); ax.set_xticks(range(len(fams))); ax.set_xticklabels(list(fams), fontsize=8, rotation=12); ax.set_ylabel("inversion residual  ‖F(−κ) + F(κ)‖ / ‖F‖")
ax.legend(frameon=False, fontsize=8, loc="center right"); ax.set_title("A  random (m, n), N = 1024, 12 draws per law", fontsize=9.5); ax.set_ylim(1e-17, 3)
m_lif, cfg = load_run(sw, rid, stage="expert", device="cpu"); sig = sig_of(cfg)
m_t0, cfg_t = swap_phi(m_lif, cfg, "tanh", zero_bias=True); m_tb, _ = swap_phi(m_lif, cfg, "tanh", zero_bias=False)
XLIM = (-1.5, 1.5)
m_pair, cfg_p = load_run("results/dual/sweep_lif_symdpa", "s0_symdpa_pair", stage="dpa", device="cpu")
from src.train import project_symmetry; project_symmetry(m_pair, "pair")          # exact sigma_1 (bias included)
m_pt, cfg_pt = swap_phi(m_pair, cfg_p, "tanh", zero_bias=True)
for k, (mdl, cf, ttl, L) in enumerate([(m_lif, cfg, "B  lif, as trained", 1.5), (m_t0, cfg_t, "C  same m, n, W_in; tanh, b = 0", 4.5), (m_tb, cfg_t, "D  same; tanh, trained b", 4.5),
                                        (m_pt, cfg_pt, "E  σ₁-tied parameters; tanh, b = 0", 4.5)]):
    ax = fig.add_subplot(gs[0, k + 1]); XL = (-L, L)
    cache, spd = _flow_panel_cache(mdl, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"], torch.zeros(1, 1, cfg["input_size"]), XL, XL, 101, field_input_noise=sig, n_fp_seeds=61, slow_tol=0.06)
    _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=mdl)
    ax.axhline(0, color="w", lw=0.9, ls=(0, (5, 4)), alpha=0.9)
    F_of, _ = field_fn(mdl, sig); r = residuals(F_of, disk(21, L))
    w = [f for f, kd, t in find_wells(mdl, cf, xlim=L + 0.5, n_seeds=61, noise_sigma=sig, with_eigs=True) if str(kd).lower().startswith(("stable", "attract"))]
    print(ttl, "residual", r["s2"], "attractors", [(round(float(f[0]), 2), round(float(f[1]), 2)) for f in w], flush=True)
    ax.set_title(f"{ttl}\nσ₂ residual {r['s2']:.1e} · {len(w)} attractors", fontsize=8)
    tk = [-1, 0, 1] if L < 2 else [-4, -2, 0, 2, 4]; ax.set_xticks(tk); ax.set_yticks(tk); ax.set_xlabel("κ₀"); ax.set_ylabel("κ₁")
fig.suptitle(f"With tanh the autonomous field is odd for any m, n as long as b = 0 — and then every attractor has its partner through the origin ({rid})", fontsize=10.5)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
