"""Autonomous flow of TRAINED nets as a grid (cols = arm, rows = seed), same conventions as the
trained-net fp_stages panels and as scratchpad/init_flow_grid.py — so an init grid and a trained grid
can be put side by side. Slow/ring-remnant attractors get the orange ring (protocol rule 12).
Usage: trained_flow_grid.py <sweep_dir> <stage> <out.png> <tag1> <tag2> ...   (tags = the arm suffixes)"""
import sys, os, math, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bifurcation_probe import load_run, run_dt_alpha, find_wells
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from src.flow_field import low_rank_numpy_params
sw, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]; tags = sys.argv[4:]
seeds = sorted({int(os.path.basename(d)[1]) for d in __import__("glob").glob(f"{sw}/s*_{tags[0]}")})
XLIM = (-1.5, 1.5); NGRID = 121; NFP = 41
def load(seed, tag):
    m, cfg = load_run(sw, f"s{seed}_{tag}", stage=stage, device="cpu"); m.eval()
    dt, alpha, _ = run_dt_alpha(cfg); return m, cfg, cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha))
cells = {}
for t in tags:
    for s in seeds:
        try: cells[(s, t)] = load(s, t)
        except Exception as e: print(f"s{s}_{t}: {str(e)[:60]}")
sp = []
for (s, t), (m, cfg, sig) in cells.items():
    _, spd = _flow_panel_cache(m, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"],
                               torch.zeros(1, 1, cfg["input_size"]), XLIM, XLIM, 61, field_input_noise=sig, n_fp_seeds=9)
    sp.append(spd)
vmax = float(np.percentile(np.concatenate(sp), 98))
fig, axes = plt.subplots(len(seeds), len(tags), figsize=(2.9 * len(tags), 2.8 * len(seeds)), dpi=130, squeeze=False)
for i, s in enumerate(seeds):
    for j, t in enumerate(tags):
        ax = axes[i][j]
        if (s, t) not in cells: ax.axis("off"); continue
        m, cfg, sig = cells[(s, t)]
        cache, _ = _flow_panel_cache(m, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"],
                                     torch.zeros(1, 1, cfg["input_size"]), XLIM, XLIM, NGRID,
                                     field_input_noise=sig, n_fp_seeds=NFP, slow_tol=0.06)
        hm = _render_flow_panel(ax, cache, speed_vmax=vmax, sim_scattered=False, kappa_traj=None,
                                cond_idx={}, colors={}, xlim=XLIM, ylim=XLIM, model=m)
        P = low_rank_numpy_params(m); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); N = M.shape[0]
        J = cfg["gain"] * (Nv.T @ M) / N; sm, sn = M.std(0), Nv.std(0)
        w = [(f, tt) for f, k, tt in find_wells(m, cfg, xlim=2.5, n_seeds=41, noise_sigma=sig, with_eigs=True)
             if str(k).lower().startswith(("stable", "attract"))]
        ts = max([t_[0] for _, t_ in w], default=float("nan"))
        head = (t + "\n") if i == 0 else ""
        ax.set_title(f"{head}λ init {cfg['memory_lambda']:g}   J {J[0,0]:.1f}/{J[1,1]:.1f}", fontsize=8)
        if not os.environ.get("NOANNOT"):
            ax.text(0.03, 0.03, f"σm {sm[0]:.1f}/{sm[1]:.1f}  σn {sn[0]:.1f}/{sn[1]:.1f}\nτ_slow max {ts:.1f}s  ({len(w)} attr)",
                    transform=ax.transAxes, fontsize=5.5, color="w", va="bottom", linespacing=1.9)
        if os.environ.get("LICKLINE"):
            import matplotlib.patheffects as pe
            ax.axhline(0, color="w", lw=1.1, ls=(0, (5, 4)), alpha=1.0, zorder=6,
                       path_effects=[pe.Stroke(linewidth=2.6, foreground="k", alpha=0.65), pe.Normal()])
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=6)
        if j == 0: ax.set_ylabel(f"seed {s}\nκ₁", fontsize=8)
        if i == len(seeds) - 1: ax.set_xlabel("κ₀", fontsize=8)
fig.colorbar(hm, ax=[a for r in axes for a in r], shrink=0.6, label="‖Ψ(κ) − κ‖")
fig.suptitle(f"{os.path.basename(sw)} — autonomous flow at the {stage.upper()} checkpoint "
             f"(input-noise-averaged field; orange ring = slow attractor)", fontsize=9)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
