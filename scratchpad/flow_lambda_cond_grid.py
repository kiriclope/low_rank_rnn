"""One figure: ROWS = arms (λ values), COLUMNS = input conditions
(Autonomous, A, B, Go, NoGo, C, D by default; override with COND_SPEC="name=dims[@value];...").
Same conventions as the trained-net fp_stages panels: input-noise-averaged field, magma speed map,
streamlines, ● attractor / × saddle / △ repeller, orange ring = slow attractor (protocol rule 12).
Usage: flow_lambda_cond_grid.py <sweep_dir> <stage> <seed> <out.png> <tag> [<tag> ...]"""
import sys, os, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bifurcation_probe import load_run, run_dt_alpha, find_wells
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from src.flow_field import low_rank_numpy_params
sw, stage, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; tags = sys.argv[5:]
XLIM = (-1.5, 1.5); NGRID = 121; NFP = 41
spec = os.environ.get("COND_SPEC", "Autonomous=;A=0;B=1;Go=4;NoGo=5;C=2;D=3")
COLS = []
for part in spec.split(";"):
    nm, _, rest = part.partition("="); dims, _, val = rest.partition("@")
    COLS.append((nm, [int(d) for d in dims.split(",") if d != ""] or None, float(val) if val else None))
nets = {}
for t in tags:
    m, cfg = load_run(sw, f"s{seed}_{t}", stage=stage, device="cpu"); m.eval()
    _, alpha, _ = run_dt_alpha(cfg); nets[t] = (m, cfg, cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha)))
def cache_for(t, dims, n_grid=NGRID, n_fp=NFP, value=None):
    m, cfg, sig = nets[t]
    sp = dict(name=("Autonomous" if dims is None else "X"), dims=dims,
              value=float(value if value is not None else cfg.get("input_scale", 1.0)), conds=[])
    return _flow_panel_cache(m, sp, cfg["input_size"], torch.zeros(1, 1, cfg["input_size"]), XLIM, XLIM, n_grid,
                             field_input_noise=sig, n_fp_seeds=n_fp, slow_tol=0.06)
vmax = float(np.percentile(np.concatenate([cache_for(t, d, 61, 9, v)[1] for t in tags for _, d, v in COLS]), 98))
fig, axes = plt.subplots(len(tags), len(COLS), figsize=(2.55 * len(COLS), 2.5 * len(tags)), dpi=130, squeeze=False)
for i, t in enumerate(tags):
    m, cfg, sig = nets[t]
    P = low_rank_numpy_params(m); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); N = M.shape[0]
    J = cfg["gain"] * (Nv.T @ M) / N
    w = [tt for f, k, tt in find_wells(m, cfg, xlim=2.5, n_seeds=41, noise_sigma=sig, with_eigs=True)
         if str(k).lower().startswith(("stable", "attract"))]
    for j, (cname, dims, val) in enumerate(COLS):
        ax = axes[i][j]
        cache, _ = cache_for(t, dims, value=val)
        hm = _render_flow_panel(ax, cache, speed_vmax=vmax, sim_scattered=False, kappa_traj=None,
                                cond_idx={}, colors={}, xlim=XLIM, ylim=XLIM, model=m)
        if i == 0: ax.set_title(cname, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"λ init {cfg['memory_lambda']:g}\nJ {J[0,0]:.1f}/{J[1,1]:.1f}   κ₁", fontsize=7.5)
            ax.text(0.02, 0.02, f"{len(w)} attr, τ_slow max {max([x[0] for x in w], default=float('nan')):.1f}s",
                    transform=ax.transAxes, fontsize=5.5, color="w", va="bottom")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=6)
        if i == len(tags) - 1: ax.set_xlabel("κ₀", fontsize=8)
fig.colorbar(hm, ax=[a for r in axes for a in r], shrink=0.5, label="‖Ψ(κ) − κ‖")
fig.suptitle(f"{os.path.basename(sw)} — seed {seed}, {stage.upper()} checkpoint: flow per input condition "
             f"(rows = init λ; input-noise-averaged field; orange ring = slow attractor)", fontsize=10)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight"); print("saved", out)
