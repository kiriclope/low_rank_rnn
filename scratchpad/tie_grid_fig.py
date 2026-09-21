"""Rows = conditions (free / a tie / a control), columns = the predicted wells (scheme) then one flow panel per
seed. Usage: tie_grid_fig.py <out.png> <stage> <title> <row>...   row = label|sweep|arm|scheme|seeds
scheme in {free, untied, pair, inv, test, klein, gng_inv, gng_dec, gng_klein, none}."""
import sys, os, glob, json, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyArrowPatch
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from bifurcation_probe import find_wells
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
out, stage, title = sys.argv[1:4]; rows = [r.split("|") for r in sys.argv[4:]]; XL = (-1.5, 1.5)
teal, plum, ember = "#0d818b", "#6b2f74", "#b0460e"
def scheme(ax, kind):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axhline(0, color=ember, lw=1.1, ls=(0, (4, 3))); ax.axvline(0, color="0.8", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.text(0.97, 0.03, "κ₀", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.4"); ax.set_ylabel("κ₁", fontsize=8)
    F = lambda x, y: ax.plot(x, y, "o", color=teal, ms=8, zorder=4); O = lambda x, y, al=1.0: ax.plot(x, y, "o", ms=8, mfc="none", mec=teal, mew=1.4, alpha=al, zorder=4)
    A = lambda p, q: ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<->", color=plum, lw=0.9, mutation_scale=9, linestyle=":"))
    a, w = 0.95, 1.1
    if kind in ("free", "untied"): ax.text(0.5, 0.5, "?", transform=ax.transAxes, ha="center", va="center", fontsize=24, color="0.6"); note = "no constraint" if kind == "free" else "same init statistics, no tie"
    elif kind == "pair": F(a, -0.22); F(-a, -0.22); A((a, -0.22), (-a, -0.22)); O(0, w); note = "σ₁: mirror pair, one height"
    elif kind == "inv": F(a, 0.35); F(-a, -0.35); A((a, 0.35), (-a, -0.35)); O(0, w); O(0, -w); A((0, w), (0, -w)); note = "σ₂: antipodes"
    elif kind == "test": F(a, 0); F(-0.7, 0); O(0, w); O(0, -w); A((0, w), (0, -w)); note = "σ₃: on the line, unmatched"
    elif kind == "klein":
        F(a, 0); F(-a, 0); A((a, 0), (-a, 0)); O(0, w); O(0, -w); A((0, w), (0, -w))
        for x in (a, -a):
            for y in (0.45, -0.45): O(x, y, 0.45)
        note = "whole group: pinned, or a quadruple"
    elif kind == "gng_inv": F(0.35, 0.85); F(-0.35, -0.85); A((0.35, 0.85), (-0.35, -0.85)); note = "−I: antipodes, angle free"
    elif kind == "gng_dec": F(0, 0.85); F(0, -0.85); A((0, 0.85), (0, -0.85)); note = "diag(+1,−1): mirror pair on the κ₁ axis"
    elif kind == "gng_klein": F(0, 0.85); F(0, -0.85); A((0, 0.85), (0, -0.85)); O(0.85, 0); O(-0.85, 0); note = "whole group: pairs on both axes"
    else: note = ""
    ax.set_title("predicted", fontsize=8.5, color="0.35"); ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
EL = {"pair": "s1", "inv": "s2", "test": "s3", "gng_inv": "s2", "gng_dec": "s3"}
nseed = max(len(r[4].split(",")) for r in rows)
fig, axes = plt.subplots(len(rows), nseed + 1, figsize=(3.0 * (nseed + 1) + 0.6, 3.05 * len(rows)), dpi=140, squeeze=False, gridspec_kw=dict(width_ratios=[0.9] + [1] * nseed))
for r, (label, sw, arm, kind, seeds) in enumerate(rows):
    scheme(axes[r][0], kind); axes[r][0].text(-0.32, 0.5, label, transform=axes[r][0].transAxes, rotation=90, ha="center", va="center", fontsize=9.5)
    seeds = [int(x) for x in seeds.split(",")]
    for c, s_ in enumerate(seeds):
        ax = axes[r][c + 1]; rid = f"s{s_}_{arm}"
        if not os.path.exists(f"{sw}/{rid}/{stage}_{rid}.pth"):
            ax.text(0.5, 0.5, f"{arm}\nseed {s_}\n(not run yet)", ha="center", va="center", transform=ax.transAxes, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); continue
        m, cfg = load_run(sw, rid, stage=stage, device="cpu"); sig = sig_of(cfg)
        cache, spd = _flow_panel_cache(m, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"], torch.zeros(1, 1, cfg["input_size"]), XL, XL, 81, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
        _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m)
        ax.axhline(0, color="w", lw=0.9, ls=(0, (5, 4)), alpha=0.9)
        F_of, P = field_fn(m, sig); rr = residuals(F_of, disk(21))
        w_ = [f for f, kd, t in find_wells(m, cfg, xlim=2.5, n_seeds=41, noise_sigma=sig, with_eigs=True) if str(kd).lower().startswith(("stable", "attract"))]
        w_ = [f for f in w_ if np.linalg.norm(f) > 0.15]
        rtxt = (f"residual {rr[EL[kind]]:.2f}" if kind in EL else f"σ₁ {rr['s1']:.2f} σ₂ {rr['s2']:.2f} σ₃ {rr['s3']:.2f}")
        ax.set_title(f"seed {s_} · {len(w_)} attractors · {rtxt}", fontsize=8)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel("κ₀" if r == len(rows) - 1 else "", fontsize=8); ax.set_ylabel("κ₁" if c == 0 else "", fontsize=8)
        print(rid, len(w_), {k: round(v, 3) for k, v in rr.items()}, [(round(float(f[0]), 2), round(float(f[1]), 2)) for f in w_], flush=True)
    for c in range(len(seeds) + 1, nseed + 1): axes[r][c].axis("off")
fig.suptitle(title, fontsize=11, y=0.995); fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
