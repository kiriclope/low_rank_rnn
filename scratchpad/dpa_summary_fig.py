"""Figure 1 of the note: the memory task (DPA) — its symmetries, the predicted wells for each tie (scheme),
and the trained networks: free, and tied to each element of the Klein group and to the whole group, at the
DPA checkpoint (tie held) and at the expert checkpoint (released, after GNG and Dual).
Usage: dpa_summary_fig.py <out.png>   (reads sweep_lif_recipe7 for the free arm and sweep_lif_symdpa for the ties)"""
import sys, os, glob, json, textwrap, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyArrowPatch
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from bifurcation_probe import find_wells
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
out = sys.argv[1]; XL = (-1.5, 1.5)
COLS = [("free", "results/dual/sweep_lif_recipe7", "recipe7", "free\n(no tie)"),
        ("pair", "results/dual/sweep_lif_symdpa", "symdpa_pair", "tied to σ₁ = diag(−1, +1)\n(A↔B and C↔D)"),
        ("inv", "results/dual/sweep_lif_symdpa", "symdpa_inv", "tied to σ₂ = −I\n(A↔B alone)"),
        ("test", "results/dual/sweep_lif_symdpa", "symdpa_test", "tied to σ₃ = diag(+1, −1)\n(C↔D alone)"),
        ("klein", "results/dual/sweep_lif_symdpa", "symdpa_klein", "tied to the whole group\n(4 blocks)")]
teal, plum, ember = "#0d818b", "#6b2f74", "#b0460e"; MK = ["o", "s", "D", "^", "v", "<", ">", "p"]
fig = plt.figure(figsize=(20, 13), dpi=140); gs = fig.add_gridspec(3, 5, height_ratios=[0.8, 1, 1])
def scheme(ax, wells, pairs, title, note, open_wells=()):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axhline(0, color=ember, lw=1.1, ls=(0, (4, 3))); ax.axvline(0, color="0.8", lw=0.8)
    for (x, y) in wells: ax.plot(x, y, "o", color=teal, ms=9, zorder=4)
    for (x, y) in open_wells: ax.plot(x, y, "o", ms=9, mfc="none", mec=teal, mew=1.6, zorder=4)
    for (x0, y0), (x1, y1) in pairs: ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<->", color=plum, lw=1, mutation_scale=10, linestyle=":"))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10, pad=6); pass
    ax.set_ylabel("κ₁", fontsize=9); ax.text(0.97, 0.03, "κ₀", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.4")
a = 0.95; wd = 1.1
ax0 = fig.add_subplot(gs[0, 0]); scheme(ax0, [], [], "free: no constraint", "any orbit to the right; the init is\nσ₂-exact and V-approximate"); ax0.text(0.5, 0.5, "?", transform=ax0.transAxes, ha="center", va="center", fontsize=26, color="0.6")
ax1 = fig.add_subplot(gs[0, 1]); scheme(ax1, [(a, -0.22), (-a, -0.22)], [((a, -0.22), (-a, -0.22))], "σ₁: mirror pair, one height", "memory (±a, w), w set by ⟨n₁⟩ and b;\nthe κ₁ axis is invariant: a lone decision well allowed", open_wells=[(0, wd)])
ax2 = fig.add_subplot(gs[0, 2]); scheme(ax2, [(a, 0.35), (-a, -0.35)], [((a, 0.35), (-a, -0.35)), ((0.3, wd), (-0.3, -wd))], "σ₂: antipodes", "memory (a, w), (−a, −w): straddling, or both on the line;\nno invariant axis; J₀₁, J₁₀ free; ring or rotation allowed", open_wells=[(0.3, wd), (-0.3, -wd)])
ax3 = fig.add_subplot(gs[0, 3]); scheme(ax3, [(a, 0), (-0.7, 0)], [((0, wd), (0, -wd))], "σ₃: on the line, unmatched", "the κ₀ axis is invariant: memory at (a, 0), (−a′, 0), a ≠ a′;\nor a vertical pair each (faint); J = 0; decision wells a pair", open_wells=[(0, wd), (0, -wd)])
ax4 = fig.add_subplot(gs[0, 4]); scheme(ax4, [(a, 0), (-a, 0)], [((a, 0), (-a, 0)), ((0, wd), (0, -wd))], "whole group: on the line, or a quadruple", "two wells ⇒ pinned at (±a, 0); off the line ⇒ the\nquadruple (±a, ±w) (faint); decision wells a pair; J = 0", open_wells=[(0, wd), (0, -wd)])
for x in (a, -0.7):
    for y in (0.45, -0.45): ax3.plot(x, y, "o", ms=8, mfc="none", mec=teal, mew=1.0, alpha=0.45, zorder=3)
for x in (a, -a):
    for y in (0.45, -0.45): ax4.plot(x, y, "o", ms=8, mfc="none", mec=teal, mew=1.0, alpha=0.45, zorder=3)
for r, (stage, stagelab) in enumerate([("dpa", "DPA checkpoint  (tie held)"), ("expert", "after GNG and Dual  (released)")]):
    for c, (key, sw, arm, collab) in enumerate(COLS):
        ax = fig.add_subplot(gs[r + 1, c])
        seeds = sorted(int(os.path.basename(d)[1]) for d in glob.glob(f"{sw}/s?_{arm}") if os.path.exists(f"{d}/{stage}_s{os.path.basename(d)[1]}_{arm}.pth"))
        if not seeds: ax.text(0.5, 0.5, f"{arm}\n(not run yet)", ha="center", va="center", transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); continue
        m0, cfg = load_run(sw, f"s{seeds[0]}_{arm}", stage=stage, device="cpu"); sig = sig_of(cfg)
        cache, spd = _flow_panel_cache(m0, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"], torch.zeros(1, 1, cfg["input_size"]), XL, XL, 81, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
        _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m0)
        ax.axhline(0, color="w", lw=0.9, ls=(0, (5, 4)), alpha=0.9)
        res_all = []; nwells = []; below = 0
        for s_ in seeds:
            m, cf = load_run(sw, f"s{s_}_{arm}", stage=stage, device="cpu"); F_of, P = field_fn(m, sig); res_all.append(residuals(F_of, disk(21)))
            w_ = [f for f, kd, t in find_wells(m, cf, xlim=2.5, n_seeds=41, noise_sigma=sig, with_eigs=True) if str(kd).lower().startswith(("stable", "attract"))]
            w_ = [f for f in w_ if np.linalg.norm(f) > 0.15]; nwells.append(len(w_))
            L = [f for f in w_ if f[0] < -0.3 and f[1] < 0]; R = [f for f in w_ if f[0] > 0.3 and f[1] < 0]; below += int(bool(L) and bool(R))
            if s_ != seeds[0]:
                for f in w_: ax.plot(f[0], f[1], MK[s_ % 8], color="w", ms=5.5, mew=1.1, mfc="none", zorder=6)
        med = {k: np.median([rr[k] for rr in res_all]) for k in ("s1", "s2", "s3")}
        ax.set_title(f"{collab.splitlines()[0]}\nresiduals (median of {len(seeds)}): σ₁ {med['s1']:.2f} · σ₂ {med['s2']:.2f} · σ₃ {med['s3']:.2f}\nattractors {nwells} · both memories below the line in {below}/{len(seeds)}", fontsize=8)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel("κ₀", fontsize=9); ax.set_ylabel(("κ₁\n" + stagelab) if c == 0 else "κ₁", fontsize=9)
        print(stage, arm, "seeds", seeds, "wells", nwells, "below", below, "res", {k: round(v, 3) for k, v in med.items()}, flush=True)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
