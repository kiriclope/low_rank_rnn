"""ONE figure for the 2AFC check (§39): row 1 = the task symmetry and predicted wells per tie (scheme); row 2 =
simulations (seed-0 flow, attractors of seeds 1-3 overlaid, residuals, task accuracy); row 3 = where L and R
write (overlaps of the L/R columns on the two readouts, all seeds). Columns: free | -I | diag(+1,-1) | diag(-1,+1) | group.
Original docstring: ONE figure for the rule-task check: row 1 = the task's symmetry and the predicted wells (scheme) for
each tie; rows 2-3 = simulations (two-sided / one-sided objective): seed-0 flow with the attractors of all
four seeds overlaid, and the equivariance residuals. Columns: free | tied -I | tied diag(+1,-1) | tied group.
Usage: afc2_summary_fig.py <sweep_dir> <out.png>"""
import sys, os, textwrap, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyArrowPatch
sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from bifurcation_probe import find_wells
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
sw, out = sys.argv[1:3]; XL = (-1.5, 1.5)
COLS = [("free", "free\n(no tie)", None), ("inv", "tied to  −I\n(both modes flip)", "s2"), ("dec", "tied to  diag(+1, −1)\n(decision flips, memory kept)", "s3"), ("mem", "tied to  diag(−1, +1)\n(memory flips, decision kept)", "s1"), ("klein", "tied to the whole group\n(4 blocks)", None)]
ARM = {("afc2", k): f"afc2_{k}" for k in ("free", "inv", "dec", "mem", "klein")}
teal, plum, ember = "#0d818b", "#6b2f74", "#b0460e"
fig = plt.figure(figsize=(20, 13.2), dpi=140); gs = fig.add_gridspec(4, 5, height_ratios=[0.5, 0.8, 1, 0.75])
# ---------- row 0: the task symmetry, as text ----------
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
W = lambda t: textwrap.fill(t, 175)
ax.text(0.0, 1.0, W("The delayed 2AFC: stimulus L or R (2–3 s), a 3 s delay, a response cue (6–7 s), then lick left or lick right, κ₁ → −1 or +1, held to the end of the trial. Relabeling L ↔ R with the response flipped leaves the task unchanged: a Z₂.") + "\n\n"
        + W("Nothing in the objective reads κ₀, so the relabeling can act on the plane as −I (both modes flip) or as diag(+1, −1) (the decision alone flips), and their product diag(−1, +1) flips the memory mode with no relabeling: a Klein four-group of parameter symmetries. The element diag(−1, +1) WITH the L ↔ R swap is not a symmetry — it forces κ₁(L) = κ₁(R) — and tying it should fail the task.") + "\n\n"
        + W("Predictions (ring log §39a): free networks keep the whole group and put the memory on the κ₁ axis, (0, ±w), with κ₀ unused; tied to −I, antipodes at a free angle with free cross-overlaps; tied to diag(+1, −1), a mirror pair on the κ₁ axis with zero cross-overlaps; tied to diag(−1, +1) with the swap, chance performance; tied to the whole group, a pair on each axis."),
        fontsize=10, va="top", family="serif", transform=ax.transAxes, linespacing=1.35)
# ---------- row 1: scheme of predicted wells ----------
def scheme(ax, wells, pairs, title, note):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axhline(0, color=ember, lw=1.1, ls=(0, (4, 3))); ax.axvline(0, color="0.8", lw=0.8)
    for (x, y) in wells: ax.plot(x, y, "o", color=teal, ms=9, zorder=4)
    for (x0, y0), (x1, y1) in pairs: ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<->", color=plum, lw=1, mutation_scale=10, linestyle=":"))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10, pad=6); ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="0.35")
    ax.set_ylabel("κ₁", fontsize=9); ax.text(0.97, 0.03, "κ₀", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.4")
a, w = 0.35, 0.85
ax0 = fig.add_subplot(gs[1, 0]); scheme(ax0, [], [], "free: no constraint", "any orbit to the right is allowed;\npredicted: the whole-group solution")
ax0.text(0.5, 0.5, "?", transform=ax0.transAxes, ha="center", va="center", fontsize=26, color="0.6")
ax1 = fig.add_subplot(gs[1, 1]); scheme(ax1, [(a, w), (-a, -w)], [((a, w), (-a, -w))], "tied to −I: antipodes", "(a, w) and (−a, −w), angle free —\nno invariant axis; J₀₁, J₁₀ free")
th = np.linspace(0.35, 1.2, 30); ax1.plot(0.92 * np.cos(th), 0.92 * np.sin(th), color="0.6", lw=1, ls=":"); ax1.plot(-0.92 * np.cos(th), -0.92 * np.sin(th), color="0.6", lw=1, ls=":")
ax2 = fig.add_subplot(gs[1, 2]); scheme(ax2, [(0, w), (0, -w)], [((0, w), (0, -w))], "tied to diag(+1, −1): mirror pair", "(a, w) and (a, −w) across the line; the κ₁ axis is\ninvariant up to ⟨n₀⟩ and b, so a ≈ 0; J₀₁ = J₁₀ = 0")
ax3 = fig.add_subplot(gs[1, 3]); scheme(ax3, [(0.85, 0), (-0.85, 0)], [((0.85, 0), (-0.85, 0))], "tied to diag(−1, +1): not a symmetry", "κ₁(L) = κ₁(R) is forced: the task\ncannot be answered — chance predicted")
ax3.text(0.5, 0.78, "✗", transform=ax3.transAxes, ha="center", va="center", fontsize=22, color=ember)
ax4 = fig.add_subplot(gs[1, 4]); scheme(ax4, [(0, w), (0, -w)], [((0, w), (0, -w))], "tied to the group: pairs on both axes", "(0, ±w) is the memory; both axes are exactly\ninvariant, and the unused mode keeps (±a, 0)")
for x in (0.85, -0.85): ax4.plot(x, 0, "o", ms=9, mfc="none", mec=teal, mew=1.6, zorder=4)
# ---------- row 2: simulations ----------
import json
MK = ["o", "s", "D", "^"]
accs = {}
for l in open(f"{sw}/results.jsonl"):
    rj = json.loads(l); accs[rj["run_id"]] = rj["accuracy"]["after_gng"]["gng"]
ov_rows = {}
for c, (key, collab, tied_el) in enumerate(COLS):
    arm = ARM[("afc2", key)]; ax = fig.add_subplot(gs[2, c])
    seeds = sorted(int(os.path.basename(d)[1]) for d in __import__("glob").glob(f"{sw}/s?_{arm}") if os.path.exists(f"{d}/naive_s{os.path.basename(d)[1]}_{arm}.pth"))
    if not seeds: ax.text(0.5, 0.5, f"{arm}\n(not run yet)", ha="center", va="center", transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); continue
    m0, cfg = load_run(sw, f"s{seeds[0]}_{arm}", stage="naive", device="cpu"); sig = sig_of(cfg)
    cache, spd = _flow_panel_cache(m0, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"], torch.zeros(1, 1, cfg["input_size"]), XL, XL, 81, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
    _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m0)
    ax.axhline(0, color="w", lw=0.9, ls=(0, (5, 4)), alpha=0.9)
    res_all = []; nwells = []; ovs = []
    for s_ in seeds:
        m, cf = load_run(sw, f"s{s_}_{arm}", stage="naive", device="cpu"); F_of, P = field_fn(m, sig); rr = residuals(F_of, disk(21)); res_all.append(rr)
        Nv = np.asarray(P["Nvec"]); Wi = np.asarray(P["Wi"]); N = Nv.shape[0]; ov = (Nv.T @ Wi) / N; ovs.append([ov[0, 4], ov[0, 5], ov[1, 4], ov[1, 5]])
        w_ = [f for f, kd, t in find_wells(m, cf, xlim=2.5, n_seeds=41, noise_sigma=sig, with_eigs=True) if str(kd).lower().startswith(("stable", "attract"))]
        w_ = [f for f in w_ if np.linalg.norm(f) > 0.15]; nwells.append(len(w_))
        if s_ != seeds[0]:
            for f in w_: ax.plot(f[0], f[1], MK[s_ % 4], color="w", ms=6, mew=1.2, mfc="none", zorder=6)
    ov_rows[key] = np.array(ovs)
    med = {k: np.median([rr[k] for rr in res_all]) for k in ("s1", "s2", "s3")}
    acc = [accs[f"s{s_}_{arm}"] for s_ in seeds if f"s{s_}_{arm}" in accs]
    ax.set_title(f"{collab.splitlines()[0]}\nresiduals (median of {len(seeds)}): −I {med['s2']:.2f} · dec {med['s3']:.2f} · mem {med['s1']:.2f}\nattractors per seed {nwells} · task {min(acc):.3f}–{max(acc):.3f}", fontsize=8.2)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel("κ₀", fontsize=9); ax.set_ylabel("κ₁", fontsize=9)
    print(arm, "seeds", seeds, "wells", nwells, "res", {k: round(v, 3) for k, v in med.items()}, "acc", acc, flush=True)
# ---------- row 3: where L and R write ----------
for c, (key, collab, tied_el) in enumerate(COLS):
    ax = fig.add_subplot(gs[3, c])
    if key not in ov_rows: ax.axis("off"); continue
    A = ov_rows[key]; x = np.arange(4); labels = ["n₀·w_R", "n₀·w_L", "n₁·w_R", "n₁·w_L"]
    for s_ in range(A.shape[0]): ax.plot(x + (s_ - 1.5) * 0.12, A[s_], MK[s_ % 4], color=plum, ms=5, mfc="none" if s_ else plum)
    ax.axhline(0, color="0.8", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(-3.2, 3.2)
    ax.set_title("where the stimuli write (per seed)", fontsize=8.5)
    if c == 0: ax.set_ylabel("overlap / N", fontsize=9)
fig.text(0.5, 0.005, "Row 2: seed 0's input-noise-averaged field (○ attractors, × saddles, △ repellers); white □ ◇ △: the attractors of seeds 1–3 of the same arm; dashed: κ₁ = 0. Residual = ‖F(Dκ) − D F(κ)‖/‖F‖ on the disk |κ| ≤ 1.5 for D = −I, diag(+1,−1) (dec), diag(−1,+1) (mem). Row 3: the overlap of the L and R input columns with the two readouts, one marker per seed.",
         ha="center", fontsize=8.5, color="0.3")
fig.suptitle("The delayed 2AFC: symmetries, predicted wells, and what free and tied networks do", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0.012, 1, 0.985]); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
