"""
Figure 1 (publication main figure): model, κ-framework, tasks, training.
Rank-2 low-rank RNN learning DPA→GNG→Dual; κ-plane latent analysis.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
sys.path.insert(0, "/home/leon/rnn")
from bifurcation_probe import load_run, autonomous_ff
from src.dynamics import low_rank_numpy_params, low_rank_field_np, find_all_fixed_points, classify_fixed_points

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.8, "svg.fonttype": "none",
})

MEM = "#2c6fbb"      # memory / κ0 colour
DEC = "#c0392b"      # decision / κ1 colour
GREY = "#555555"
BOXFC = "#f4f6f9"

def box(ax, x, y, w, h, text="", fc=BOXFC, ec="#333", lw=1.0, fs=9, r=0.02, tc="#111", weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if text:
        ax.text(x + (w/2 if ha=="center" else 0.02), y + h/2, text, ha=ha, va="center",
                fontsize=fs, color=tc, zorder=3, weight=weight)

def arrow(ax, p0, p1, color=GREY, lw=1.4, style="-|>", mut=11, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=1, linestyle=ls))

def clean(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

fig = plt.figure(figsize=(15, 11.2))
gs = gridspec.GridSpec(3, 6, height_ratios=[1.05, 0.92, 0.95],
                       hspace=0.55, wspace=0.42,
                       left=0.035, right=0.978, top=0.94, bottom=0.045)

# =========================================================================
# PANEL A — architecture + equations
# =========================================================================
axA = fig.add_subplot(gs[0, 0:3]); clean(axA)
axA.text(0.0, 1.02, "a", fontsize=16, weight="bold", transform=axA.transAxes)
axA.text(0.09, 1.02, "Rank-2 low-rank RNN", fontsize=12, weight="bold", transform=axA.transAxes)

# input channels
chans = ["A", "B", "C", "D", "go", "nogo", "cue", "rwd"]
cx, cy0, ch = 0.045, 0.44, 0.052
for i, c in enumerate(chans):
    yy = cy0 + (len(chans)-1-i)*ch
    fc = MEM if c in ("A","B") else "#7aa8d6" if c in ("C","D") else DEC if c in ("go","nogo") else "#d7a94b" if c=="cue" else "#bbb"
    box(axA, cx, yy, 0.052, ch*0.86, c, fc=fc, fs=7.5, r=0.01, tc="white" if c not in ("cue","rwd") else "#111")
axA.text(cx+0.026, cy0+len(chans)*ch+0.01, r"input $\mathbf{x}$", ha="center", fontsize=8.5, weight="bold")

# recurrent network box
nx, ny, nw, nh = 0.30, 0.40, 0.30, 0.34
box(axA, nx, ny, nw, nh, "", fc="#eef3fa", ec="#33639e", lw=1.4, r=0.03)
axA.text(nx+nw/2, ny+nh+0.028, r"recurrent net  ($N$ units, rates $\mathbf{r}$)",
         ha="center", fontsize=8.7, weight="bold")
# example units
rng = np.random.default_rng(3)
for _ in range(14):
    ux, uy = nx+0.04+rng.uniform(0,nw-0.08), ny+0.05+rng.uniform(0,nh-0.10)
    axA.add_patch(Circle((ux, uy), 0.011, fc="#c9dcf2", ec="#33639e", lw=0.6, zorder=3))
# recurrence self-loop (arc above the box)
arrow(axA, (nx+nw-0.04, ny+nh+0.002), (nx+0.04, ny+nh+0.002), color="#33639e", lw=1.3, rad=-0.6, mut=10)
axA.text(nx+nw+0.055, ny+nh-0.02, r"$W_{\rm rec}$", ha="left", fontsize=8.5, color="#33639e")

# input arrow
arrow(axA, (cx+0.058, cy0+0.18), (nx-0.005, ny+nh/2), color=GREY, lw=1.6)
axA.text((cx+0.058+nx)/2-0.005, ny+nh/2+0.055, r"$A_i W_i\mathbf{x}$", ha="center", fontsize=8, color=GREY)

# readout to kappa
kxx = nx+nw+0.075
arrow(axA, (nx+nw+0.004, ny+nh/2), (kxx-0.03, ny+nh/2), color="#111", lw=1.6)
box(axA, kxx-0.028, ny+nh/2+0.012, 0.10, 0.055, r"$\kappa_0$ memory", fc="white", ec=MEM, fs=8, tc=MEM, r=0.015)
box(axA, kxx-0.028, ny+nh/2-0.07, 0.10, 0.055, r"$\kappa_1$ decision", fc="white", ec=DEC, fs=8, tc=DEC, r=0.015)
axA.text(kxx+0.022, ny+nh/2+0.11, r"${\kappa}=\frac{1}{N}\mathbf{r}^{\!\top}\mathbf{n}$",
         ha="center", fontsize=8.7, weight="bold")

# equations box
eqy = 0.005
box(axA, 0.02, eqy, 0.96, 0.345, "", fc="#fbfbfd", ec="#ccc", lw=0.9, r=0.02)
axA.text(0.5, eqy+0.305, "two-timescale update", ha="center", fontsize=8.6, weight="bold", color="#444")
eqs = [
    r"$\mathbf{h} \leftarrow e^{-\alpha_{\rm rec}}\mathbf{h} + (1-e^{-\alpha_{\rm rec}})\,W_{\rm rec}\,\mathbf{r}$",
    r"$\mathbf{r} \leftarrow e^{-\alpha}\mathbf{r} + (1-e^{-\alpha})\,\varphi\!\left(g\,(A_i W_i\mathbf{x}+\mathbf{h})\right)$",
    r"$W_{\rm rec}=\frac{1}{N}\mathbf{m}\,\mathbf{n}^{\!\top},\ \ \ \kappa=\frac{1}{N}\mathbf{r}^{\!\top}\mathbf{n},\ \ \ \mathrm{rank}\,2$",
]
for i, e in enumerate(eqs):
    axA.text(0.5, eqy+0.235-i*0.070, e, ha="center", fontsize=9.3)

# =========================================================================
# PANEL B — kappa-plane latent framework (real flow field)
# =========================================================================
axB = fig.add_subplot(gs[0, 3:6])
axB.text(-0.14, 1.02, "b", fontsize=16, weight="bold", transform=axB.transAxes)
axB.text(-0.02, 1.02, r"Latent dynamics in the $\kappa$-plane", fontsize=12, weight="bold", transform=axB.transAxes)
try:
    m, cfg = load_run("results/dual/sweep_kappa1reg", "s0_reg1")
    p = low_rank_numpy_params(m); ff = autonomous_ff(cfg)
    axg = np.linspace(-1.6, 1.6, 55); K0, K1 = np.meshgrid(axg, axg)
    F = low_rank_field_np(p, np.stack([K0, K1], -1), ff_input=ff)
    d0, d1 = F[..., 0], F[..., 1]
    axB.streamplot(axg, axg, d0, d1, color=np.hypot(d0, d1), cmap="magma", density=1.15, linewidth=0.6, arrowsize=0.7)
    fps, _ = find_all_fixed_points(m, xlim=(-1.6,1.6), ylim=(-1.6,1.6), ff_input=ff, n_seeds=41)
    st, _ = classify_fixed_points(m, fps, ff_input=ff)
    for f, k in zip(fps, st):
        if k == "attractor":
            axB.plot(*f, "o", mfc="lime", mec="k", ms=10, zorder=6)
    axB.set_xlim(-1.6, 1.6); axB.set_ylim(-1.6, 1.6)
except Exception as e:
    axB.text(0.5, 0.5, f"[flow n/a: {e}]", ha="center", transform=axB.transAxes)
axB.axhline(0, color="0.6", lw=0.7, ls=":"); axB.axvline(0, color="0.6", lw=0.7, ls=":")
axB.set_xlabel(r"$\kappa_0$   (memory)", color=MEM, fontsize=10)
axB.set_ylabel(r"$\kappa_1$   (decision)", color=DEC, fontsize=10)
axB.set_aspect("equal"); axB.tick_params(labelsize=7)
axB.text(0.02, 0.965, "lick", transform=axB.transAxes, fontsize=7.5, color=DEC, va="top")
axB.text(0.02, 0.05, "no-lick", transform=axB.transAxes, fontsize=7.5, color=DEC)
axB.text(0.5, -0.225, r"$\dot{\kappa}=\Psi(\kappa)-\kappa,\quad$"
         r"$\Psi_r=\frac{1}{N}\sum_i n_{ir}\,\varphi\!\left(g(\mathbf{m}_i\!\cdot\!\kappa+I_i)\right)$",
         fontsize=8.8, ha="center", transform=axB.transAxes)
axB.text(0.5, -0.295, r"self-gain $g\lambda_r=\frac{g}{N}\,\mathbf{n}_r^{\!\top}\mathbf{m}_r$   "
         r"($>1$: mode is bistable)", fontsize=8.4, ha="center", color="#444", transform=axB.transAxes)

# =========================================================================
# PANEL C — the three tasks
# =========================================================================
def task_timeline(ax, title, epochs, dec_trace, colorbands, note):
    clean(ax)
    ax.text(0.5, 1.02, title, fontsize=10.5, weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    y_in = 0.60; hbar = 0.16
    ax.add_line(plt.Line2D([0.04, 0.98], [y_in-0.02, y_in-0.02], color="#333", lw=1.0))
    for (x0, x1, lab, col, tc) in colorbands:
        ax.add_patch(Rectangle((x0, y_in), x1-x0, hbar, fc=col, ec="none", alpha=0.95, zorder=2))
        ax.text((x0+x1)/2, y_in+hbar/2, lab, ha="center", va="center", fontsize=7.3, color=tc, zorder=3)
    ax.text(0.02, y_in+hbar/2, "input", ha="right", fontsize=7.8, rotation=90, va="center")
    # epoch labels
    for (xt, lab) in epochs:
        ax.text(xt, y_in-0.10, lab, ha="center", fontsize=6.8, color="#666")
    # decision target trace
    yb = 0.14; amp = 0.13
    ax.add_line(plt.Line2D([0.04, 0.98], [yb, yb], color="#bbb", lw=0.7, ls=":"))
    xs = np.array([p[0] for p in dec_trace]); ys = yb + amp*np.array([p[1] for p in dec_trace])
    ax.add_line(plt.Line2D(xs, ys, color=DEC, lw=2.0))
    ax.text(0.02, yb, r"target $\kappa_1$", ha="right", fontsize=7.8, rotation=90, va="center", color=DEC)
    ax.text(0.99, yb+amp*1.05, "+1", fontsize=6.5, color=DEC, va="center")
    ax.text(0.99, yb-amp*1.05, "−1", fontsize=6.5, color=DEC, va="center")
    ax.text(0.5, -0.02, note, ha="center", fontsize=7.6, color="#444", transform=ax.transAxes)

axC1 = fig.add_subplot(gs[1, 0:2])
axC1.text(0.0, 1.30, "c", fontsize=16, weight="bold", transform=axC1.transAxes)
axC1.text(0.13, 1.30, "Tasks (learned sequentially)", fontsize=12, weight="bold", transform=axC1.transAxes)
task_timeline(axC1, "DPA", [(0.14,"sample"),(0.40,"delay"),(0.66,"test"),(0.88,"decision")],
    [(0.04,0),(0.66,0),(0.80,1),(0.98,1)],
    [(0.10,0.22,"A / B",MEM,"white"),(0.60,0.72,"C / D","#7aa8d6","white")],
    "sample A/B → test C/D → match (+1) / non-match (−1)")

axC2 = fig.add_subplot(gs[1, 2:4])
task_timeline(axC2, "Go / No-Go", [(0.14,"sample"),(0.40,"delay"),(0.63,"cue"),(0.86,"response")],
    [(0.04,0),(0.60,0),(0.74,1),(0.98,1)],
    [(0.10,0.22,"go / nogo",DEC,"white"),(0.58,0.68,"cue","#d7a94b","#111")],
    "go → lick (+1);  nogo → no-lick (held ≤0)")

axC3 = fig.add_subplot(gs[1, 4:6])
task_timeline(axC3, "Dual", [(0.10,"sample"),(0.30,"g/ng"),(0.52,"cue"),(0.78,"test")],
    [(0.04,0),(0.42,0),(0.50,1),(0.62,1),(0.70,0),(0.86,1),(0.98,1)],
    [(0.08,0.18,"A/B",MEM,"white"),(0.26,0.36,"g/ng",DEC,"white"),(0.48,0.56,"cue","#d7a94b","#111"),(0.74,0.84,"C/D","#7aa8d6","white")],
    "4 epochs [sample, g/ng, cue, test] — memory + decision interleaved")

# =========================================================================
# PANEL D — training curriculum + freezing
# =========================================================================
axD = fig.add_subplot(gs[2, 0:3]); clean(axD)
axD.text(0.0, 1.06, "d", fontsize=16, weight="bold", transform=axD.transAxes)
axD.text(0.07, 1.06, "Sequential training with selective freezing", fontsize=12, weight="bold", transform=axD.transAxes)
stages = [
    ("1. DPA", "#eaf3ea", "#4a8f4a", "all params free", "memory rank forms"),
    ("2. GNG", "#fdf3e3", "#c98a2b", "freeze $\\mathbf{m}_0,\\mathbf{n}_0$\n+ DPA/rwd inputs", "decision rank added"),
    ("3. Dual", "#f6eaf0", "#a03a6f", "freeze all inputs\n(± $\\mathbf{m}_0,\\mathbf{n}_0$; $\\kappa_1$-reg)", "retain DPA through Dual"),
]
bw, bx0, by = 0.27, 0.045, 0.30
for i, (t, fc, ec, frz, note) in enumerate(stages):
    x = bx0 + i*0.315
    box(axD, x, by, bw, 0.40, "", fc=fc, ec=ec, lw=1.5, r=0.03)
    axD.text(x+bw/2, by+0.335, t, ha="center", fontsize=10, weight="bold", color=ec)
    axD.text(x+bw/2, by+0.20, frz, ha="center", fontsize=7.8, color="#333")
    axD.text(x+bw/2, by-0.06, note, ha="center", fontsize=7.4, color="#666", style="italic")
    axD.text(x+bw/2, by+0.44, ["dpa_*.pth","naive_*.pth","expert_*.pth"][i], ha="center", fontsize=6.6, color="#999")
    if i < 2:
        arrow(axD, (x+bw+0.005, by+0.20), (x+0.315-0.003, by+0.20), color="#333", lw=1.8, mut=13)
axD.text(0.5, 0.14, "freeze = zero grads + restore values after the optimizer step (exact under AdamW)",
         ha="center", fontsize=7.4, color="#555", transform=axD.transAxes)

# =========================================================================
# PANEL E — nonlinearities
# =========================================================================
axE = fig.add_subplot(gs[2, 3:4])
axE.text(-0.28, 1.06, "e", fontsize=16, weight="bold", transform=axE.transAxes)
axE.text(-0.12, 1.06, "Nonlinearity", fontsize=11, weight="bold", transform=axE.transAxes)
xx = np.linspace(-2.5, 2.5, 200)
import scipy.special as sp
axE.plot(xx, np.tanh(xx), color=MEM, lw=2.0, label="tanh")
axE.plot(xx, sp.erf(xx), color="#2e8b57", lw=1.6, ls="--", label="erf")
axE.plot(xx, np.maximum(xx, 0), color=DEC, lw=1.6, label="relu")
axE.axhline(0, color="0.7", lw=0.6); axE.axvline(0, color="0.7", lw=0.6)
axE.set_xlim(-2.5, 2.5); axE.set_ylim(-1.4, 2.4)
axE.legend(fontsize=7, frameon=False, loc="upper left")
axE.set_xlabel("net input", fontsize=8); axE.set_ylabel(r"$\varphi$", fontsize=9)
axE.tick_params(labelsize=6.5)
axE.text(0.5, -0.34, "ring-capable = odd + saturating\n(tanh, erf)", ha="center",
         transform=axE.transAxes, fontsize=7.4, color="#444")

# =========================================================================
# PANEL F — objective / analysis
# =========================================================================
axF = fig.add_subplot(gs[2, 4:6]); clean(axF)
axF.text(-0.05, 1.06, "f", fontsize=16, weight="bold", transform=axF.transAxes)
axF.text(0.04, 1.06, "Objective & readout", fontsize=11, weight="bold", transform=axF.transAxes)
box(axF, 0.02, 0.05, 0.96, 0.86, "", fc="#fbfbfd", ec="#ccc", r=0.02)
lines = [
    (0.85, r"masked multi-target loss (per stage):", "#444", 8.4, "bold"),
    (0.73, r"$\mathcal{L}=\sum_{\rm ch}\langle \mathbb{1}_{\rm fin}\,( \kappa_{\rm ch}-t_{\rm ch})^2\rangle$", "#111", 9.2, "n"),
    (0.60, r"decision hinge (go/nogo asymmetric):", "#444", 8.2, "bold"),
    (0.49, r"go: $\mathrm{relu}(\theta-\kappa_1)^2\ \ $ nogo: $\mathrm{relu}(\kappa_1-\theta_{\rm ng})^2$", "#111", 8.6, "n"),
    (0.37, r"no-lick pressure (free windows):", "#444", 8.2, "bold"),
    (0.27, r"$w_{\rm nl}\,\langle \mathrm{relu}(\kappa_1)^2\rangle$", "#111", 8.8, "n"),
    (0.15, r"isolation reg (Dual):", "#444", 8.2, "bold"),
    (0.05, r"$w\,\mathrm{relu}\!\left(g\,\mathbf{n}_1^{\!\top}\mathbf{m}_1/N-1\right)^2$", "#111", 8.8, "n"),
]
for y, s, c, fs, w in lines:
    axF.text(0.06, y, s, fontsize=fs, color=c, weight=("bold" if w=="bold" else "normal"), ha="left")

fig.suptitle("Rank-2 low-rank RNNs learning memory and decision tasks: model and latent-dynamics framework",
             fontsize=13.5, weight="bold", y=0.985)

out = "/home/leon/rnn/results/figures/paper"
os.makedirs(out, exist_ok=True)
fig.savefig(f"{out}/fig1_model.pdf", bbox_inches="tight")
fig.savefig(f"{out}/fig1_model.png", dpi=150, bbox_inches="tight")
print("saved", f"{out}/fig1_model.png")
