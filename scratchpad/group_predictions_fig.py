"""What the group predicts for the unit ensemble, and what it leaves open (artifact figure).
Top row (predictions): the orbit types of V in the (m0, m1) plane; the allowed population counts; the allowed
angular harmonics of the unit density. Bottom row (data): the whole-group and the free network at the DPA
checkpoint with the orbit decomposition and the angular harmonics, and the isotropic init that is V-invariant
without any clusters. Usage: group_predictions_fig.py <out.png>"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, "/home/leon/rnn"); sys.path.insert(0, "/home/leon/rnn/scratchpad")
from bifurcation_probe import load_run; from symtools import build_init
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
teal, plum, ember, gray = "#0d818b", "#6b2f74", "#b0460e", "0.45"; cols = sns.color_palette("deep")
QC = {(1, 1): cols[0], (-1, 1): cols[1], (-1, -1): cols[2], (1, -1): cols[3]}
out = sys.argv[1]
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), dpi=140)

# A — orbit types
ax = axes[0, 0]; ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axhline(0, color="0.85"); ax.axvline(0, color="0.85")
for (x, y) in [(0.8, 0.5), (-0.8, 0.5), (-0.8, -0.5), (0.8, -0.5)]: ax.plot(x, y, "o", color=QC[(int(np.sign(x)), int(np.sign(y)))], ms=11, zorder=4)
for (x, y) in [(0, 1.0), (0, -1.0)]: ax.plot(x, y, "o", mfc="none", mec=gray, mew=2, ms=11, zorder=4)
for (x, y) in [(1.05, 0), (-1.05, 0)]: ax.plot(x, y, "o", mfc="none", mec=gray, mew=2, ms=11, zorder=4, ls=":")
ax.plot(0, 0, "s", mfc="none", mec=gray, mew=1.5, ms=9, zorder=4)
ax.annotate("generic orbit: 4\n(±a, ±w)", (0.8, 0.5), (0.05, 1.05), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
ax.text(0.12, -1.12, "axis orbits: 2\n(0, ±w) or (±a, 0)", fontsize=9, color=gray)
ax.text(0.08, 0.06, "origin: 1", fontsize=9, color=gray)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("m₀"); ax.set_ylabel("m₁"); ax.set_title("A  the orbit types of V in the unit plane", loc="left", fontsize=10.5)

# B — allowed population counts
ax = axes[0, 1]
counts = range(1, 9); decomp = {1: "origin only", 2: "one axis pair", 3: "axis pair + origin", 4: "a quadruple, or two axis pairs", 5: "quadruple + origin", 6: "quadruple + axis pair", 7: "quadruple + pair + origin", 8: "two quadruples, or one + two pairs"}
for k in counts:
    needs_origin = (k % 2 == 1)
    ax.barh(k, 1, color=("0.85" if needs_origin else teal), alpha=0.35 if needs_origin else 0.8, height=0.7)
    ax.text(0.02, k, f"{k}:  {decomp[k]}" + ("   (needs an untuned population at the origin)" if needs_origin else ""), va="center", fontsize=8.6, color="0.25" if needs_origin else "k")
ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_yticks(list(counts)); ax.set_ylabel("number of populations"); ax.invert_yaxis(); ax.spines["bottom"].set_visible(False)
ax.set_title("B  allowed counts: 4a + 2b + c, c ≤ 1", loc="left", fontsize=10.5)
ax.text(0.5, 8.9, "four and six are both allowed; without an origin population the count is even", fontsize=8.5, color="0.35", ha="center", va="top")

# C — allowed angular harmonics
ax = axes[0, 2]; th = np.linspace(0, 2 * np.pi, 400)
for k, lab, c, ls in [(2, "cos 2θ  allowed", teal, "-"), (4, "cos 4θ  allowed", plum, "-"), (1, "cos θ  forbidden", "0.6", "--"), (3, "sin 2θ  forbidden", "0.6", ":")]:
    r = 1 + 0.45 * (np.cos(k * th) if k != 3 else np.sin(2 * th)); ax.plot(r * np.cos(th), r * np.sin(th), color=c, ls=ls, lw=1.8 if c != "0.6" else 1.2, label=lab)
ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.axhline(0, color="0.9", lw=0.8); ax.axvline(0, color="0.9", lw=0.8); ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=8, loc="lower left", frameon=False); ax.set_xlabel("m₀"); ax.set_ylabel("m₁")
ax.set_title("C  density of units on the circle: p(θ) = Σ a₂ⱼ cos 2jθ", loc="left", fontsize=10.5)
ax.text(0, 1.55, "θ→−θ kills the sines, θ→θ+π the odd harmonics", ha="center", fontsize=8.5, color="0.35")

# D–F — data
def harmonics(m):
    th = np.arctan2(m[:, 1], m[:, 0]); return {k: (2 * np.cos(k * th).mean(), 2 * np.sin(k * th).mean()) for k in range(1, 5)}
def panel(ax, m, title, note, orbit_colors=True):
    lat = np.abs(m[:, 0]) > 1.5; axu = (np.abs(m[:, 0]) < 0.8) & (np.abs(m[:, 1]) > 2.5)
    if orbit_colors:
        for q, c in QC.items():
            sel = lat & (np.sign(m[:, 0]) == q[0]) & (np.sign(m[:, 1]) == q[1]); ax.scatter(m[sel, 0], m[sel, 1], s=5, color=c, alpha=0.6)
        ax.scatter(m[axu, 0], m[axu, 1], s=7, color=gray, alpha=0.8); rest = ~(lat | axu); ax.scatter(m[rest, 0], m[rest, 1], s=3, color="0.8", alpha=0.6)
        note = note + f"\ngeneric orbit {lat.sum()} units · axis pair {axu.sum()} · other {rest.sum()}"
    else: ax.scatter(m[:, 0], m[:, 1], s=4, color="0.55", alpha=0.6)
    ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8); ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect("equal"); ax.set_xlabel("m₀"); ax.set_ylabel("m₁")
    h = harmonics(m); ax.set_title(title, loc="left", fontsize=10.5)
    ax.text(0.02, 0.98, note + "\nharmonics a₂ %+.2f  a₄ %+.2f   forbidden: a₁ %+.2f  b₂ %+.2f  a₃ %+.2f" % (h[2][0], h[4][0], h[1][0], h[2][1], h[3][0]), transform=ax.transAxes, va="top", fontsize=8, color="0.25")
    # polar inset: angular histogram
    ins = ax.inset_axes([0.72, 0.02, 0.27, 0.27], projection="polar"); th = np.arctan2(m[:, 1], m[:, 0]); cnt, edges = np.histogram(th, bins=36, range=(-np.pi, np.pi))
    ins.bar((edges[:-1] + edges[1:]) / 2, cnt, width=edges[1] - edges[0], color=teal, alpha=0.7); ins.set_xticks([]); ins.set_yticks([])
mk, _ = load_run("results/dual/sweep_lif_symdpa", "s0_symdpa_klein", stage="dpa"); mf, _ = load_run("results/dual/sweep_lif_recipe7", "s1_recipe7", stage="dpa"); mi, _ = build_init("results/dual/sweep_lif_symdpa", "s0_symdpa_klein")
panel(axes[1, 0], mk.m.detach().numpy(), "D  whole group tied, DPA checkpoint (s0)", "one generic orbit + one axis pair = 6")
panel(axes[1, 1], mf.m.detach().numpy(), "E  free, DPA checkpoint (s1)", "the same decomposition, approximately")
panel(axes[1, 2], mi.m.detach().numpy(), "F  the initialization: V-invariant, no clusters", "one isotropic population: allowed too", orbit_colors=False)
fig.suptitle("What the group predicts for the unit ensemble — orbit types, allowed counts, allowed harmonics — and what it leaves open", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
