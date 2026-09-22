"""Figures 1 and 2 of the note (two outputs): 1 = the task, the group, its actions, the wells and the populations it predicts;
2 = the overlaps, the inputs and the remaining predictions. Originally: Figure 1 of the note: the Klein four-group of the DPA task and every prediction it makes, per tie —
the task and its relabelings, the group, its actions on the plane and on the units; then per tie: the
autonomous-flow schemes, the unit-ensemble orbit schemes drawn as population clouds, the overlap table, the
character table of the input combinations, the predicted (m, input) scatters, and a table of the remaining
predictions. No data. Usage: theory_fig.py <out.png>"""
import sys, textwrap, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
teal, plum, ember, gray = "#0d818b", "#6b2f74", "#b0460e", "0.45"; cols = sns.color_palette("deep")
QC = {(1, 1): cols[0], (-1, 1): cols[1], (-1, -1): cols[2], (1, -1): cols[3]}
out = sys.argv[1]; rng = np.random.default_rng(3)
out2 = sys.argv[2] if len(sys.argv) > 2 else out.replace(".png", "_predictions.png")
fig1 = plt.figure(figsize=(17, 15), dpi=130); gs1 = fig1.add_gridspec(3, 4, height_ratios=[1.3, 1, 1], hspace=0.6, wspace=0.28, top=0.95, bottom=0.04, left=0.05, right=0.985)
fig2 = plt.figure(figsize=(17, 24), dpi=130); gs2 = fig2.add_gridspec(3, 4, height_ratios=[3.0, 2.3, 2.5], hspace=0.45, wspace=0.28, top=0.945, bottom=0.02, left=0.05, right=0.985)
def panel_at(r, c): return fig1.add_subplot(gs1[r, c]) if r < 3 else fig2.add_subplot(gs2[r - 3, c])
TIES = [("σ₁ = diag(−1, +1)", "A↔B and C↔D, response kept"), ("σ₂ = −I", "A↔B alone, response flipped"), ("σ₃ = diag(+1, −1)", "C↔D alone, response flipped"), ("the whole group V", "all three at once")]
# characters as (χ(σ₁), χ(σ₃)); χ(σ₂) is the product. Elements of each tie's subgroup as index into (σ₁, σ₃, σ₂).
CH = {"m₀": (-1, 1), "m₁": (1, -1), "b": (1, 1), "A−B": (-1, 1), "A+B": (1, 1), "C−D": (-1, -1), "C+D": (1, 1), "go": (1, 1)}
def chi(c): return (c[0], c[1], c[0] * c[1])
SUB = {0: [0], 1: [2], 2: [1], 3: [0, 1, 2]}
def zero(k, a, b): return any(chi(CH[a])[g] != chi(CH[b])[g] for g in SUB[k])
def images(k, a, b):  # sign pairs (sx, sy) of the images of a point (x=a-like, y=b-like) under the tie's group
    S = {(1, 1)}
    for g in SUB[k]: S |= {(chi(CH[a])[g] * sx, chi(CH[b])[g] * sy) for (sx, sy) in list(S)}
    changed = True
    while changed:
        changed = False
        for g in SUB[k]:
            new = {(chi(CH[a])[g] * sx, chi(CH[b])[g] * sy) for (sx, sy) in S}
            if not new <= S: S |= new; changed = True
    return sorted(S)

# ───────── row 0: the task, the group, the actions ─────────
ax = panel_at(0, slice(0, 2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0, 9.6, "A  The task and its relabelings", fontsize=12, fontweight="bold", va="top")
for i, s_ in enumerate(["A", "B"]):
    for j, t_ in enumerate(["C", "D"]):
        r = (i == j); x, y = 1.9 + j * 1.5, 7.0 - i * 1.3
        ax.add_patch(Rectangle((x - 0.7, y - 0.55), 1.4, 1.1, fc=(teal if r else "0.92"), ec="0.5", lw=0.8))
        ax.text(x, y, "lick" if r else "no lick", ha="center", va="center", fontsize=10, color="w" if r else "0.3")
    ax.text(0.35, 7.0 - i * 1.3, f"sample {s_}", ha="center", va="center", fontsize=9.5)
for j, t_ in enumerate(["C", "D"]): ax.text(1.9 + j * 1.5, 8.05, f"test {t_}", ha="center", va="center", fontsize=10)
ax.text(2.4, 4.5, "r = ¬(s ⊕ t): lick iff the pair matches", ha="center", fontsize=9.5, color="0.3")
ax.text(5.2, 8.3, "three relabelings leave the table unchanged", fontsize=10, va="top")
for k, (name, what, col) in enumerate([("σ₁", "A↔B and C↔D, response kept", teal), ("σ₂", "A↔B alone, response flipped", plum), ("σ₃", "C↔D alone, response flipped", ember)]):
    ax.text(5.2, 7.5 - k * 0.8, name, fontsize=11, color=col, fontweight="bold", va="top"); ax.text(5.9, 7.5 - k * 0.8, what, fontsize=10, va="top")
ax.text(5.2, 4.9, "with the identity: the Klein four-group V = Z₂ × Z₂,\nevery element its own inverse, σ₁σ₂ = σ₃", fontsize=10, va="top")
names = ["e", "σ₁", "σ₂", "σ₃"]; ax.text(5.2, 3.2, "·", fontsize=10)
for i in range(4):
    ax.text(5.9 + i * 0.75, 3.2, names[i], fontsize=9.5, ha="center", color="0.3"); ax.text(5.2, 2.45 - i * 0.6, names[i], fontsize=9.5, color="0.3")
    for j in range(4): ax.text(5.9 + j * 0.75, 2.45 - i * 0.6, names[i ^ j], fontsize=9.5, ha="center")
ax.text(0.2, 3.6, "a symmetry of the objective: the loss is\nthe same function of the trial after any\nrelabeling, so gradient descent has no\nreason to prefer one element over another", fontsize=9.5, va="top", color="0.3")

ax = panel_at(0, slice(2, 4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0, 9.6, "B  How the group acts", fontsize=12, fontweight="bold", va="top")
ax.text(0, 8.7, "on the plane (κ₀ memory, κ₁ decision)", fontsize=10.5, va="top")
for k, (name, D, col) in enumerate([("σ₁: (−κ₀, +κ₁)", np.diag([-1, 1]), teal), ("σ₂: (−κ₀, −κ₁)", -np.eye(2), plum), ("σ₃: (+κ₀, −κ₁)", np.diag([1, -1]), ember)]):
    cx, cy = 1.0 + k * 3.1, 6.6; s = 0.9
    ax.plot([cx - s, cx + s], [cy, cy], color="0.85", lw=0.8); ax.plot([cx, cx], [cy - s, cy + s], color="0.85", lw=0.8)
    p = np.array([0.55, 0.35]); q = D @ p
    ax.plot(cx + p[0], cy + p[1], "o", color="k", ms=6); ax.plot(cx + q[0], cy + q[1], "o", mfc="none", mec=col, mew=1.8, ms=7)
    ax.add_patch(FancyArrowPatch((cx + p[0], cy + p[1]), (cx + q[0], cy + q[1]), arrowstyle="->", color=col, lw=1, mutation_scale=10, linestyle=":", shrinkA=4, shrinkB=4))
    ax.text(cx, cy - 1.25, name, ha="center", fontsize=9.5, color=col)
ax.text(0, 4.7, "on the units: a permutation ρ of the N units with", fontsize=10.5, va="top")
ax.text(0.3, 4.0, "ρ m = m D      ρ n = n D      ρ W_in = W_in S      ρ b = b", fontsize=10.5, va="top", family="serif")
ax.text(0, 3.1, "for the whole group: four blocks (a, b), m₀, n₀ ∝ (−1)ᵃ, m₁, n₁ ∝ (−1)ᵇ,\nA/B columns exchanged by a, C/D columns by b, go, nogo, cue and the bias shared", fontsize=9.5, va="top", color="0.3")
for k in range(4):
    a, b = k >> 1, k & 1; x = 0.4 + k * 2.4
    ax.add_patch(Rectangle((x, 0.1), 2.0, 1.15, fc=QC[((-1) ** a, (-1) ** b)], alpha=0.25, ec="0.5", lw=0.6))
    ax.text(x + 1.0, 0.68, f"block ({a},{b})\nm₀ {'+' if a == 0 else '−'}μ₀  m₁ {'+' if b == 0 else '−'}μ₁", ha="center", va="center", fontsize=8.5)
ax.text(0, 2.1, "then F(Dκ; Sx) = D F(κ; x), exactly, for every input and along whole trials", fontsize=10, va="top", fontweight="bold")

def plane(ax, lab_x="κ₀", lab_y="κ₁", lick=True, lim=1.4):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(0, color="0.85", lw=0.8); ax.axhline(0, color=(ember if lick else "0.85"), lw=(1.1 if lick else 0.8), ls=((0, (4, 3)) if lick else "-"))
    ax.set_xlabel(lab_x, fontsize=9, labelpad=1); ax.set_ylabel(lab_y, fontsize=9)
def dot(ax, x, y, c=teal, open_=False, ms=9, **kw): ax.plot(x, y, "o", ms=ms, **(dict(mfc="none", mec=c, mew=1.6) if open_ else dict(color=c)), zorder=4, **kw)
def link(ax, p, q, c=plum): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<->", color=c, lw=0.9, mutation_scale=9, linestyle=":", shrinkA=5, shrinkB=5))
a, wd = 0.95, 1.05

# ───────── row 1 (C): autonomous flow ─────────
axs = [panel_at(1, c) for c in range(4)]
for ax, (t, sub) in zip(axs, TIES): plane(ax); ax.set_title(t + "\n" + sub, fontsize=9.5, pad=6)
dot(axs[0], a, -0.22); dot(axs[0], -a, -0.22); link(axs[0], (a, -0.22), (-a, -0.22));
for x, lab in ((a, "A"), (-a, "B = σ₁A")): axs[0].text(x, -0.42, lab, ha="center", va="top", fontsize=8.5, color=teal)
dot(axs[0], 0, wd, open_=True)
axs[0].text(0.5, -0.2, "mirror pair at one height w (free);\nlone decision well allowed on the κ₁ axis", transform=axs[0].transAxes, ha="center", va="top", fontsize=8, color="0.35")
dot(axs[1], a, 0.35); dot(axs[1], -a, -0.35); link(axs[1], (a, 0.35), (-a, -0.35));
axs[1].text(a, 0.15, "A", ha="center", va="top", fontsize=8.5, color=teal); axs[1].text(-a, -0.55, "B = −A", ha="center", va="top", fontsize=8.5, color=teal)
dot(axs[1], 0.3, wd, open_=True); dot(axs[1], -0.3, -wd, open_=True); link(axs[1], (0.3, wd), (-0.3, -wd))
axs[1].text(0.5, -0.2, "antipodes, straddling or both on the line;\nno invariant axis; rotation allowed", transform=axs[1].transAxes, ha="center", va="top", fontsize=8, color="0.35")
dot(axs[2], a, 0); dot(axs[2], -0.62, 0); dot(axs[2], 0, wd, open_=True); dot(axs[2], 0, -wd, open_=True); link(axs[2], (0, wd), (0, -wd))
# a and a′ are not related by σ₃: two independent distances from the origin, drawn as separate brackets
for x, lab, yy in ((a, "a", 0.28), (-0.62, "a′", 0.28)):
    axs[2].add_patch(FancyArrowPatch((0, yy), (x, yy), arrowstyle="|-|", color="0.3", lw=0.9, mutation_scale=4, shrinkA=0, shrinkB=0)); axs[2].text(x / 2, yy + 0.08, lab, ha="center", va="bottom", fontsize=9, color="0.3")
axs[2].text(0.03, 0.975, "a ≠ a′ allowed: no element maps A to B", transform=axs[2].transAxes, ha="left", va="top", fontsize=7.8, color="0.3")
for x, lab in ((a, "A"), (-0.62, "B")): axs[2].text(x, -0.2, lab, ha="center", va="top", fontsize=8.5, color=teal)
for x in (a, -0.62):
    for y in (0.45, -0.45): dot(axs[2], x, y, open_=True, ms=8, alpha=0.45)
axs[2].text(0.5, -0.2, "mean memory on the κ₀ axis, at unrelated distances;\nsplits to (a, ±w) if the axis is a saddle", transform=axs[2].transAxes, ha="center", va="top", fontsize=8, color="0.35")
dot(axs[3], a, 0); dot(axs[3], -a, 0); link(axs[3], (a, 0), (-a, 0));
for x, lab in ((a, "A"), (-a, "B = σ₁A")): axs[3].text(x, -0.2, lab, ha="center", va="top", fontsize=8.5, color=teal)
dot(axs[3], 0, wd, open_=True); dot(axs[3], 0, -wd, open_=True); link(axs[3], (0, wd), (0, -wd))
for x in (a, -a):
    for y in (0.45, -0.45): dot(axs[3], x, y, open_=True, ms=8, alpha=0.45)
axs[3].text(0.5, -0.2, "pair pinned at (±a, 0), or the quadruple;\nboth axes invariant; origin the rest state", transform=axs[3].transAxes, ha="center", va="top", fontsize=8, color="0.35")
axs[0].text(-0.15, 1.42, "C  the autonomous flow: the attractor set is closed under the tie's action", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 2 (D): the unit ensemble as population clouds ─────────
axs = [panel_at(2, c) for c in range(4)]
base = np.array([0.8, 0.5]) + rng.normal(size=(70, 2)) * [0.13, 0.09]
axis_cloud = np.array([0.0, 1.05]) + rng.normal(size=(30, 2)) * [0.05, 0.12]
notes = ["every cluster has its mirror image across the m₁ axis;\nan axis cluster is its own image; ⟨m₀⟩ = ⟨n₀⟩ = 0",
         "every cluster has its antipode, the axis pair too;\n⟨m⟩ = ⟨n⟩ = 0; the lattice may tilt (J free)",
         "every cluster has its mirror image across the m₀ axis;\nthe axis pair is exchanged; ⟨m₁⟩ = ⟨n₁⟩ = 0",
         "every cluster has three images; populations come as\n4a + 2b + c; density on the circle ∝ Σ a₂ⱼ cos 2jθ"]
for k, (ax, note) in enumerate(zip(axs, notes)):
    plane(ax, "m₀ (and n₀)", "m₁ (and n₁)", lick=False)
    for (sx, sy) in images(k, "m₀", "m₁"):
        pts = base * [sx, sy]; ax.scatter(pts[:, 0], pts[:, 1], s=9, color=QC[(sx, sy)], alpha=0.75, edgecolors="none", zorder=3)
        if (sx, sy) != (1, 1): ax.annotate("", (0.8 * sx, 0.5 * sy), (0.8, 0.5), arrowprops=dict(arrowstyle="->", color=plum, lw=0.9, linestyle=":", shrinkA=10, shrinkB=10))
    for sy in (1, -1):
        pts = axis_cloud * [1, sy]; ax.scatter(pts[:, 0], pts[:, 1], s=9, color=gray, alpha=0.7, edgecolors="none", zorder=3)
    if k in (2, 3): ax.annotate("", (0, -1.05), (0, 1.05), arrowprops=dict(arrowstyle="->", color=gray, lw=0.8, linestyle=":", shrinkA=12, shrinkB=12))
    ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
axs[0].text(-0.15, 1.08, "D  the units: m and n transform like the plane, so the populations are a union of orbits (same picture for n)", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 3 (E): the overlaps, as predicted scatters of n against each input column, m (the cross-overlaps) and the marginal of n ─────────
axs = [panel_at(3, c) for c in range(4)]
ELEM = {"e": (1, 1, 1), "σ₁": None, "σ₃": None, "σ₂": None}
IDX = {"σ₁": 0, "σ₃": 1, "σ₂": 2}
SWAP = {"e": {}, "σ₁": {"A": "B", "B": "A", "C": "D", "D": "C"}, "σ₂": {"A": "B", "B": "A"}, "σ₃": {"C": "D", "D": "C"}}
GROUP = {0: ["e", "σ₁"], 1: ["e", "σ₂"], 2: ["e", "σ₃"], 3: ["e", "σ₁", "σ₂", "σ₃"]}
def chij(mode, g): return 1 if g == "e" else chi(CH[mode])[IDX[g]]
def smap(g, c): return SWAP[g].get(c, c)
base = {"A": np.array([0.75, 0.6]) + rng.normal(size=(110, 2)) @ np.array([[0.32, 0.22], [0.0, 0.10]]),
        "C": np.array([0.7, -0.55]) + rng.normal(size=(110, 2)) @ np.array([[0.30, -0.2], [0.0, 0.10]]),
        "go": np.array([0.6, 0.65]) + rng.normal(size=(110, 2)) @ np.array([[0.28, 0.18], [0.0, 0.12]]),
        "B*": np.array([-0.5, 0.7]) + rng.normal(size=(110, 2)) @ np.array([[0.3, -0.1], [0.0, 0.12]]),
        "D*": np.array([0.55, 0.5]) + rng.normal(size=(110, 2)) @ np.array([[0.3, 0.05], [0.0, 0.12]])}
REP = {"A": "A", "B": "A", "C": "C", "D": "C", "go": "go"}
ROWS = ["A", "B", "C", "D", "go"]
def cloud_for(k, mode, c):
    """the predicted (n_mode, w_c) cloud: images of the representative column's cloud under the elements that map it to c"""
    c0 = REP[c]; pts = []; related = False
    for g in GROUP[k]:
        if smap(g, c0) == c: pts.append(base[c0] * [chij(mode, g), 1]); related = related or (g != "e" or c == c0)
    if not pts:   # no element relates c to its representative (σ₃ alone for B): an unrelated cloud, closed under the group
        for g in GROUP[k]:
            if smap(g, c) == c: pts.append(base[c + "*"] * [chij(mode, g), 1])
        related = False
    return np.vstack(pts), related
def forced(k, mode, c):   # overlap n_mode·w_c forced to zero iff some element fixes c and flips the mode
    return any(smap(g, c) == c and chij(mode, g) == -1 for g in GROUP[k] if g != "e")
def relation(k, mode, c):  # label for a swapped column: how its cloud relates to the representative's
    if c == REP[c]: return ""
    for g in GROUP[k]:
        if g != "e" and smap(g, REP[c]) == c: return ("mirror of w_" + REP[c]) if chij(mode, g) == -1 else ("copy of w_" + REP[c])
    return "unrelated to w_" + REP[c]
for k, ax in enumerate(axs):
    ax.axis("off"); nrow = 7; h = 0.125; gap = 0.14
    for i, c in enumerate(ROWS):
        for j, mode in enumerate(["m₀", "m₁"]):   # n_j has the character of m_j
            ins = ax.inset_axes([0.2 + j * 0.42, 0.97 - (i + 1) * gap, 0.36, h]); ins.set_xlim(-1.5, 1.5); ins.set_ylim(-1.5, 1.5); ins.set_xticks([]); ins.set_yticks([])
            ins.axhline(0, color="0.85", lw=0.7); ins.axvline(0, color="0.85", lw=0.7)
            pts, rel = cloud_for(k, mode, c); z = forced(k, mode, c)
            ins.scatter(pts[:, 0], pts[:, 1], s=3, color=(gray if z else teal), alpha=0.6, edgecolors="none")
            lab = "0" if z else (relation(k, mode, c) or "free")
            ins.text(0.97, 0.05, lab, transform=ins.transAxes, ha="right", va="bottom", fontsize=(7.5 if len(lab) < 6 else 6.3), color=("0.35" if z else teal), fontweight=("bold" if len(lab) < 6 else "normal"))
        ax.text(0.17, 0.97 - (i + 1) * gap + h / 2, "w_" + c, transform=ax.transAxes, ha="right", va="center", fontsize=8)
    # cross-overlaps: (m₁, n₀) and (m₀, n₁)
    for j, (mode_n, mode_m) in enumerate([("m₀", "m₁"), ("m₁", "m₀")]):
        ins = ax.inset_axes([0.2 + j * 0.42, 0.97 - 6 * gap, 0.36, h]); ins.set_xlim(-1.5, 1.5); ins.set_ylim(-1.5, 1.5); ins.set_xticks([]); ins.set_yticks([])
        ins.axhline(0, color="0.85", lw=0.7); ins.axvline(0, color="0.85", lw=0.7)
        z = zero(k, mode_m, mode_n)
        for (sx, sy) in images(k, mode_n, mode_m):
            pts = base["go"] * [sx, sy]; ins.scatter(pts[:, 0], pts[:, 1], s=3, color=(gray if z else teal), alpha=0.6, edgecolors="none")
        ins.text(0.97, 0.05, "0" if z else "free", transform=ins.transAxes, ha="right", va="bottom", fontsize=7.5, color=("0.35" if z else teal), fontweight="bold")
        ins.set_ylabel(mode_m, fontsize=7.5, labelpad=1)
    ax.text(0.17, 0.97 - 6 * gap + h / 2, "J: m", transform=ax.transAxes, ha="right", va="center", fontsize=8)
    # marginal of n
    for j, mode in enumerate(["m₀", "m₁"]):
        ins = ax.inset_axes([0.2 + j * 0.42, 0.97 - 7 * gap, 0.36, h]); ins.set_xticks([]); ins.set_yticks([]); ins.set_xlim(-3, 3)
        sym = any(chij(mode, g) == -1 for g in GROUP[k] if g != "e")
        v = rng.normal(0.9, 0.5, 400); v = np.r_[v, -v] if sym else np.r_[v, rng.normal(-0.6, 0.5, 400)]
        ins.hist(v, bins=30, color=(gray if sym else teal), alpha=0.7); ins.axvline(0, color="0.5", lw=0.7)
        ins.text(0.03, 0.95, "⟨n⟩ = 0" if sym else "⟨n⟩ free", transform=ins.transAxes, ha="left", va="top", fontsize=7, color=("0.35" if sym else teal))
        ins.set_xlabel("n" + mode[1], fontsize=8, labelpad=1)
    ax.text(0.17, 0.97 - 7 * gap + h / 2, "marginal", transform=ax.transAxes, ha="right", va="center", fontsize=8)
    ax.text(0.41, 0.985, "n₀", transform=ax.transAxes, ha="center", va="bottom", fontsize=9); ax.text(0.83, 0.985, "n₁", transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
axs[0].text(-0.15, 1.04, "E  the overlaps as scatters: each input column and the other mode against n₀ and n₁; a swapped column's cloud is the image of its partner's, a fixed column against a flipped readout is mirror-symmetric (overlap 0)", transform=axs[0].transAxes, fontsize=11, fontweight="bold")

# ───────── row 4 (F): the input combinations against the modes, as predicted scatters ─────────
axs = [panel_at(4, c) for c in range(4)]
cloud = np.array([0.75, 0.6]) + rng.normal(size=(120, 2)) @ np.array([[0.32, 0.22], [0.0, 0.10]])   # a correlated cloud of units
notes5 = ["the test contrast may write on the memory readout;\nthe sample contrast never reaches the decision readout", "both readouts deaf to the test odors and to GNG:\nonly the sample contrast can be heard", "the test contrast may drive the decision directly\n(allowed, useless); the samples never reach it", "the test contrast is orthogonal to every mode: the pairing\nis gain modulation only; no common stimulus-on transient"]
COMBOS = [("w_A − w_B", "A−B"), ("w_A + w_B", "A+B"), ("w_C − w_D", "C−D"), ("w_C + w_D", "C+D"), ("w_go, w_nogo, w_cue", "go")]
for k, ax in enumerate(axs):
    ax.axis("off")
    for i, (lab, key) in enumerate(COMBOS):
        for j, xm in enumerate(["m₀", "m₁"]):
            ins = ax.inset_axes([0.2 + j * 0.42, 0.83 - i * 0.2, 0.36, 0.16]); ins.set_xlim(-1.5, 1.5); ins.set_ylim(-1.5, 1.5); ins.set_xticks([]); ins.set_yticks([])
            ins.axhline(0, color="0.85", lw=0.7); ins.axvline(0, color="0.85", lw=0.7)
            z = zero(k, key, xm)
            for (sx, sy) in images(k, xm, key):
                pts = cloud * [sx, sy]; ins.scatter(pts[:, 0], pts[:, 1], s=3, color=(gray if z else teal), alpha=0.6, edgecolors="none")
            ins.text(0.97, 0.05, ("0" if z else "free"), transform=ins.transAxes, ha="right", va="bottom", fontsize=7.5, color=("0.35" if z else teal), fontweight="bold")
            if i == 4: ins.set_xlabel(xm + " (or n" + xm[1] + ")", fontsize=8, labelpad=1)
        ax.text(0.17, 0.91 - i * 0.2, lab, transform=ax.transAxes, ha="right", va="center", fontsize=8)
    ax.text(0.5, -0.06, notes5[k], transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="0.35")
axs[0].text(-0.15, 1.05, "F  the inputs against the modes: a correlated cloud of units and the images the tie demands; an antipode keeps the correlation, a mirror image forces it to 0", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 5 (G): the rest of the predictions ─────────
ax = panel_at(5, slice(None)); ax.axis("off")
rows = [("trial trajectories", ["κ̄_B(t) = D₁κ̄_A(t): same κ₁ time course, opposite κ₀; the A·C lick is the mirror of the B·D lick", "κ̄_B(t) = −κ̄_A(t); the B·C response is minus the A·C response", "κ̄_A(t) on the κ₀ axis for all t; the A·D response is the mirror of the A·C response", "the four mean trial trajectories form one orbit"]),
        ("rest state (silence)", ["on the κ₁ axis", "the origin, exactly", "on the κ₀ axis", "the origin, exactly"]),
        ("invariant lines", ["κ₁ axis (F₀ = 0 on it)", "none", "κ₀ axis (F₁ = 0 on it)", "both axes"]),
        ("Jacobian at rest", ["diagonal: modes are eigen-directions", "unconstrained: complex pair possible → rotation", "diagonal", "diagonal"]),
        ("attractor count", ["odd allowed (lone axis well)", "even (origin excluded)", "odd allowed", "even; symmetric about both axes"]),
        ("image wells", ["same depth, same relaxation time τ, same landing spread", "same", "same", "same, for all four images"]),
        ("input-driven fields", ["A-field = D₁(B-field); C-field = D₁(D-field)", "A-field = −(B-field); C, D, go fields odd: cannot leave the origin", "C-field = D₃(D-field); A, go, nogo fields mirror-symmetric across the line: a go pulse at rest cannot lick", "all of these"]),
        ("input weights", ["a swapped column on a unit equals its partner column on the image unit; unswapped columns are constant on orbits", "same", "same", "each lattice population's input weights are set by one prototype; the axis population has ⟨w_A⟩ = ⟨w_B⟩ and ⟨w_C⟩ = ⟨w_D⟩"]),
        ("behavior", ["acc(A·C) = acc(B·D), acc(A·D) = acc(B·C)", "misses on A·C = false alarms on B·C (boundary at 0)", "acc(A·C) = acc(A·D): C and D tests equally hard", "all four trial types equally hard"]),
        ("single units", ["as many A- as B-preferring units, exactly; paired PSTHs", "unit ρ(i) on B trials = unit i on A trials, sign-flipped", "as many lick- as no-lick-preferring units", "memory and decision preferences independent across units (no mixed-selectivity correlation)"]),
        ("learning", ["the tied subspace is exact under Adam; released nets break in a seed-dependent direction", "same; the inversion is exact at init for lif (⟨n⟩ = 0, b = 0)", "same", "same; residuals seeded by finite N"])]
x0 = [0.0, 0.15, 0.36, 0.57, 0.78]; ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for j, (t, sub) in enumerate(TIES): ax.text(x0[j + 1], 0.985, t, ha="left", va="top", fontsize=9.5, fontweight="bold")
for i, (name, cells) in enumerate(rows):
    y = 0.94 - i * 0.085; ax.text(0.0, y, name, va="top", fontsize=9, fontweight="bold")
    for j, c in enumerate(cells): ax.text(x0[j + 1], y, textwrap.fill(c, 44), va="top", fontsize=7.6, color="0.2", linespacing=1.12)
    ax.plot([0, 1], [y + 0.02, y + 0.02], color="0.9", lw=0.6)
ax.text(0.0, 1.06, "G  the rest of what the group says, per tie", transform=ax.transAxes, fontsize=11.5, fontweight="bold")
fig1.suptitle("The Klein four-group of DPA: the group, its actions, and the wells and populations it predicts", fontsize=13, y=0.985)
fig2.suptitle("What the group predicts for the overlaps, the inputs, and the rest, tie by tie", fontsize=13, y=0.985)
fig1.savefig(out, bbox_inches="tight"); fig2.savefig(out2, bbox_inches="tight"); print("wrote", out, out2)
