"""The GNG stage: the go/nogo task, its relabeling tau, how tau relates to the DPA group, which elements of V
survive the GNG stage and why (the deafness lemma), and the predictions per element for the flows under go and
nogo input, the GNG columns against the modes, and the rest. No data. Usage: gng_theory_fig.py <out.png>"""
import sys, textwrap, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
teal, plum, ember, gray = "#0d818b", "#6b2f74", "#b0460e", "0.45"; cols = sns.color_palette("deep")
QC = {(1, 1): cols[0], (-1, 1): cols[1], (-1, -1): cols[2], (1, -1): cols[3]}
out = sys.argv[1]; rng = np.random.default_rng(5)
out2 = sys.argv[2] if len(sys.argv) > 2 else out.replace(".png", "_supp.png")
fig1 = plt.figure(figsize=(17, 19), dpi=130); gs1 = fig1.add_gridspec(4, 4, height_ratios=[1.35, 1.0, 1.0, 1.0], hspace=0.62, wspace=0.28, top=0.955, bottom=0.03, left=0.05, right=0.985)
fig2 = plt.figure(figsize=(17, 17), dpi=130); gs2 = fig2.add_gridspec(2, 4, height_ratios=[2.0, 2.0], hspace=0.5, wspace=0.28, top=0.94, bottom=0.02, left=0.05, right=0.985)
def panel_at(r, c): return fig1.add_subplot(gs1[r, c]) if r < 4 else fig2.add_subplot(gs2[r - 4, c])
# columns: the elements that matter in the GNG stage
COLS = [("σ₁ kept", "A↔B, C↔D: touches no GNG channel", "σ₁"), ("τ tied (two-sided GNG)", "go↔nogo, response flipped: acts as σ₃", "τ"),
        ("σ₂ or σ₃ kept", "flip κ₁ and fix go, nogo, cue: deaf", "σ₃"), ("free, one-sided GNG", "no symmetry: an explicit breaking field", "none")]
# characters (χ(σ₁), χ(σ₃)); for the GNG stage τ acts on the plane as σ₃ and swaps go↔nogo
CH = {"m₀": (-1, 1), "m₁": (1, -1), "A−B": (-1, 1), "C−D": (-1, -1), "go−nogo": (1, -1), "go+nogo": (1, 1), "cue": (1, 1), "b": (1, 1)}
# per column: the tied elements as (index into (σ₁, σ₃, σ₂), swap map)
GROUP = {"σ₁": [("σ₁", {"A": "B", "B": "A", "C": "D", "D": "C"})], "τ": [("τ", {"go": "nogo", "nogo": "go"})], "σ₃": [("σ₃", {"C": "D", "D": "C"})], "none": []}
IDX = {"σ₁": 0, "σ₃": 1, "σ₂": 2, "τ": 1}
def chi(c): return (c[0], c[1], c[0] * c[1])
def chi_in(name, g, swap):   # character of an input combination under element g: the combination's own sign flips only if the element swaps its channels
    if name in ("go−nogo",): return -1 if "go" in swap else 1
    if name in ("A−B",): return -1 if "A" in swap else 1
    if name in ("C−D",): return -1 if "C" in swap else 1
    return 1
def zero(colkey, mode, inp): return any(chi(CH[mode])[IDX[g]] != chi_in(inp, g, sw) for g, sw in GROUP[colkey])
def images(colkey, mode, inp):
    S = {(1, 1)}; changed = True
    while changed:
        changed = False
        for g, sw in GROUP[colkey]:
            new = {(chi(CH[mode])[IDX[g]] * sx, chi_in(inp, g, sw) * sy) for (sx, sy) in S}
            if not new <= S: S |= new; changed = True
    return sorted(S)

# ───────── row 0: the GNG task, its relabeling, the two objectives ─────────
ax = panel_at(0, slice(0, 2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0, 9.6, "A  The GNG task and its one candidate relabeling", fontsize=12, fontweight="bold", va="top")
for i, (stim, resp, col) in enumerate([("go", "lick", teal), ("nogo", "no lick", "0.92")]):
    y = 7.6 - i * 1.3
    ax.text(0.9, y, f"{stim} + cue", ha="center", va="center", fontsize=10)
    ax.add_patch(Rectangle((2.0, y - 0.5), 1.8, 1.0, fc=col, ec="0.5", lw=0.8)); ax.text(2.9, y, resp, ha="center", va="center", fontsize=10, color=("w" if i == 0 else "0.3"))
ax.text(2.4, 5.3, "one bit: r = s; the cue only says when", ha="center", fontsize=9.5, color="0.3")
ax.text(5.0, 8.4, "τ: exchange go with nogo and flip the response", fontsize=10.5, va="top", color=ember, fontweight="bold")
ax.text(5.0, 7.6, "the only relabeling of the abstract GNG rule; with e it is a Z₂", fontsize=9.5, va="top", color="0.3")
ax.text(5.0, 6.4, "but the objective we train is one-sided:", fontsize=10.5, va="top")
ax.text(5.2, 5.7, "go:   κ₁ ≥ +θ after the cue\nnogo: κ₁ ≤ 0 after the cue (no target below)", fontsize=9.5, va="top", family="monospace")
ax.text(5.0, 4.2, "τ maps the go cost onto a nogo cost that does not\nexist: the one-sided GNG is invariant under nothing.\nA two-sided GNG (nogo → −θ) would be τ-invariant.\nThe asymmetry is an explicit symmetry-breaking\nfield with a definite sign.", fontsize=9.2, va="top", color="0.3")
ax.text(0.2, 4.2, "the GNG stage of the curriculum releases\nthe memory tie, freezes the memory mode and\nthe sample columns, and trains the decision\nmode, the bias, and the go, nogo, cue columns", fontsize=9.2, va="top", color="0.3")

ax = panel_at(0, slice(2, 4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0, 9.6, "B  How τ acts, and which elements of V can survive GNG", fontsize=12, fontweight="bold", va="top")
cx, cy, s_ = 1.2, 7.6, 0.9
ax.plot([cx - s_, cx + s_], [cy, cy], color="0.85", lw=0.8); ax.plot([cx, cx], [cy - s_, cy + s_], color="0.85", lw=0.8)
p = np.array([0.55, 0.45]); ax.plot(cx + p[0], cy + p[1], "o", color="k", ms=6); ax.plot(cx + p[0], cy - p[1], "o", mfc="none", mec=ember, mew=1.8, ms=7)
ax.add_patch(FancyArrowPatch((cx + p[0], cy + p[1]), (cx + p[0], cy - p[1]), arrowstyle="->", color=ember, lw=1, mutation_scale=10, linestyle=":", shrinkA=4, shrinkB=4))
ax.text(cx, cy - 1.3, "τ: (+κ₀, −κ₁), the action of σ₃", ha="center", fontsize=9.5, color=ember)
ax.text(2.9, 8.5, "on the units: m₁, n₁ flipped, go and nogo columns exchanged,\ncue and bias shared, the memory side untouched", fontsize=9.5, va="top")
ax.text(0, 5.6, "the deafness lemma: for an element that fixes a channel c and flips a readout,", fontsize=10, va="top", fontweight="bold")
ax.text(0.3, 4.9, "nᵀ w_c = Σ_pairs (ν − ν) w_c = 0,  exactly", fontsize=10.5, va="top", family="serif")
rowsB = [("σ₁", "keeps κ₁; fixes go, nogo, cue", "n₀·w_go = n₀·w_nogo = n₀·w_cue = 0: GNG cannot write on the memory", "compatible: protects the memory"),
         ("σ₂, σ₃", "flip κ₁; fix go, nogo, cue", "n₁·w_go = n₁·w_nogo = n₁·w_cue = 0: the decision cannot hear GNG", "incompatible with any rule"),
         ("τ", "flips κ₁; swaps go ↔ nogo", "n₁·w_go = −n₁·w_nogo, n₀·w_go = n₀·w_nogo", "compatible with a two-sided GNG only")]
for i, (el, act, cons, verdict) in enumerate(rowsB):
    y = 3.9 - i * 1.15
    ax.text(0.0, y, el, fontsize=10, fontweight="bold", va="top", color=(teal if i == 0 else plum if i == 1 else ember))
    ax.text(1.1, y, act + "  —  " + verdict, fontsize=8.8, va="top", fontweight="bold"); ax.text(1.1, y - 0.45, cons, fontsize=8.4, va="top", color="0.3")

def plane(ax, lab_x="κ₀", lab_y="κ₁", lick=True, lim=1.4):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(0, color="0.85", lw=0.8); ax.axhline(0, color=(ember if lick else "0.85"), lw=(1.1 if lick else 0.8), ls=((0, (4, 3)) if lick else "-"))
    ax.set_xlabel(lab_x, fontsize=9, labelpad=1); ax.set_ylabel(lab_y, fontsize=9)
def dot(ax, x, y, c=teal, open_=False, ms=9, **kw): ax.plot(x, y, "o", ms=ms, **(dict(mfc="none", mec=c, mew=1.6) if open_ else dict(color=c)), zorder=4, **kw)
def link(ax, p, q, c=plum): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<->", color=c, lw=0.9, mutation_scale=9, linestyle=":", shrinkA=5, shrinkB=5))

# ───────── row 1 (C): the go-driven and nogo-driven fields ─────────
axs = [panel_at(1, c) for c in range(4)]
for ax, (t, sub, key) in zip(axs, COLS): plane(ax); ax.set_title(t + "\n" + sub, fontsize=9.5, pad=6)
# σ₁ kept: F₀ = 0 on the κ₁ axis for go, nogo and cue input ⇒ the GNG wells sit ON the decision axis; memory pair untouched
dot(axs[0], 0, 1.0, c=teal); dot(axs[0], 0, -0.45, c=teal, open_=True); dot(axs[0], 0.95, -0.22, c="0.6", ms=7); dot(axs[0], -0.95, -0.22, c="0.6", ms=7)
axs[0].text(0.5, -0.2, "go well and nogo state ON the κ₁ axis (F₀ = 0 there\nfor every GNG input); the memory pair untouched", transform=axs[0].transAxes, ha="center", va="top", fontsize=8, color="0.35")
# τ tied: go-field = D₃(nogo-field): the go well at (u, +w) and the nogo well at (u, −w), a mirror pair across the line
dot(axs[1], 0.1, 0.95, c=teal); dot(axs[1], 0.1, -0.95, c=ember); link(axs[1], (0.1, 0.95), (0.1, -0.95), c=ember)
axs[1].text(0.5, -0.2, "go-driven field = mirror of the nogo-driven field:\na go well at (u, +w) and a nogo well at (u, −w)", transform=axs[1].transAxes, ha="center", va="top", fontsize=8, color="0.35")
# σ₂/σ₃ kept: the go-driven field is mirror-symmetric across the line ⇒ a go pulse at rest cannot lick: either no lick or a symmetric pair
dot(axs[2], 0, 0.9, c="0.6", open_=True); dot(axs[2], 0, -0.9, c="0.6", open_=True); link(axs[2], (0, 0.9), (0, -0.9), c="0.6"); axs[2].text(0, 0, "✕", ha="center", va="center", fontsize=16, color=ember)
axs[2].text(0.5, -0.2, "go-driven field mirror-symmetric across the line:\nno lick from rest; the GNG stage must break it", transform=axs[2].transAxes, ha="center", va="top", fontsize=8, color="0.35")
# free one-sided: go well above, nogo anywhere at or below, off-axis allowed
dot(axs[3], 0.25, 1.0, c=teal); dot(axs[3], -0.3, -0.4, c=teal, open_=True)
axs[3].text(0.5, -0.2, "go well above the line; the nogo state anywhere\nat or below it; nothing pins either to an axis", transform=axs[3].transAxes, ha="center", va="top", fontsize=8, color="0.35")
axs[0].text(-0.15, 1.42, "C  the input-driven fields of the GNG stage: where the go well and the nogo state may sit (● go, ○ nogo, gray the memory pair)", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 2 (D): what the GNG stage changes — the autonomous field and the input columns ─────────
axs = [panel_at(2, c) for c in range(4)]
notesD = ["GNG enters the go, nogo and cue columns of the\ndecision readout; the autonomous field and the memory\nwells are unchanged; σ₁ residual stays where DPA left it",
          "two-sided: τ is exact through the stage; go and nogo\noverlaps exactly opposite on both readouts; the\nautonomous field stays τ-symmetric (odd in κ₁)",
          "the stage breaks the element from an exact zero:\nn₁·w_go and n₁·w_nogo grow from 0.000; the σ₃\nresidual of the autonomous field barely moves",
          "the go and nogo overlaps grow unequal, and the cue\nacquires an overlap with κ₁; the field stays as odd\nas the two-sided one: the break lands in the inputs"]
for ax, note, (t, sub, key) in zip(axs, notesD, COLS):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    items = {"σ₁": [("n₀·w_go, n₀·w_nogo, n₀·w_cue", "0", True), ("n₁·w_go, n₁·w_nogo, n₁·w_cue", "free", False), ("memory wells", "unchanged", True), ("v_σ₁ of the field", "unchanged", True)],
             "τ": [("n₁·w_go = −n₁·w_nogo", "exact", True), ("n₀·w_go = n₀·w_nogo", "exact", True), ("n·w_cue on n₁", "0", True), ("F(κ₀, −κ₁) = D₃F(κ)", "exact", True)],
             "σ₃": [("n₁·w_go, n₁·w_nogo, n₁·w_cue", "0 → grows", False), ("v_σ₃ of the field", "≈ unchanged", True), ("v_σ₁ (if σ₁ was kept)", "unchanged", True), ("memory wells", "unchanged", True)],
             "none": [("n₁·w_go vs −n₁·w_nogo", "unequal", False), ("n₁·w_cue", "> 0", False), ("oddness of the field", "≈ kept", True), ("n₀·w_nogo (a leak)", "may grow", False)]}[key]
    for i, (what, how, ok) in enumerate(items):
        y = 0.9 - i * 0.2
        ax.add_patch(Rectangle((0.02, y - 0.08), 0.96, 0.16, fc=(teal if ok else "0.94"), alpha=(0.85 if ok else 1), ec="0.6", lw=0.5))
        ax.text(0.05, y, what, va="center", fontsize=8.2, color=("w" if ok else "0.25")); ax.text(0.95, y, how, va="center", ha="right", fontsize=8.6, fontweight="bold", color=("w" if ok else "0.25"))
    ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="0.35")
axs[0].text(-0.15, 1.1, "D  what the GNG stage can and cannot change (teal: fixed by the element; gray: what training moves)", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 3 (E): the units — go/nogo columns and the decision mode, as population orbits ─────────
axs = [panel_at(3, c) for c in range(4)]
base = np.array([0.8, 0.5]) + rng.normal(size=(70, 2)) * [0.13, 0.09]
axis_cloud = np.array([0.0, 1.05]) + rng.normal(size=(30, 2)) * [0.05, 0.12]
notesE = ["the memory tie is released: the lattice keeps σ₁\n(mirror across m₁) as a trend; ⟨n₀⟩ = 0 as a trend",
          "τ pairs every unit with a twin at (m₀, −m₁): the\nlattice is exactly mirror-symmetric across m₀;\nthe axis pair is exchanged; ⟨n₁⟩ = 0",
          "a kept σ₃ makes the same lattice, but GNG\ncannot be learned in it: the tie must break, and the\nfirst thing to go is the mirror in the go/nogo columns",
          "no orbit structure required; the lattice of the\nmemory stage persists because the memory mode\nis frozen; the decision side drifts"]
imgs = {"σ₁": [(-1, 1)], "τ": [(1, -1)], "σ₃": [(1, -1)], "none": []}
for ax, note, (t, sub, key) in zip(axs, notesE, COLS):
    plane(ax, "m₀", "m₁", lick=False)
    pts = base; ax.scatter(pts[:, 0], pts[:, 1], s=9, color=QC[(1, 1)], alpha=0.75, edgecolors="none", zorder=3)
    for (sx, sy) in imgs[key]:
        q = base * [sx, sy]; ax.scatter(q[:, 0], q[:, 1], s=9, color=QC[(sx, sy)], alpha=(0.75 if key != "σ₁" else 0.45), edgecolors="none", zorder=3)
        ax.annotate("", (0.8 * sx, 0.5 * sy), (0.8, 0.5), arrowprops=dict(arrowstyle="->", color=plum, lw=0.9, linestyle=(":" if key != "σ₁" else (0, (1, 3))), shrinkA=10, shrinkB=10))
    for sy in (1, -1):
        q = axis_cloud * [1, sy]; ax.scatter(q[:, 0], q[:, 1], s=9, color=gray, alpha=0.7, edgecolors="none", zorder=3)
    if key in ("τ", "σ₃"): ax.annotate("", (0, -1.05), (0, 1.05), arrowprops=dict(arrowstyle="->", color=gray, lw=0.8, linestyle=":", shrinkA=12, shrinkB=12))
    ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
axs[0].text(-0.15, 1.08, "E  the units in the GNG stage: which images the lattice keeps (exact under a tie, a trend after release)", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 4 (F): the GNG columns against the modes, as predicted scatters ─────────
axs = [panel_at(4, c) for c in range(4)]
cloud = np.array([0.75, 0.6]) + rng.normal(size=(120, 2)) @ np.array([[0.32, 0.22], [0.0, 0.10]])
COMBOS = [("w_go − w_nogo", "go−nogo"), ("w_go + w_nogo", "go+nogo"), ("w_cue", "cue"), ("w_A − w_B", "A−B"), ("w_C − w_D", "C−D")]
notesF = ["the GNG columns are orthogonal to the memory mode: GNG writes on κ₁ only and the sample contrast stays with κ₀",
          "the go/nogo contrast may drive κ₁ and only κ₁; the sum and the cue are orthogonal to κ₁: no common go/nogo drive on the decision",
          "every GNG column is orthogonal to the decision mode: GNG can only be heard through the memory mode or by gain modulation; the stage must break this",
          "nothing forced; the network is free to put the go/nogo sum and the cue on the decision readout, which is the cue → κ₁ overlap the free networks show"]
for k, (ax, (t, sub, key)) in enumerate(zip(axs, COLS)):
    ax.axis("off")
    for i, (lab, inp) in enumerate(COMBOS):
        for j, xm in enumerate(["m₀", "m₁"]):
            ins = ax.inset_axes([0.2 + j * 0.42, 0.83 - i * 0.2, 0.36, 0.16]); ins.set_xlim(-1.5, 1.5); ins.set_ylim(-1.5, 1.5); ins.set_xticks([]); ins.set_yticks([])
            ins.axhline(0, color="0.85", lw=0.7); ins.axvline(0, color="0.85", lw=0.7)
            z = zero(key, xm, inp)
            for (sx, sy) in images(key, xm, inp):
                pts = cloud * [sx, sy]; ins.scatter(pts[:, 0], pts[:, 1], s=3, color=(gray if z else teal), alpha=0.6, edgecolors="none")
            ins.text(0.97, 0.05, ("0" if z else "free"), transform=ins.transAxes, ha="right", va="bottom", fontsize=7.5, color=("0.35" if z else teal), fontweight="bold")
            if i == 4: ins.set_xlabel(xm + " (or n" + xm[1] + ")", fontsize=8, labelpad=1)
        ax.text(0.17, 0.91 - i * 0.2, lab, transform=ax.transAxes, ha="right", va="center", fontsize=8)
    ax.text(0.5, -0.06, textwrap.fill(notesF[k], 60), transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="0.35")
axs[0].text(-0.15, 1.05, "F  the GNG columns against the modes: a correlated cloud and the images the element demands (a mirror image forces the correlation to 0)", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 5 (G): the rest ─────────
ax = panel_at(5, slice(None)); ax.axis("off")
rows = [("which elements survive", ["σ₁ survives the GNG stage: it constrains nothing GNG needs", "τ is exact if tied (two-sided GNG); it is GNG's own group", "σ₂ and σ₃ cannot survive any rule: the stage breaks them from an exact zero", "no element is kept; the break lands in the input columns first"]),
        ("autonomous field", ["unchanged (memory mode and sample columns frozen); wells stay on the line", "odd in κ₁ exactly; a GNG-stage well off the line brings its mirror", "barely changes; the residual of the broken element rises through the bias and ⟨n₁⟩ only", "as odd as the two-sided one (0.05–0.09); GNG lives in the inputs"]),
        ("trial trajectories", ["go and nogo trials stay on the κ₁ axis in the mean (σ₁ fixes both channels)", "the nogo trajectory is the mirror of the go trajectory: κ̄_nogo(t) = D₃ κ̄_go(t)", "go and nogo trajectories identical in κ₁ until the element breaks", "no relation"]),
        ("memory retention", ["GNG cannot write on the memory readout (n₀·w_go = n₀·w_nogo = 0): after_gng/dpa protected", "τ says nothing about the memory", "a broken σ₃ does not protect the memory; retention depends on σ₁", "the nogo → n₀ leak is allowed and appears in several seeds"]),
        ("behavior", ["go and nogo accuracies unrelated", "acc(go) = acc(nogo); miss rate = false-alarm rate", "chance until the element breaks", "go ≈ 1, nogo variable: the one-sided cost prices only one error"]),
        ("single units", ["as many A- as B-preferring units still; go/nogo preference independent of sample preference", "as many go- as nogo-preferring units, exactly; paired PSTHs", "no unit prefers go over nogo until the break", "go-preferring units outnumber nogo-preferring ones"]),
        ("what Dual inherits", ["a memory pair at one height, with the GNG wells on the axis", "a decision axis that is mirror-symmetric: the Dual cost must break τ to put the memory below the line", "the seed-dependent direction of the break", "a field that is still nearly odd, and inputs that are not"])]
x0 = [0.0, 0.15, 0.36, 0.57, 0.78]; ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for j, (t, sub, key) in enumerate(COLS): ax.text(x0[j + 1], 0.985, t, ha="left", va="top", fontsize=9.5, fontweight="bold")
for i, (name, cells) in enumerate(rows):
    y = 0.92 - i * 0.13; ax.text(0.0, y, textwrap.fill(name, 16), va="top", fontsize=9, fontweight="bold")
    for j, c in enumerate(cells): ax.text(x0[j + 1], y, textwrap.fill(c, 44), va="top", fontsize=7.6, color="0.2", linespacing=1.12)
    ax.plot([0, 1], [y + 0.02, y + 0.02], color="0.9", lw=0.6)
ax.text(0.0, 1.05, "G  the rest of what the group says about the GNG stage", transform=ax.transAxes, fontsize=11.5, fontweight="bold")
fig1.suptitle("The GNG stage: one candidate relabeling, an objective that breaks it, and what each element of the DPA group predicts", fontsize=13, y=0.985)
fig2.suptitle("The GNG stage, continued: the GNG columns against the modes, and the rest of the predictions", fontsize=13, y=0.985)
fig1.savefig(out, bbox_inches="tight"); fig2.savefig(out2, bbox_inches="tight"); print("wrote", out, out2)
