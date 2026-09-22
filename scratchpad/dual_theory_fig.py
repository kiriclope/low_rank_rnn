"""The Dual task: its truth table, the relabelings it inherits from DPA and GNG, how sharing one decision axis
couples them, which elements survive the two-sided and the one-sided objective, and the predictions per
element for the flow after Dual, the populations, what the stage can change; a second output carries the
input scatters and the table. No data. Usage: dual_theory_fig.py <out.png> [<out_supp.png>]"""
import sys, textwrap, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle
sns.set_context("notebook"); sns.set_style("ticks"); plt.rc("axes.spines", top=False, right=False)
teal, plum, ember, gray = "#0d818b", "#6b2f74", "#b0460e", "0.45"; cols = sns.color_palette("deep")
QC = {(1, 1): cols[0], (-1, 1): cols[1], (-1, -1): cols[2], (1, -1): cols[3]}
out = sys.argv[1]; out2 = sys.argv[2] if len(sys.argv) > 2 else out.replace(".png", "_supp.png"); rng = np.random.default_rng(7)
fig1 = plt.figure(figsize=(17, 19), dpi=130); gs1 = fig1.add_gridspec(4, 4, height_ratios=[1.45, 1.0, 1.0, 1.0], hspace=0.62, wspace=0.28, top=0.955, bottom=0.03, left=0.05, right=0.985)
fig2 = plt.figure(figsize=(17, 17), dpi=130); gs2 = fig2.add_gridspec(2, 4, height_ratios=[2.0, 2.0], hspace=0.5, wspace=0.28, top=0.94, bottom=0.02, left=0.05, right=0.985)
def panel_at(r, c): return fig1.add_subplot(gs1[r, c]) if r < 4 else fig2.add_subplot(gs2[r - 4, c])
COLS = [("σ₁ kept", "A↔B, C↔D; touches no GNG channel", "σ₁"), ("σ₃τ tied (two-sided Dual)", "C↔D and go↔nogo, both responses flipped", "σ₃τ"),
        ("σ₂τ tied (two-sided Dual)", "A↔B and go↔nogo, both responses flipped", "σ₂τ"), ("free, the real Dual", "one-sided nogo and no-lick: no κ₁ flip survives", "none")]
# characters (χ(σ₁), χ(σ₃)); τ acts on the plane as σ₃ and swaps go↔nogo; σ₃τ = plane σ₃ with both C↔D and go↔nogo; σ₂τ = plane σ₂ with A↔B and go↔nogo
CH = {"m₀": (-1, 1), "m₁": (1, -1)}
GROUP = {"σ₁": [("σ₁", {"A": "B", "B": "A", "C": "D", "D": "C"})], "σ₃τ": [("σ₃", {"C": "D", "D": "C", "go": "nogo", "nogo": "go"})], "σ₂τ": [("σ₂", {"A": "B", "B": "A", "go": "nogo", "nogo": "go"})], "none": []}
IDX = {"σ₁": 0, "σ₃": 1, "σ₂": 2}
def chi(c): return (c[0], c[1], c[0] * c[1])
def chi_in(name, sw):
    if name == "go−nogo": return -1 if "go" in sw else 1
    if name == "A−B": return -1 if "A" in sw else 1
    if name == "C−D": return -1 if "C" in sw else 1
    return 1
def zero(key, mode, inp): return any(chi(CH[mode])[IDX[g]] != chi_in(inp, sw) for g, sw in GROUP[key])
def images(key, mode, inp):
    S = {(1, 1)}; changed = True
    while changed:
        changed = False
        for g, sw in GROUP[key]:
            new = {(chi(CH[mode])[IDX[g]] * sx, chi_in(inp, sw) * sy) for (sx, sy) in S}
            if not new <= S: S |= new; changed = True
    return sorted(S)

# ───────── row 0: the Dual task and its group ─────────
ax = panel_at(0, slice(0, 2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0, 9.7, "A  The Dual task: two bits in, two responses on one axis", fontsize=12, fontweight="bold", va="top")
ax.text(0, 8.9, "sample A/B  →  go/nogo  →  cue  →  test C/D", fontsize=10.5, va="top", family="monospace")
ax.text(0, 8.2, "at the cue: lick iff go          at the test: lick iff the pair matches", fontsize=9.5, va="top", color="0.3")
ax.text(0, 7.3, "relabelings of the abstract task: V = {e, σ₁, σ₂, σ₃} from DPA and τ from GNG, commuting: V × Z₂", fontsize=10, va="top")
ax.text(0, 6.4, "but both responses are read from the same κ₁, so an element that flips κ₁ must flip BOTH responses:", fontsize=10, va="top")
ax.text(0.3, 5.7, "σ₃ alone flips the DPA response and fixes go, nogo ⇒ deaf to GNG (n₁·w_go = 0): out\nτ alone flips the GNG response and fixes C, D ⇒ deaf to the test odors on κ₁ (n₁·w_C = 0): out\nσ₃τ and σ₂τ flip κ₁ and swap both pairs: consistent", fontsize=9.3, va="top", color="0.3")
ax.text(0, 3.7, "two-sided Dual (nogo → −θ, nonmatch → −θ):  V′ = {e, σ₁, σ₂τ, σ₃τ} ≅ V, the same plane actions", fontsize=10, va="top", fontweight="bold")
ax.text(0, 2.9, "the real Dual: nogo one-sided, and no lick where a lick is wrong (κ₁ ≤ 0 in the delay windows):\nevery element with a minus sign on κ₁ is broken by the objective ⇒  Z₂ = {e, σ₁}", fontsize=10, va="top", fontweight="bold", color=ember)
ax.text(0, 1.5, "so the only symmetry the real Dual objective has is the one that leaves the decision axis alone;\nthe no-lick cost is an explicit breaking field with a definite sign, downward", fontsize=9.5, va="top", color="0.3")

ax = panel_at(0, slice(2, 4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0, 9.7, "B  How the surviving elements act, and what the stage trains", fontsize=12, fontweight="bold", va="top")
for k, (name, D, col) in enumerate([("σ₁: (−κ₀, +κ₁)", np.diag([-1, 1]), teal), ("σ₃τ: (+κ₀, −κ₁)", np.diag([1, -1]), ember), ("σ₂τ: (−κ₀, −κ₁)", -np.eye(2), plum)]):
    cx, cy, s_ = 1.0 + k * 3.1, 7.6, 0.9
    ax.plot([cx - s_, cx + s_], [cy, cy], color="0.85", lw=0.8); ax.plot([cx, cx], [cy - s_, cy + s_], color="0.85", lw=0.8)
    p = np.array([0.55, 0.35]); q = D @ p
    ax.plot(cx + p[0], cy + p[1], "o", color="k", ms=6); ax.plot(cx + q[0], cy + q[1], "o", mfc="none", mec=col, mew=1.8, ms=7)
    ax.add_patch(FancyArrowPatch((cx + p[0], cy + p[1]), (cx + q[0], cy + q[1]), arrowstyle="->", color=col, lw=1, mutation_scale=10, linestyle=":", shrinkA=4, shrinkB=4))
    ax.text(cx, cy - 1.25, name, ha="center", fontsize=9.5, color=col)
ax.text(0, 5.6, "on the units: σ₁ as in DPA; σ₃τ flips m₁, n₁ and exchanges C↔D AND go↔nogo; σ₂τ flips both modes and exchanges A↔B AND go↔nogo;\ncue and bias shared throughout", fontsize=9.3, va="top")
ax.text(0, 4.3, "the Dual stage: inputs and bias frozen, m and n trained on the whole trial with the pairing, the go/nogo\nresponse, and the no-lick cost; the memory mode is free (unless freeze_rank0_dual)", fontsize=9.3, va="top", color="0.3")
ax.text(0, 2.9, "what the breaking field can move (Theorem 6.1): only ⟨n⟩ and the bias break the inversion; with the bias\nfrozen the Dual cost has one handle, ⟨n₁⟩, which it drives down", fontsize=9.3, va="top", color="0.3")
ax.text(0, 1.5, "σ₁, if it holds, locks the two memory wells to a common height: they descend together or not at all", fontsize=10, va="top", fontweight="bold")

def plane(ax, lab_x="κ₀", lab_y="κ₁", lick=True, lim=1.4):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(0, color="0.85", lw=0.8); ax.axhline(0, color=(ember if lick else "0.85"), lw=(1.1 if lick else 0.8), ls=((0, (4, 3)) if lick else "-"))
    ax.set_xlabel(lab_x, fontsize=9, labelpad=1); ax.set_ylabel(lab_y, fontsize=9)
def dot(ax, x, y, c=teal, open_=False, ms=9, **kw): ax.plot(x, y, "o", ms=ms, **(dict(mfc="none", mec=c, mew=1.6) if open_ else dict(color=c)), zorder=4, **kw)
def link(ax, p, q, c=plum): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<->", color=c, lw=0.9, mutation_scale=9, linestyle=":", shrinkA=5, shrinkB=5))
a = 0.95

# ───────── row 1 (C): the autonomous flow after Dual ─────────
axs = [panel_at(1, c) for c in range(4)]
for ax, (t, sub, key) in zip(axs, COLS): plane(ax); ax.set_title(t + "\n" + sub, fontsize=9.5, pad=6)
dot(axs[0], a, -0.55); dot(axs[0], -a, -0.55); link(axs[0], (a, -0.55), (-a, -0.55)); dot(axs[0], 0, 1.0, open_=True); dot(axs[0], a, 0.45, open_=True, ms=8, alpha=0.45); dot(axs[0], -a, 0.45, open_=True, ms=8, alpha=0.45)
axs[0].text(0.5, -0.2, "a level pair, pushed below the line together by the\nbreaking field; an unoccupied upper pair allowed", transform=axs[0].transAxes, ha="center", va="top", fontsize=8, color="0.35")
dot(axs[1], a, 0); dot(axs[1], -0.7, 0); dot(axs[1], 0, 1.0, open_=True); dot(axs[1], 0, -1.0, open_=True); link(axs[1], (0, 1.0), (0, -1.0), c=ember)
axs[1].text(0.5, -0.2, "the κ₀ axis invariant: the memory cannot leave the\nline; the go and nogo wells a mirror pair", transform=axs[1].transAxes, ha="center", va="top", fontsize=8, color="0.35")
dot(axs[2], a, 0.35); dot(axs[2], -a, -0.35); link(axs[2], (a, 0.35), (-a, -0.35))
axs[2].text(0.5, -0.2, "antipodes: one memory above the line, one below,\nunless both sit on it; forbidden by no-lick", transform=axs[2].transAxes, ha="center", va="top", fontsize=8, color="0.35")
dot(axs[3], a, -0.6); dot(axs[3], -a, -0.5); dot(axs[3], 0.2, 1.0, open_=True)
axs[3].text(0.5, -0.2, "both below if σ₁ survived the earlier stages;\notherwise one up, one down (a residual σ₂)", transform=axs[3].transAxes, ha="center", va="top", fontsize=8, color="0.35")
axs[0].text(-0.15, 1.42, "C  the autonomous flow after Dual: the breaking field is downward, and σ₁ decides whether the pair moves together", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 2 (D): the populations after Dual ─────────
axs = [panel_at(2, c) for c in range(4)]
base = np.array([0.8, 0.5]) + rng.normal(size=(70, 2)) * [0.13, 0.09]; axis_cloud = np.array([0.0, 1.05]) + rng.normal(size=(30, 2)) * [0.05, 0.12]
imgs = {"σ₁": [(-1, 1)], "σ₃τ": [(1, -1)], "σ₂τ": [(-1, -1)], "none": []}
notesD = ["mirror across m₁ as a trend; the two memory\npopulations keep equal |m₀|; ⟨n₀⟩ ≈ 0", "exact mirror across m₀; ⟨n₁⟩ = 0: the readout\nmean that lowers the pair is forbidden", "exact antipodes; ⟨n⟩ = 0 on both modes;\nthe lattice may tilt (J free)", "the DPA lattice persists (memory mode frozen or\nslow); ⟨n₁⟩ drifts negative: the pair descends"]
for ax, note, (t, sub, key) in zip(axs, notesD, COLS):
    plane(ax, "m₀", "m₁", lick=False)
    ax.scatter(base[:, 0], base[:, 1], s=9, color=QC[(1, 1)], alpha=0.75, edgecolors="none", zorder=3)
    for (sx, sy) in imgs[key]:
        q = base * [sx, sy]; ax.scatter(q[:, 0], q[:, 1], s=9, color=QC[(sx, sy)], alpha=(0.45 if key == "σ₁" else 0.75), edgecolors="none", zorder=3)
        ax.annotate("", (0.8 * sx, 0.5 * sy), (0.8, 0.5), arrowprops=dict(arrowstyle="->", color=plum, lw=0.9, linestyle=((0, (1, 3)) if key == "σ₁" else ":"), shrinkA=10, shrinkB=10))
    for sy in (1, -1):
        q = axis_cloud * [1, sy]; ax.scatter(q[:, 0], q[:, 1], s=9, color=gray, alpha=0.7, edgecolors="none", zorder=3)
    ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
axs[0].text(-0.15, 1.08, "D  the populations after Dual: which images survive, and which readout mean is free to move", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── row 3 (E): what the Dual stage can and cannot change ─────────
axs = [panel_at(3, c) for c in range(4)]
notesE = ["the pair descends together: the breaking field acts\non the common height; imbalance forbidden", "the memory cannot descend at all: ⟨n₁⟩ = 0 and the\nκ₀ axis is invariant; the no-lick cost is unpayable", "the memory can only tilt: one well up, one down;\nthe no-lick cost forces the tie to break", "which well descends is decided by what DPA and\nGNG left of σ₁ and σ₂ (the ledger ordering)"]
items = {"σ₁": [("common height of the pair", "exact", True), ("⟨n₁⟩", "free ↓", False), ("J₀₁, J₁₀", "0", True), ("depth", "free ↓", False)],
         "σ₃τ": [("wells on the line", "exact", True), ("⟨n₁⟩", "0", True), ("depth", "0", True), ("no-lick cost", "unpayable", True)],
         "σ₂τ": [("antipodal pair", "exact", True), ("⟨n₀⟩, ⟨n₁⟩", "0", True), ("J₀₁, J₁₀", "free", False), ("no-lick cost", "half the trials", True)],
         "none": [("⟨n₁⟩", "drives down", False), ("bias", "frozen (fixed code)", True), ("pair imbalance", "free", False), ("upper well", "allowed", False)]}
for ax, note, (t, sub, key) in zip(axs, notesE, COLS):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, (what, how, ok) in enumerate(items[key]):
        y = 0.9 - i * 0.2
        ax.add_patch(Rectangle((0.02, y - 0.08), 0.96, 0.16, fc=(teal if ok else "0.94"), alpha=(0.85 if ok else 1), ec="0.6", lw=0.5))
        ax.text(0.05, y, what, va="center", fontsize=8.4, color=("w" if ok else "0.25")); ax.text(0.95, y, how, va="center", ha="right", fontsize=8.6, fontweight="bold", color=("w" if ok else "0.25"))
    ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="0.35")
axs[0].text(-0.15, 1.1, "E  what the Dual stage can and cannot change (teal: fixed by the element; gray: what the breaking field moves)", transform=axs[0].transAxes, fontsize=11.5, fontweight="bold")

# ───────── supplement row 4 (F): inputs against the modes ─────────
axs = [panel_at(4, c) for c in range(4)]
cloud = np.array([0.75, 0.6]) + rng.normal(size=(120, 2)) @ np.array([[0.32, 0.22], [0.0, 0.10]])
COMBOS = [("w_A − w_B", "A−B"), ("w_C − w_D", "C−D"), ("w_go − w_nogo", "go−nogo"), ("w_go + w_nogo", "go+nogo"), ("w_cue", "cue")]
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
    ax.set_title(t, fontsize=9.5, pad=14)
axs[0].text(-0.15, 1.12, "F  the inputs against the modes under the Dual elements (inputs are frozen in Dual: these are inherited from DPA and GNG, and checked, not trained)", transform=axs[0].transAxes, fontsize=11, fontweight="bold")

# ───────── supplement row 5 (G): the rest ─────────
ax = panel_at(5, slice(None)); ax.axis("off")
rows = [("memory wells", ["level pair; both below or both on the line; depth set by ⟨n₁⟩", "on the line, exactly; cannot descend", "antipodal; cannot both descend", "6 of 8 both below; the two failures keep a partner above"]),
        ("decision wells", ["one may stand alone on the κ₁ axis (the go well)", "a mirror pair across the line", "an antipodal pair", "a go well above; a nogo state at or below"]),
        ("trial trajectories", ["A and B trials are mirror images in κ₀ with the same κ₁ time course", "match and nonmatch, go and nogo, are mirror images in κ₁", "the B·nogo trial is minus the A·go trial", "no relation"]),
        ("behavior", ["acc(A) = acc(B) for every trial type", "misses = false alarms at the cue and at the test", "misses on A·go = false alarms on B·nogo", "the one-sided cost prices only licks: go and nogo accuracies unrelated"]),
        ("single units", ["as many A- as B-preferring units; memory and decision preferences independent", "as many lick- as no-lick-preferring units", "unit ρ(i) on B·nogo trials = unit i on A·go, sign-flipped", "lick-preferring units outnumber no-lick ones; ⟨n₁⟩ < 0"]),
        ("what the ledger shows", ["σ₁ residual stays 0.16–0.29 through Dual", "σ₃ residual rises to 0.6–0.8: broken hard", "σ₂ residual rises to 0.6–0.8: broken hard", "the inversion breaks in every seed; the pair exchange drifts"])]
x0 = [0.0, 0.15, 0.36, 0.57, 0.78]; ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for j, (t, sub, key) in enumerate(COLS): ax.text(x0[j + 1], 0.985, t, ha="left", va="top", fontsize=9.5, fontweight="bold")
for i, (name, cells) in enumerate(rows):
    y = 0.9 - i * 0.15; ax.text(0.0, y, textwrap.fill(name, 16), va="top", fontsize=9, fontweight="bold")
    for j, c in enumerate(cells): ax.text(x0[j + 1], y, textwrap.fill(c, 44), va="top", fontsize=7.6, color="0.2", linespacing=1.12)
    ax.plot([0, 1], [y + 0.02, y + 0.02], color="0.9", lw=0.6)
ax.text(0.0, 1.05, "G  the rest of what the group says about the Dual stage", transform=ax.transAxes, fontsize=11.5, fontweight="bold")
fig1.suptitle("The Dual task: one decision axis for two responses, the group it leaves, and what the breaking field can do", fontsize=13, y=0.985)
fig2.suptitle("The Dual task, continued: the inputs against the modes, and the rest of the predictions", fontsize=13, y=0.985)
fig1.savefig(out, bbox_inches="tight"); fig2.savefig(out2, bbox_inches="tight"); print("wrote", out, out2)
