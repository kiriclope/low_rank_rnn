"""The finding: DPA-stage well TILT off the no-lick axis predicts the cue's damage to GNG.
Left  — memory wells in the kappa-plane, foundation (lam0 3.0) vs subcritical (lam0 1.6).
Right — tilt vs after_gng/gng WITH the cue on (cue2 / sclc2)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bifurcation_probe import load_run, find_wells

ARMS = [("nocue", "results/dual/sweep_r2nocue", "foundation  λ₀ init 3.0", "#3b6ea5", "o"),
        ("scl16", "results/dual/sweep_r2scl",   "subcritical λ₀ init 1.6", "#c0392b", "s")]
ACC = {}   # (arm, seed) -> after_gng/gng with the cue ON
for arm, sw in [("nocue", "results/dual/sweep_r2cue2"), ("scl16", "results/dual/sweep_r2sclc")]:
    for ln in open(f"{sw}/results.jsonl"):
        r = json.loads(ln); s = int(r["run_id"][1])
        ACC[(arm, s)] = (r.get("accuracy", {}).get("after_gng") or {}).get("gng", np.nan)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.6))
for arm, sw, lab, col, mk in ARMS:
    tilts, accs = [], []
    for s in range(4):
        m, cfg = load_run(sw, f"s{s}_{arm}", stage="dpa", device="cpu"); m.eval()
        w = [k for k, t in find_wells(m, cfg, xlim=2.5, n_seeds=41)
             if str(t).lower().startswith(("stable", "attract")) and abs(k[0]) > 0.5]
        if not w: continue
        axA.scatter([k[0] for k in w], [k[1] for k in w], c=col, marker=mk, s=52,
                    edgecolor="k", linewidth=0.4, zorder=3, label=lab if s == 0 else None)
        for k in w: axA.plot([k[0], k[0]], [0, k[1]], color=col, lw=0.8, alpha=0.5, zorder=2)
        tilts.append(np.mean(np.abs([k[1] for k in w]))); accs.append(ACC[(arm, s)])
    axB.scatter(tilts, accs, c=col, marker=mk, s=80, edgecolor="k", linewidth=0.5, label=lab, zorder=3)
    for t, a, s in zip(tilts, accs, range(4)):
        axB.annotate(f"s{s}", (t, a), textcoords="offset points", xytext=(6, -3), fontsize=8, color=col)

axA.axhline(0, color="k", lw=1.1); axA.axvline(0, color="0.7", lw=0.6)
axA.text(0.02, 0.02, "κ₁ = 0  (lick line)", transform=axA.transAxes, fontsize=8, color="0.35")
axA.set_xlabel("κ₀  (memory)"); axA.set_ylabel("κ₁  (decision)")
axA.set_title("A · memory wells at the DPA solution\n(stems = tilt off the lick line)", fontsize=10)
axA.legend(fontsize=8, loc="upper left"); axA.set_xlim(-1.6, 1.6); axA.set_ylim(-0.75, 0.75)

axB.set_xlabel("well tilt at the DPA solution   mean |κ₁|")
axB.set_ylabel("after_gng/gng   WITH cue (dose 2)")
axB.set_title("B · the tilt predicts the cue's damage\n(cue pushes κ₁ up on BOTH trial types)", fontsize=10)
axB.legend(fontsize=8, loc="lower left"); axB.grid(alpha=0.25)
fig.tight_layout()
out = "/home/leon/rnn/results/figures/sweep_r2sclc/tilt_vs_cue_damage"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=130, bbox_inches="tight"); print("saved", out + ".png")
