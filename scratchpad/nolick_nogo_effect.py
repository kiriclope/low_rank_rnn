"""The nogo no-lick hinge on the subcritical substrate: it repairs the rule, and the memory pays.
A - the subcritical DPA solution is born ENTANGLED (coupling == well tilt, two views of one thing)
B - the hinge recovers GNG in entanglement order, and costs DPA retention in the same order."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TILT = {0: 0.34, 1: 0.49, 2: 0.12, 3: 0.29}
CPL  = {0: -2.18, 1: -3.76, 2: 0.28, 3: -0.61}      # g*n0^T m1 at the DPA ckpt
def acc(sw, arm):
    d = {}
    for ln in open(f"results/dual/{sw}/results.jsonl"):
        r = json.loads(ln)
        if arm in r["run_id"]: d[int(r["run_id"][1])] = r["accuracy"]["after_gng"]
    return d
C, N = acc("sweep_r2sclc", "sclc2"), acc("sweep_r2sclnl", "sclnl")
order = sorted(TILT, key=lambda s: TILT[s])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.5))
axA.axhline(0.4, color="0.6", ls=":", lw=1); axA.axhline(-0.4, color="0.6", ls=":", lw=1)
axA.text(0.125, 0.55, "foundation stays inside ±0.4", fontsize=8, color="0.4")
for s in TILT:
    axA.scatter(TILT[s], CPL[s], s=110, c="#c0392b", edgecolor="k", lw=0.5, zorder=3)
    axA.annotate(f"s{s}", (TILT[s], CPL[s]), textcoords="offset points", xytext=(7, -3), fontsize=9)
axA.set_xlabel("memory-well tilt at the DPA solution   mean |κ₁|")
axA.set_ylabel("coupling  g·n₀ᵀm₁ / N   (DPA ckpt)")
axA.set_title("A · the subcritical DPA solution is born entangled\n(tilt and coupling are one thing, two views)", fontsize=10)
axA.grid(alpha=0.25)

x = np.arange(4); w = 0.36
axB.bar(x - w/2, [C[s]["gng"] for s in order], w, color="#6aafd4", edgecolor="k", lw=0.4, label="cue only — GNG")
axB.bar(x + w/2, [N[s]["gng"] for s in order], w, color="#1a6fa8", edgecolor="k", lw=0.4, label="+ no-lick nogo — GNG")
axB.plot(x - w/2, [C[s]["dpa"] for s in order], "o--", color="#e8867d", mec="k", ms=8, lw=1.4, label="cue only — DPA retention")
axB.plot(x + w/2, [N[s]["dpa"] for s in order], "s-",  color="#c0392b", mec="k", ms=8, lw=1.8, label="+ no-lick nogo — DPA retention")
for i, s in enumerate(order):
    axB.annotate(f"{N[s]['gng']-C[s]['gng']:+.2f}", (i, 1.012), ha="center", fontsize=8, color="#1a6fa8")
    axB.annotate(f"{N[s]['dpa']-C[s]['dpa']:+.2f}", (i + w/2, N[s]["dpa"]),
                 textcoords="offset points", xytext=(11, -4), fontsize=8, color="#c0392b")
axB.set_xticks(x); axB.set_xticklabels([f"s{s}\ntilt {TILT[s]:.2f}" for s in order])
axB.set_ylim(0.66, 1.06); axB.set_ylabel("accuracy after GNG")
axB.set_title("B · the hinge repairs the rule in entanglement order —\nand the memory pays in the same order", fontsize=10)
axB.legend(fontsize=8, loc="lower center", ncol=2, framealpha=0.95); axB.grid(alpha=0.25, axis="y")
fig.tight_layout()
out = "/home/leon/rnn/results/figures/sweep_r2sclnl/nolick_nogo_effect"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=130, bbox_inches="tight"); print("saved", out + ".png")
