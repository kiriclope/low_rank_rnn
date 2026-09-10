"""Pinning kappa1=0 through the DPA delay: it cuts the rank1->memory coupling, and retention follows.
A - the pin acts on n0^T m1 (rank1 fed back into memory), NOT on n1^T m0 (memory read by decision)
B - retention tracks the surviving |n0^T m1|: the one seed where it stayed >1 LOST the memory."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BEFORE = {0:(-2.18,0.87), 1:(-3.76,0.75), 2:(0.28,-0.44), 3:(-0.61,1.08)}   # scl16 DPA ckpt
AFTER  = {0:(-0.51,0.85), 1:(-2.56,0.65), 2:(0.28,-0.62), 3:(-0.08,0.82)}   # k1zero DPA ckpt
def acc(sw,arm):
    d={}
    for ln in open(f"results/dual/{sw}/results.jsonl"):
        r=json.loads(ln)
        if arm in r["run_id"]: d[int(r["run_id"][1])]=r["accuracy"]["after_gng"]["dpa"]
    return d
A, B = acc("sweep_r2sclf","sclf16"), acc("sweep_lif_sub_k1zero","k1zero")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.5))
x = np.arange(4); w = 0.35
axA.bar(x-w/2, [abs(BEFORE[s][0]) for s in range(4)], w, color="#e8867d", edgecolor="k", lw=.4, label="|n₀ᵀm₁|  free κ₁")
axA.bar(x+w/2, [abs(AFTER[s][0])  for s in range(4)], w, color="#c0392b", edgecolor="k", lw=.4, label="|n₀ᵀm₁|  κ₁ pinned")
axA.plot(x-w/2, [abs(BEFORE[s][1]) for s in range(4)], "o--", color="#6aafd4", mec="k", ms=7, label="|n₁ᵀm₀|  free κ₁")
axA.plot(x+w/2, [abs(AFTER[s][1])  for s in range(4)], "s-",  color="#1a6fa8", mec="k", ms=7, label="|n₁ᵀm₀|  κ₁ pinned")
axA.axhline(0.4, color="0.5", ls=":", lw=1.2); axA.text(-0.45, 0.47, "foundation stays under 0.4", fontsize=8, color="0.4")
axA.set_xticks(x); axA.set_xticklabels([f"s{s}" for s in range(4)])
axA.set_ylabel("|coupling| at the DPA ckpt"); axA.legend(fontsize=8, ncol=2)
axA.set_title("A · the pin cuts rank1→memory (n₀ᵀm₁),\nnot memory→decision (n₁ᵀm₀)", fontsize=10)
axA.grid(alpha=.25, axis="y")

for s in range(4):
    axB.plot([abs(BEFORE[s][0]), abs(AFTER[s][0])], [A[s], B[s]], "-", color="0.7", lw=1, zorder=1)
    axB.scatter(abs(BEFORE[s][0]), A[s], s=80, c="#e8867d", edgecolor="k", lw=.5, zorder=3)
    axB.scatter(abs(AFTER[s][0]),  B[s], s=110, marker="s", c="#c0392b", edgecolor="k", lw=.5, zorder=3)
    axB.annotate(f"s{s}", (abs(AFTER[s][0]), B[s]), textcoords="offset points", xytext=(8,-3), fontsize=9)
axB.axhspan(0.996, 1.0, color="#1a6fa8", alpha=.12)
axB.text(2.6, 0.975, "foundation 0.996–1.000", fontsize=8, color="#1a6fa8")
axB.scatter([], [], s=80, c="#e8867d", edgecolor="k", label="free κ₁ (sclf16)")
axB.scatter([], [], s=110, marker="s", c="#c0392b", edgecolor="k", label="κ₁ pinned (k1zero)")
axB.set_xlabel("|n₀ᵀm₁| at the DPA ckpt   (rank1 fed back into the memory)")
axB.set_ylabel("after_gng/dpa"); axB.legend(fontsize=8, loc="lower left")
axB.set_title("B · retention follows the surviving coupling —\ns1 kept −2.56 and LOST the memory", fontsize=10)
axB.grid(alpha=.25)
fig.tight_layout()
out="/home/leon/rnn/results/figures/sweep_lif_sub_k1zero/k1zero_effect"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out+".png", dpi=130, bbox_inches="tight"); print("saved", out+".png")
