"""Pinning kappa1 during DPA makes the substrate CUE-ROBUST.
Both arms carry cue 2 and both load a DPA ckpt, so the RNG streams match (the k1zero->k1zcue
comparison does NOT: k1zero trained its DPA stage and consumed RNG, k1zcue loaded it)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TILT={0:0.34,1:0.49,2:0.12,3:0.29}; CPL={0:-0.51,1:-2.56,2:0.28,3:-0.08}
def A(sw,arm,key):
    d={}
    for ln in open(f"results/dual/{sw}/results.jsonl"):
        r=json.loads(ln)
        if arm in r["run_id"]: d[int(r["run_id"][1])]=r["accuracy"]["after_gng"][key]
    return d
S=("sweep_r2sclc","sclc2"); K=("sweep_lif_sub_k1zero_cue","k1zcue")
order=[2,3,0,1]; x=np.arange(4); w=0.36
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.8,4.6),sharex=True)
for ax,key,lo,ref in ((a1,"gng",0.65,(0.943,0.984)),(a2,"dpa",0.5,(0.987,1.000))):
    t=[A(*S,key)[s] for s in order]; p=[A(*K,key)[s] for s in order]
    ax.axhspan(*ref,color="#1a6fa8",alpha=.12,zorder=0)
    ax.text(3.45,ref[0]-0.055,"foundation\nwith cue",fontsize=7.5,color="#1a6fa8",ha="right")
    ax.bar(x-w/2,t,w,color="#e8867d",edgecolor="k",lw=.4,label="tilted wells (sclc2)")
    ax.bar(x+w/2,p,w,color="#c0392b",edgecolor="k",lw=.4,label="κ₁ pinned in DPA (k1zcue)")
    for i,s in enumerate(order):
        yy=max(t[i],p[i])+0.012
        if key=="dpa" and s==1: yy=min(t[i],p[i])-0.045
        ax.annotate(f"{p[i]-t[i]:+.2f}",(i,yy),ha="center",fontsize=8.5,
                    color="#1a6fa8" if p[i]>=t[i] else "#c0392b")
    ax.set_ylim(lo,1.06); ax.grid(alpha=.25,axis="y")
    ax.set_ylabel(f"after_gng/{key}")
    ax.set_title(("A · the rule survives the cue" if key=="gng" else "B · and so does the memory"),fontsize=11)
a1.legend(fontsize=8,loc="lower left")
a2.set_xticks(x)
a2.set_xticklabels([f"s{s}\ntilt {TILT[s]:.2f}→{'%.2f'%(0.02 if s==2 else 0.06 if s==3 else 0.11 if s==0 else 0.19)}\nn₀ᵀm₁ {CPL[s]:+.2f}" for s in order],fontsize=8)
a1.set_xticks(x); a1.set_xticklabels([f"s{s}" for s in order])
a2.annotate("n₀ᵀm₁ survived the pin\n(−2.56): memory LOST", (3.0,0.64),
            ha="center", fontsize=8, color="#c0392b")
fig.suptitle("κ₁ pinned through the DPA delay → cue-robust substrate (both arms carry cue 2, RNG-matched)",fontsize=11.5)
fig.tight_layout()
out="/home/leon/rnn/results/figures/sweep_lif_sub_k1zero_cue/k1zcue_effect"
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out+".png",dpi=130,bbox_inches="tight"); print("saved",out+".png")
