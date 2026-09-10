"""The 2x2: what the no-lick hinge COSTS depends entirely on how the memory was built.
All four cells carry cue 2 and load a DPA ckpt (RNG-matched). Seed s1 is excluded from the
means and drawn hollow: its rank1->memory coupling survived the pin (n0^T m1 = -2.56) and it
lost the memory before any of this."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CELLS=[("tilted wells\nno hinge","sweep_r2sclc","sclc2","#f2b8b2"),
       ("tilted wells\n+ no-lick nogo","sweep_r2sclnl","sclnl","#e8867d"),
       ("κ₁ pinned in DPA\nno hinge","sweep_lif_sub_k1zero_cue","k1zcue","#d0625a"),
       ("κ₁ pinned in DPA\n+ no-lick nogo","sweep_lif_sub_k1zero_cue_nolick_nogo","k1zcnl","#8e2b23")]
def A(sw,arm,k):
    d={}
    for ln in open(f"results/dual/{sw}/results.jsonl"):
        r=json.loads(ln)
        if arm in r["run_id"]: d[int(r["run_id"][1])]=r["accuracy"]["after_gng"][k]
    return d
fig,axes=plt.subplots(1,2,figsize=(12.4,4.8))
CLEAN=[0,2,3]
for ax,key,ref,lo in ((axes[0],"dpa",(0.987,1.000),0.55),(axes[1],"gng",(0.943,0.984),0.62)):
    ax.axhspan(*ref,color="#1a6fa8",alpha=.13,zorder=0)
    ax.text(3.42,ref[0]-0.055,"foundation\nwith cue",fontsize=7.5,color="#1a6fa8",ha="right")
    for i,(lbl,sw,arm,c) in enumerate(CELLS):
        v=A(sw,arm,key); m=np.mean([v[s] for s in CLEAN])
        ax.bar(i,m,0.6,color=c,edgecolor="k",lw=.5,zorder=2)
        ax.scatter([i]*3,[v[s] for s in CLEAN],s=34,c="k",zorder=4)
        ax.scatter([i],[v[1]],s=42,facecolor="none",edgecolor="0.45",lw=1.2,zorder=4)
        ax.annotate(f"{m:.3f}",(i,m+0.008),ha="center",fontsize=9,fontweight="bold",zorder=5)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in CELLS],fontsize=8.5)
    ax.set_ylim(lo,1.045); ax.grid(alpha=.25,axis="y")
    ax.set_ylabel(f"after_gng/{key}")
    ax.set_title(("A · MEMORY retention" if key=="dpa" else "B · GNG rule"),fontsize=11)
axes[0].annotate("",xy=(1,0.916),xytext=(0,0.938),arrowprops=dict(arrowstyle="->",color="#c0392b",lw=1.6))
axes[0].text(0.5,0.878,"hinge costs\n−0.022",ha="center",fontsize=8.5,color="#c0392b")
axes[0].annotate("",xy=(3,0.945),xytext=(2,0.945),arrowprops=dict(arrowstyle="->",color="#1a6fa8",lw=1.6))
axes[0].text(2.5,0.905,"hinge costs\nNOTHING",ha="center",fontsize=9,color="#1a6fa8",fontweight="bold")
fig.legend(handles=[plt.Line2D([],[],marker="o",ls="",color="k",ms=6,label="clean seeds (s0,s2,s3) — bar = their mean"),
                    plt.Line2D([],[],marker="o",ls="",mfc="none",mec="0.45",ms=7,label="s1 — coupling survived the pin, excluded")],
           loc="upper center",ncol=2,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,1.06))
fig.suptitle("What the no-lick rule costs depends on how the memory was built",fontsize=12.5,y=0.995)
fig.tight_layout()
out="/home/leon/rnn/results/figures/sweep_lif_sub_k1zero_cue_nolick_nogo/recipe_2x2"
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out+".png",dpi=130,bbox_inches="tight"); print("saved",out+".png")
