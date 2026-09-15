"""PROTOCOL readout, tau=0.15 cells of the tau x noise grid. Per seed: every memory attractor
(|k0|>0.5) with kappa1 in sigma units, where the delay-end state lands, nearest attractor + distance,
deepest sub-line attractor + landing distance. Both probes: noise-free field, and the field/landings
UNDER THE TRAINED NOISE (rule 11)."""
import sys, numpy as np, torch, math, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/leon/rnn")
from bifurcation_probe import load_run, find_wells, run_dt_alpha
from src.tasks import make_timings
import traj_verdict as tv
from traj_verdict import kappa_from_rates
sw="results/dual/sweep_lif_sub_tau_noise"
ARMS=[("tau.15 n1.0","tau15_n10"),("tau.15 n1.5","tau15_n15")]
def acc(s,arm):
    for ln in open(f"{sw}/results.jsonl"):
        r=json.loads(ln)
        if r["run_id"]==f"s{s}_{arm}": return r["accuracy"]["after_gng"]["dpa"]
def probe(m,cfg,sig,noisy):
    mem=[f for f,t in find_wells(m,cfg,xlim=2.5,n_seeds=61,**({"noise_sigma":sig} if noisy else {}))
         if str(t).lower().startswith(("stable","attract")) and abs(f[0])>0.5]
    dt,_,_=run_dt_alpha(cfg); T=make_timings(dt)["dual"]; m.noise=sig if noisy else 0.0; torch.manual_seed(0)
    X,y,names=tv._gen("dual",cfg,T,768,0,sig if noisy else 0.0)
    with torch.no_grad(): _,r,_=m(X,y,ret_rates=True)
    k=kappa_from_rates(m,r).numpy(); i=int(T.n_stim_on[3])-1
    out=[]
    for pre in ("A","B"):
        msk=np.array([(("_go_" not in n and "_nogo_" not in n) and n.startswith(pre)) for n in names])
        st=np.array([k[msk,i,0].mean(),k[msk,i,1].mean()]); sd=k[msk,i,:].std(0)
        j=int(np.argmin([np.hypot(f[0]-st[0],f[1]-st[1]) for f in mem])) if mem else None
        out.append((pre,st,sd,mem[j] if mem else None,np.hypot(mem[j][0]-st[0],mem[j][1]-st[1]) if mem else float('nan')))
    deep=[f for f in mem if f[1]<0]; deepest=min(deep,key=lambda f:f[1]) if deep else None
    dd=min(np.hypot(deepest[0]-o[1][0],deepest[1]-o[1][1]) for o in out) if deepest is not None else float('nan')
    return mem,out,deepest,dd
for lab,arm in ARMS:
    for noisy in (False,True):
        print(f"=== {lab} — {'UNDER TRAINED NOISE' if noisy else 'noise-free field'} ===")
        for s in range(4):
            m,cfg=load_run(sw,f"s{s}_{arm}",stage="expert",device="cpu"); m.eval()
            dt,a,_=run_dt_alpha(cfg); sig=cfg["noise"]*math.sqrt(1-math.exp(-2*a))
            mem,out,deepest,dd=probe(m,cfg,sig,noisy)
            allk=" ".join(f"({f[0]:+.2f},{f[1]:+.2f}={f[1]/sig:+.1f}σ)" for f in sorted(mem,key=lambda f:f[1]))
            print(f" s{s}  ret {acc(s,arm):.3f}  σ {sig:.2f}  attractors: {allk if allk else 'none found'}")
            for pre,st,sd,f,d in out:
                if f is None: print(f"      {pre} lands ({st[0]:+.2f},{st[1]:+.2f}) ±({sd[0]:.2f},{sd[1]:.2f})"); continue
                print(f"      {pre} lands ({st[0]:+.2f},{st[1]:+.2f}) ±({sd[0]:.2f},{sd[1]:.2f}) -> occupies ({f[0]:+.2f},{f[1]:+.2f}) d={d:.2f}  depth {f[1]/sig:+.2f}σ {'BELOW' if f[1]<0 else 'above'}")
            if deepest is not None: print(f"      deepest sub-line ({deepest[0]:+.2f},{deepest[1]:+.2f}) = {deepest[1]/sig:+.1f}σ   landing distance {dd:.2f}")
        print()
