"""PROTOCOL readout for one arm vs its baseline: per seed, per stage (dpa ckpt, expert), every memory
attractor with kappa1 in sigma units, landings (delay end, under trained noise), nearest attractor,
deepest sub-line attractor + landing distance. Usage: readout_arm.py <sweep> <arm> [<sweep> <arm> ...]"""
import sys, numpy as np, torch, math, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/leon/rnn")
from bifurcation_probe import load_run, find_wells, run_dt_alpha
from src.tasks import make_timings
import traj_verdict as tv
from traj_verdict import kappa_from_rates
args=sys.argv[1:]; ARMS=list(zip(args[::2],args[1::2]))
def acc(sw,s,arm,key):
    for ln in open(f"{sw}/results.jsonl"):
        r=json.loads(ln)
        if r["run_id"]==f"s{s}_{arm}": return r["accuracy"][key]["dpa"]
def cpl(m):
    g=float(m.gain) if hasattr(m,"gain") else 1.0; N=m.m.shape[0]
    return g*float(m.n[:,0]@m.m[:,1])/N, g*float(m.n[:,1]@m.m[:,0])/N
for sw,arm in ARMS:
    for stage,task in (("dpa","dpa"),("expert","dual")):
        print(f"=== {arm} — {stage} ckpt ({task} trials, TRAINED noise = input noise σ={0:.2f}, recurrent 0; wells of the deterministic field) ===".format(0.373*cfg0["noise"]) if False else f"=== {arm} — {stage} ckpt ({task} trials; wells of the INPUT-noise-averaged field; landings: input noise σ as trained, recurrent 0) ===")
        for s in range(4):
            try: m,cfg=load_run(sw,f"s{s}_{arm}",stage=stage,device="cpu")
            except Exception as e: print(f" s{s} load failed: {e}"); continue
            m.eval(); dt,a,_=run_dt_alpha(cfg); sig=cfg["noise"]*math.sqrt(1-math.exp(-2*a))
            mem=[f for f,t in find_wells(m,cfg,xlim=2.5,n_seeds=61,noise_sigma=sig) if str(t).lower().startswith(("stable","attract")) and abs(f[0])>0.5]
            T=make_timings(dt)[task]; m.noise=0.0; torch.manual_seed(0)   # input noise sig in X (as trained); NO recurrent noise
            X,y,names=tv._gen(task,cfg,T,768,0,sig)
            with torch.no_grad(): _,r,_=m(X,y,ret_rates=True)
            k=kappa_from_rates(m,r).numpy(); i=int(T.n_stim_on[-1])-1
            c01,c10=cpl(m)
            allk=" ".join(f"({f[0]:+.2f},{f[1]:+.2f}={f[1]/sig:+.1f}σ)" for f in sorted(mem,key=lambda f:f[1]))
            print(f" s{s}  aDPA {acc(sw,s,arm,'after_dpa'):.3f} gng/dpa {acc(sw,s,arm,'after_gng'):.3f}  n0·m1 {c01:+.2f} n1·m0 {c10:+.2f}  attractors: {allk or 'none found'}")
            for pre in "AB":
                if task=="dual": msk=np.array([(("_go_" not in n and "_nogo_" not in n) and n.startswith(pre)) for n in names])
                else:
                    # inputs carry noise: identify A by the sample-window MEAN of (chan0 - chan1), not one step
                    a0,a1=int(T.n_stim_on[0]),int(T.n_stim_off[0]); isA=((X[:,a0:a1,0]-X[:,a0:a1,1]).mean(1)>0).numpy(); msk=isA if pre=="A" else ~isA
                st=np.array([k[msk,i,0].mean(),k[msk,i,1].mean()]); sd=k[msk,i,:].std(0)
                if mem:
                    d=[np.hypot(f[0]-st[0],f[1]-st[1]) for f in mem]; j=int(np.argmin(d))
                    print(f"      {pre} lands ({st[0]:+.2f},{st[1]:+.2f}) ±({sd[0]:.2f},{sd[1]:.2f}) -> nearest ({mem[j][0]:+.2f},{mem[j][1]:+.2f}) d={d[j]:.2f} depth {mem[j][1]/sig:+.2f}σ {'BELOW' if mem[j][1]<0 else 'above'}")
                else: print(f"      {pre} lands ({st[0]:+.2f},{st[1]:+.2f}) ±({sd[0]:.2f},{sd[1]:.2f})")
            deep=[f for f in mem if f[1]<0]
            if deep:
                dp=min(deep,key=lambda f:f[1]); print(f"      deepest sub-line ({dp[0]:+.2f},{dp[1]:+.2f}) = {dp[1]/sig:+.1f}σ")
        print()
