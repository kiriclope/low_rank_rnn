"""Reduced model of the Dual loss as a function of the memory-well height w (kappa1 of the A/B wells),
with the response kicks FIXED (inputs are frozen in Dual) at values measured on the trained nets, and the
delay state fluctuating with sd s around w (input noise). Terms mirror UnifiedLoss with weight 1:
  DPA pairing (one-sided relu2, thresh 1):  paired  hinge(1 - (w + K))^2 ; unpaired hinge((w - K) + 1)^2
  go response at cue (one-sided, thresh 1): hinge(1 - (w + d + C))^2
  nogo response at cue (one-sided nolick):  shape(w - d + C)            [nolick_nogo_in_cue]
  no-lick over the delay on none rows:      shape(w)  (x fraction of steps)  [nolick_full_delay]
  no-lick on nogo rows from cue-on:         shape(w - d + C)
d = displacement of go(+)/nogo(-) rows from the well at cue time (0 with no hold and full decay;
the old absolute hold is NOT a displacement). Prints argmin w* and the curvature of each term."""
import numpy as np
rng=np.random.default_rng(0); xi=rng.standard_normal(20000)
relu2=lambda x: np.maximum(x,0)**2; softplus=lambda x: np.log1p(np.exp(x))
def L(w,K,C,d,s,shape,wn=1.0,pair=1.0,go=1.0,frac_delay=1.0):
    z=w+s*xi
    Ldpa = pair*(relu2(1-(z+K)).mean()+relu2((z-K)+1).mean())/2
    Lgo  = go*relu2(1-(z+d+C)).mean()
    Lnl  = wn*(shape(z).mean()*frac_delay + shape(z-d+C).mean())   # none-row delay + nogo-at-cue (split means: each full weight)
    return Ldpa,Lgo,Lnl
ws=np.linspace(-2.5,1.0,351)
for shape_name,shape in (("relu2",relu2),("softplus",softplus)):
    print(f"=== nolick shape {shape_name}  (K_test 0.8, C_cue 1.0, s 0.3; weights 1) ===")
    print(f"{'d':>4s} {'w*':>6s} {'L(w*)':>7s} | {'w* DPA-only':>11s} {'w* GNG-only':>11s} | {'w* if wn=2':>10s} {'w* if wn=4':>10s}")
    for d in (0.0,0.3,0.6,1.0):
        tot=[sum(L(w,0.8,1.0,d,0.3,shape)) for w in ws]; i=int(np.argmin(tot))
        dpa=[L(w,0.8,1.0,d,0.3,shape)[0] for w in ws]; gng=[L(w,0.8,1.0,d,0.3,shape)[1]+L(w,0.8,1.0,d,0.3,shape)[2] for w in ws]
        t2=[L(w,0.8,1.0,d,0.3,shape,wn=2.0) for w in ws]; t4=[L(w,0.8,1.0,d,0.3,shape,wn=4.0) for w in ws]
        w2=ws[int(np.argmin([sum(x) for x in t2]))]; w4=ws[int(np.argmin([sum(x) for x in t4]))]
        # DPA-only plateau: report the range within 1e-3 of its min
        dmin=min(dpa); plateau=ws[np.array(dpa)<=dmin+1e-3]
        print(f"{d:4.1f} {ws[i]:+6.2f} {tot[i]:7.3f} | [{plateau.min():+.2f},{plateau.max():+.2f}] {ws[int(np.argmin(gng))]:+11.2f} | {w2:+10.2f} {w4:+10.2f}")
    # the landscape itself at d=0.3 (a few points)
    d=0.3; print("  L(w) at d=0.3:", "  ".join(f"w{w:+.1f}:{sum(L(w,0.8,1.0,d,0.3,shape)):.2f}" for w in (-2,-1.5,-1,-0.5,0,0.5)))
