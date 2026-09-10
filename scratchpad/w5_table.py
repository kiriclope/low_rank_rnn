"""The §28c table for the w5 (terminal A/B hold-window) arms, plus the new decay readout.

Per run, on the DPA checkpoint / DPA task:
  dpa acc · g·λ₀ (full overlap) · λ⁺ (positive-half overlap — the relu half-line self-gain)
  · measured asymptotic slope of F₀/κ₀ · the WELL TEST (does F₀/κ₀ cross + → − at finite κ₀)
  · κ₀ at sample-off / mid-delay / test-on and the decay ratio over the now-unsupervised delay
  · ⟨r²⟩ and max unit rate.
Usage: python scratchpad/w5_table.py [arm=sweep_dir ...]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from bifurcation_probe import load_run, autonomous_ff, run_dt_alpha
from src.flow_field import low_rank_numpy_params, low_rank_field_np
from src.tasks import make_timings
import traj_verdict as tv
from traj_verdict import kappa_from_rates

ARMS = [("nocue", "results/dual/sweep_r2nocue"), ("fdrelu", "results/dual/sweep_r2fdrelu"),
        ("rr01", "results/dual/sweep_r2rr01"),
        ("w5lif", "results/dual/sweep_r2w5lif"), ("w5relu", "results/dual/sweep_r2w5r"),
        ("w5rl2", "results/dual/sweep_r2w5r")]
if len(sys.argv) > 1:
    ARMS = [tuple(a.split("=", 1)) for a in sys.argv[1:]]

KS = np.logspace(np.log10(0.05), 2.2, 220)

def acc(sweep, rid):
    f = os.path.join(sweep, "results.jsonl")
    if not os.path.exists(f): return float("nan")
    for ln in open(f):
        r = json.loads(ln)
        if r.get("run_id") == rid:
            a = r.get("accuracy", r)
            for k in ("after_dpa", "dpa"):
                v = a.get(k) if isinstance(a, dict) else None
                if isinstance(v, dict): v = v.get("dpa", v.get("overall"))
                if v is not None: return float(v)
    return float("nan")

hdr = f"{'run':14s} {'dpa':>5s} {'g·λ0':>6s} {'λ+':>6s} {'slope∞':>7s} {'well? (κ0*)':>13s} " \
      f"{'F/κ@1':>6s} {'F/κ@3':>6s} " \
      f"{'κ0 3s':>7s} {'κ0 5.5s':>8s} {'κ0 test':>8s} {'ratio':>6s} {'⟨r²⟩':>8s} {'max r':>7s}"
print(hdr); print("-" * len(hdr))
for arm, sweep in ARMS:
    for s in range(4):
        rid = f"s{s}_{arm}"
        try:
            m, cfg = load_run(sweep, rid, stage="dpa", device="cpu")
        except Exception as e:
            print(f"{rid:14s} -- {type(e).__name__}: {str(e)[:60]}"); continue
        m.eval()
        N = cfg["hidden_size"]; g = cfg["gain"]
        m0 = m.m.detach()[:, 0].numpy(); n0 = m.n.detach()[:, 0].numpy()
        glam0 = g * float((n0 * m0).sum()) / N
        pos   = m0 > 0
        lamp  = g * float((n0[pos] * m0[pos]).sum()) / N

        p = low_rank_numpy_params(m); ff = autonomous_ff(cfg)
        F = np.asarray(low_rank_field_np(p, np.stack([KS, 0 * KS], -1), ff))
        prof = F[:, 0] / KS
        slope_inf = float(prof[-1])
        # WELL TEST: the largest κ₀ where the profile goes + → − (an attracting FP on the memory axis)
        sgn = np.sign(prof); cr = np.where((sgn[:-1] > 0) & (sgn[1:] < 0))[0]
        well = f"YES {KS[cr[-1]]:.2f}" if len(cr) else "no"
        at = lambda kv: float(prof[int(np.argmin(np.abs(KS - kv)))])
        p1, p3 = at(1.0), at(3.0)

        _, alpha, _ = run_dt_alpha(cfg); sig = float(np.sqrt(1 - np.exp(-2 * alpha)))
        dt, _, _ = run_dt_alpha(cfg); T = make_timings(dt)["dpa"]
        X, y, names = tv._gen("dpa", cfg, T, 256, 0, cfg["noise"] * sig)
        lb = tv._labels("dpa", cfg, T, X, names); m.noise = 0.0
        with torch.no_grad(): _, r, _ = m(X, y, ret_rates=True)
        k = kappa_from_rates(m, r).numpy(); rn = r.numpy()
        i = lambda t: min(int(round(t / T.dt)), k.shape[1] - 1)
        sep = lambda t: float(k[lb["A"], i(t), 0].mean() - k[lb["B"], i(t), 0].mean()) / 2
        k3, k55, k8 = sep(3.0), sep(5.5), float(T.stim_on[1])
        k8 = sep(k8 - T.dt)
        ratio = k8 / k3 if abs(k3) > 1e-9 else float("nan")
        r2 = float((rn ** 2).mean()); rmax = float(rn.max())
        print(f"{rid:14s} {acc(sweep, rid):5.2f} {glam0:6.2f} {lamp:6.2f} {slope_inf:7.2f} {well:>13s} "
              f"{p1:6.2f} {p3:6.2f} "
              f"{k3:7.2f} {k55:8.2f} {k8:8.2f} {ratio:6.2f} {r2:8.1f} {rmax:7.1f}")
    print()
