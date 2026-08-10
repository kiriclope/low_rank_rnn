"""Rank-agnostic autonomous-well reader (works for rank-2 AND rank-3). Finds the autonomous fixed
points of the trained net in the FULL κ space (attention clamped on), classifies stability from the
reduced-field Jacobian, and lists every MEMORY well (|κ₀| large) with its full (κ₀, κ₁, κ₂...) —
reporting the LICK coordinate κ[-1] (κ₁ in rank-2, κ₂ in rank-3). GOAL: all memory wells at lick < 0
(no-lick plane). For rank-3 the sample (κ₀) and rule (κ₁) are held while the lick (κ₂) should rest
below the line autonomously (no cue ⇒ no lick). Read-only.
"""
import os, sys, collections
import numpy as np, torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
from scipy.optimize import root
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt, make_input
from src.dynamics import low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np

DEV = "cpu"
K0_WELL_FRAC = 0.4     # a memory well has |κ₀| ≥ this fraction of the run's κ₀ scale
LICK_DEAD    = 0.05    # |lick| < this ⇒ "on the line"


def all_wells(meta, sweep_dir, ngrid=6, box=2.2, noise_sigma=0.0):
    model = _build_model(meta, DEV)
    if not _load_ckpt(model, sweep_dir, "expert", meta.run_id, DEV):
        return None
    rank = model.m.shape[1]
    p    = low_rank_numpy_params(model)
    ff   = make_input(meta.input_size, None, 1.0, device=DEV, dtype=torch.float32)
    if meta.attention_input:
        ff[-1] = getattr(meta, "attention_scale", 1.0)
    ff = ff.detach().cpu().numpy().astype(np.float64)
    fun = lambda k: low_rank_field_np(p, k.reshape(1, -1), ff_input=ff[None, :], noise_sigma=noise_sigma).reshape(-1)
    jac = lambda k: low_rank_jacobian_flow_np(p, k.reshape(1, -1), ff_input=ff[None, :], noise_sigma=noise_sigma)
    # seed a coarse rank-D grid, root-find from each, dedup
    axes  = [np.linspace(-box, box, ngrid)] * rank
    seeds = np.stack([g.ravel() for g in np.meshgrid(*axes)], axis=1)
    fps = []
    for s in seeds:
        sol = root(fun, s, jac=jac, tol=1e-11)
        if sol.success and np.max(np.abs(fun(sol.x))) < 1e-7:
            if not any(np.linalg.norm(sol.x - f) < 5e-2 for f in fps):
                fps.append(sol.x)
    if not fps:
        return []
    k0_scale = max(float(np.max(np.abs(np.array(fps)[:, 0]))), 1.0)
    wells = []
    for f in fps:
        ev = np.linalg.eigvals(jac(f))
        stable = bool(np.all(ev.real < 5e-2))            # attractor / slow attractor (not saddle/repeller)
        if abs(f[0]) > K0_WELL_FRAC * k0_scale and stable:
            wells.append(tuple(float(x) for x in f))     # (κ0, κ1[, κ2])
    return sorted(wells, key=lambda w: w[0])


def _fmt(w):
    return " ".join("(" + ",".join(f"{v:+.2f}" for v in x) + ")" +
                    ("up" if x[-1] > LICK_DEAD else "dn" if x[-1] < -LICK_DEAD else "0") for x in w)


def _main(sweeps, noisy=True):
  for sweep in sweeps:
    sd = f"results/dual/{sweep}"
    metas = _load_sweep_meta(sd)
    rank = metas[0].rank if metas else 2
    tag_sig = f"; NOISE σ per-run (input-noise mean field, exact Gaussian resummation)" if noisy else ""
    print(f"\n== {sweep} :: autonomous memory wells (rank={rank}; lick = κ[-1]; "
          f"GOAL = all wells lick<{-LICK_DEAD:+.2f}{tag_sig}) ==")
    alldn = collections.defaultdict(int); ntot = collections.defaultdict(int); licks = collections.defaultdict(list)
    for m in sorted(metas, key=lambda x: x.run_id):
        sig = float(m.noise_sigma()) if noisy else 0.0
        try:
            w0 = all_wells(m, sd, noise_sigma=0.0)
            wn = all_wells(m, sd, noise_sigma=sig) if noisy else w0
        except Exception as e:
            print(f"  {m.run_id:14s} (failed: {type(e).__name__}: {e})"); continue
        if w0 is None:
            print(f"  {m.run_id:14s} (no ckpt)"); continue
        tag = m.run_id.split("_", 1)[1]; ntot[tag] += 1
        up  = [x for x in wn if x[-1] >  LICK_DEAD]
        if wn and not up: alldn[tag] += 1                 # GOAL evaluated on the NOISE-corrected wells
        licks[tag] += [x[-1] for x in wn]
        print(f"  {m.run_id:14s} clean [{len(w0)}w]  {_fmt(w0)}")
        if noisy:
            print(f"  {'':14s} σ={sig:.2f} [{len(wn)}w {len(up)}up]  {_fmt(wn)}")
    print("  " + "-" * 66)
    for tag in sorted(ntot):
        lk = np.array(licks[tag]) if licks[tag] else np.array([np.nan])
        print(f"  arm {tag:12s} ALL-lick-down (GOAL{' @noise' if noisy else ''}): {alldn[tag]}/{ntot[tag]}   "
              f"lick mean={np.nanmean(lk):+.3f}  up-frac={np.mean(lk>LICK_DEAD):.2f}  (n={len(licks[tag])})")


if __name__ == "__main__":
    _main(sys.argv[1:])
