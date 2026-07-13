#!/usr/bin/env python
"""
bifurcation_probe.py — self-gain + fixed-point ("well") probe for a low-rank sweep.

For each run in a sweep, load its expert checkpoint and report:
  • the mode self-gains  g·λ_r = gain · n_rᵀ m_r / N   (r=0 memory, r=1 decision)
    — g·λ_1 is the decision-mode pitchfork parameter (>1 bistable/ring, ≈1 critical/isolated)
  • the off-diagonal Jacobian overlaps g·n_0ᵀm_1/N, g·n_1ᵀm_0/N (mode coupling at the origin)
  • the AUTONOMOUS fixed points (attention on if the run used it), classified into
    attractor / saddle / repeller, with the attractor ("well") positions in the κ-plane
  • the after-Dual accuracies (dpa / dual_dpa / dual_go / dual_nogo) from results.jsonl

This is the "did it find the sweet spot?" table: 2 attractors at κ₁<0 with g·λ₁≈1 = isolated
low wells; 3+ wells with g·λ₁≳3 = the ring/U.

Only LowRankModel runs are probed (eistp/ei use ei_flow.py). Analytic reduced field, so it is
exact for LowRankModel and fast.

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python bifurcation_probe.py \
      --sweep_dir results/dual/sweep_gainscan [--run_ids s0_g05 s1_g05] [--xlim 2.5] [--csv out.csv]
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import LowRankModel
from src.dynamics import (low_rank_numpy_params, find_all_fixed_points,
                          classify_fixed_points, low_rank_jacobian_flow_np)


def discover_run_ids(sweep_dir):
    """Run ids = subdirs that contain an expert_<rid>.pth checkpoint."""
    rids = []
    for cfg in sorted(glob.glob(os.path.join(sweep_dir, "*", "config.json"))):
        rid = os.path.basename(os.path.dirname(cfg))
        if os.path.exists(os.path.join(sweep_dir, rid, f"expert_{rid}.pth")):
            rids.append(rid)
    return rids


def load_run(sweep_dir, rid, stage="expert", device="cpu"):
    """Rebuild a LowRankModel from <sweep_dir>/<rid>/config.json + load <stage>_<rid>.pth."""
    cfg = json.load(open(os.path.join(sweep_dir, rid, "config.json")))
    if cfg.get("model_type", "lowrank") != "lowrank":
        raise ValueError(f"{rid}: model_type={cfg.get('model_type')} not supported "
                         f"(use ei_flow.py for ei/eistp).")
    m = LowRankModel(
        input_size=cfg["input_size"], hidden_size=cfg["hidden_size"], output_size=0,
        rank=cfg.get("rank", 2), gain=cfg["gain"], alpha=0.075, alpha_rec=0.075, noise=0.0,
        nonlinearity=cfg["nonlinearity"], nl_gamma=cfg.get("nl_gamma", 0.0),
        use_unit_bias=cfg.get("use_unit_bias", False),
        unit_bias_trainable=cfg.get("unit_bias_trainable", False),
        use_rec_scale=cfg.get("use_rec_scale", False), device=device)
    sd = torch.load(os.path.join(sweep_dir, rid, f"{stage}_{rid}.pth"),
                    map_location=device, weights_only=True)
    miss, unexp = m.load_state_dict(sd, strict=False)
    bad = [k for k in miss if k != "gain"]
    if bad or unexp:
        raise RuntimeError(f"{rid}: checkpoint mismatch missing={bad} unexpected={unexp}")
    m.eval()
    return m, cfg


def self_gains(model):
    """2x2 origin Jacobian overlap matrix G[r,s] = gain · n_rᵀ m_s / N (before −I)."""
    p = low_rank_numpy_params(model)
    N = p["M"].shape[0]
    return p["gain"] * (p["Nvec"].T @ p["M"]) / N


def autonomous_ff(cfg):
    """Frozen input for the autonomous field: all zeros, but attention channel ON if used
    (matches the trained resting condition; makes the origin a fixed point)."""
    ff = np.zeros(cfg["input_size"], dtype=float)
    if cfg.get("attention_input", False):
        ff[-1] = 1.0
    return ff


def find_wells(model, cfg, xlim=2.5, n_seeds=41):
    """Autonomous fixed points + stability. Returns list of (kappa(2,), kind)."""
    ff = autonomous_ff(cfg)
    fps, _ = find_all_fixed_points(model, xlim=(-xlim, xlim), ylim=(-xlim, xlim),
                                   ff_input=ff, n_seeds=n_seeds)
    stabs, _ = classify_fixed_points(model, fps, ff_input=ff)
    return list(zip(fps, stabs))


def find_wells_brainpy(model, cfg, xlim=2.5, grid=13, slow_tol=1e-7, marg_tol=0.04, num_opt=2000):
    """Fixed/slow points of the autonomous reduced field via brainpy's SlowPointFinder
    (Adam gradient descent on the speed ½‖F(κ)‖² from a grid of candidates), classified with the
    exact analytic flow Jacobian. `slow_tol` is the max squared speed kept — raise it (e.g. 1e-3)
    to retain SLOW points that trace near-marginal manifolds (relu integrator, rings) that a
    root-finder misses. Returns list of (kappa(2,), kind). Requires jax + brainpy."""
    import jax, jax.numpy as jnp
    import brainpy as bp, brainpy.math as bm
    p = low_rank_numpy_params(model); N = p["M"].shape[0]
    M = jnp.asarray(p["M"]); Nv = jnp.asarray(p["Nvec"]); g = float(p["gain"])
    ff = autonomous_ff(cfg)
    drive = jnp.asarray(p["Ai"] * (ff @ p["Wi"].T + p["bi"]))
    phi = {"tanh": jnp.tanh, "relu": lambda x: jnp.maximum(x, 0.0),
           "erf": jax.scipy.special.erf, "softplus": jax.nn.softplus,
           "elu": jax.nn.elu}.get(cfg.get("nonlinearity", "tanh"), jnp.tanh)

    def f_cell(k):
        return phi(g * (drive[None, :] + k @ M.T)) @ Nv / N - k

    gx = np.linspace(-xlim, xlim, grid)
    cand = np.array([[a, b] for a in gx for b in gx], dtype=np.float32)
    finder = bp.analysis.SlowPointFinder(f_cell=f_cell, f_type="continuous")
    finder.find_fps_with_gd_method(
        candidates=bm.asarray(cand), tolerance=1e-9, num_batch=200, num_opt=num_opt,
        optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.05, 1, 0.9999)))
    finder.filter_loss(slow_tol)
    finder.keep_unique(tolerance=8e-2)
    MARG = marg_tol   # |Re eigenvalue| below this = a slow/marginal direction (matches scipy's label)
    out = []
    for fp in np.asarray(finder.fixed_points):
        ev = np.sort(np.linalg.eigvals(low_rank_jacobian_flow_np(p, fp, ff_input=ff)).real)
        lo, hi = ev[0], ev[-1]
        if   hi < -MARG:            kind = "attractor"   # both directions clearly attracting
        elif lo >  MARG:            kind = "repeller"
        elif hi > MARG and lo < -MARG: kind = "saddle"
        else:                       kind = "marginal"    # a near-zero (slow-manifold) direction
        out.append((np.asarray(fp, float), kind))
    return out


def read_accuracy(sweep_dir):
    """rid -> dict of after-Dual accuracy fields (missing keys tolerated)."""
    out = {}
    path = os.path.join(sweep_dir, "results.jsonl")
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("status") != "ok":
            continue
        a = (d.get("accuracy", {}) or {}).get("after_dual", {}) or {}
        out[d["run_id"]] = {k: a.get(k) for k in ("dpa", "dual_dpa", "dual_go", "dual_nogo")}
    return out


def _pick_wells(model, cfg, xlim, n_seeds, finder, slow_tol=1e-7, marg_tol=0.04):
    """brainpy (default) SlowPointFinder with scipy fallback."""
    if finder == "brainpy":
        try:
            return find_wells_brainpy(model, cfg, xlim=xlim, slow_tol=slow_tol, marg_tol=marg_tol)
        except Exception as e:
            print(f"[brainpy finder unavailable ({e}); falling back to scipy]")
    return find_wells(model, cfg, xlim=xlim, n_seeds=n_seeds)


def probe(sweep_dir, run_ids=None, xlim=2.5, n_seeds=41, finder="brainpy",
          slow_tol=1e-7, marg_tol=0.04):
    """Yield a dict of results per run."""
    run_ids = run_ids or discover_run_ids(sweep_dir)
    acc = read_accuracy(sweep_dir)
    for rid in run_ids:
        try:
            m, cfg = load_run(sweep_dir, rid)
        except Exception as e:
            yield {"run_id": rid, "error": str(e)}
            continue
        G = self_gains(m)
        # spectral abscissa of the reduced flow Jacobian at the origin (>0 = locally
        # unstable / spirals out; <0 = converges). Convergence readout for e.g. relu.
        Jf = low_rank_jacobian_flow_np(low_rank_numpy_params(m), np.zeros(2),
                                       ff_input=autonomous_ff(cfg))
        re0 = float(np.max(np.linalg.eigvals(Jf).real))
        wells = _pick_wells(m, cfg, xlim, n_seeds, finder, slow_tol, marg_tol)
        atts = [f for f, k in wells if k == "attractor"]
        yield {
            "run_id": rid, "gain": cfg["gain"],
            "gl0": float(G[0, 0]), "gl1": float(G[1, 1]),
            "off01": float(G[0, 1]), "off10": float(G[1, 0]),
            "re0": re0, "n_att": len(atts),
            "n_saddle": sum(k == "saddle" for _, k in wells),
            "n_marg": sum(k == "marginal" for _, k in wells),
            "wells": [(float(f[0]), float(f[1])) for f in atts],
            "acc": acc.get(rid, {}),
        }


def _fmt(x, w=5):
    return f"{x:.2f}".rjust(w) if isinstance(x, (int, float)) else "  -  "[:w].rjust(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--xlim", type=float, default=2.5,
                    help="fixed-point search half-window (widen for low-gain deep-memory runs)")
    ap.add_argument("--n_seeds", type=int, default=41)
    ap.add_argument("--finder", choices=["brainpy", "scipy"], default="brainpy",
                    help="fixed-point finder: brainpy SlowPointFinder (default) or scipy.root grid")
    ap.add_argument("--slow_tol", type=float, default=1e-7,
                    help="max squared speed ‖F‖² kept as a (slow) fixed point; raise to reveal slow manifolds")
    ap.add_argument("--marg", type=float, default=0.04,
                    help="|Re eigenvalue| below this → labeled 'marginal' (slow-manifold direction)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV path")
    args = ap.parse_args()

    rows = list(probe(args.sweep_dir, args.run_ids, args.xlim, args.n_seeds, args.finder,
                      args.slow_tol, args.marg))
    print(f"finder: {args.finder}")
    hdr = (f"{'run_id':10s} {'gain':>5s} {'gλ0':>6s} {'gλ1':>6s} {'off01':>6s} {'off10':>6s} "
           f"{'Re0':>6s} {'#att':>4s} {'#sad':>4s} {'#mrg':>4s}  {'wells (κ0,κ1)':32s} {'dpa':>5s} {'go':>5s} {'nogo':>5s}")
    print(hdr)
    print("-" * len(hdr))
    csv_lines = ["run_id,gain,gl0,gl1,off01,off10,n_att,n_saddle,dpa,dual_go,dual_nogo"]
    for r in rows:
        if "error" in r:
            print(f"{r['run_id']:10s}  ERROR: {r['error']}")
            continue
        a = r["acc"]
        wells = "  ".join(f"({x:+.1f},{y:+.1f})" for x, y in r["wells"]) or "—"
        print(f"{r['run_id']:10s} {r['gain']:5.2f} {r['gl0']:6.2f} {r['gl1']:6.2f} "
              f"{r['off01']:6.2f} {r['off10']:6.2f} {r['re0']:+6.3f} {r['n_att']:4d} {r['n_saddle']:4d} "
              f"{r.get('n_marg', 0):4d}  "
              f"{wells:32s} {_fmt(a.get('dual_dpa') or a.get('dpa'))} "
              f"{_fmt(a.get('dual_go'))} {_fmt(a.get('dual_nogo'))}")
        csv_lines.append(",".join(str(v) for v in [
            r["run_id"], r["gain"], f"{r['gl0']:.4f}", f"{r['gl1']:.4f}",
            f"{r['off01']:.4f}", f"{r['off10']:.4f}", r["n_att"], r["n_saddle"],
            a.get("dual_dpa") or a.get("dpa"), a.get("dual_go"), a.get("dual_nogo")]))
    if args.csv:
        open(args.csv, "w").write("\n".join(csv_lines) + "\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
