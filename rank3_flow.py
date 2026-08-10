"""Rank-3 flow-field portraits — CLI. Renders the three pairwise κ-plane slices (κ0κ1, κ0κ2, κ1κ2)
per input condition, with the shared rank-general FP finder (brainpy backend). Implementation lives in
`src/flow_rank3.py`; the shared engine/finder in `src/flow_field.py` + `src/flow_fixedpoints.py`.

Usage:
  python rank3_flow.py --sweep_dir results/dual/sweep_r3o --run_ids s0_r3o10 [--noise] [--stage expert]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation_probe import discover_run_ids
from src.flow_rank3 import render_run, ALL_CONDS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out_root", default="results/figures")
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDS, choices=ALL_CONDS,
                    help="dual input-driven conditions (rows); default = all 8")
    ap.add_argument("--xlim", type=float, default=3.0)
    ap.add_argument("--n_seeds", type=int, default=11, help="brainpy candidate grid per axis (n_seeds³)")
    ap.add_argument("--stage", default="expert", choices=["dpa", "naive", "expert"])
    ap.add_argument("--slice", dest="slice_mode", default="adiabatic",
                    choices=["adiabatic", "attractor", "zero"],
                    help="off-plane coord: 'adiabatic' (relax to its nullcline per grid point so "
                         "streamlines converge onto every FP — default), or a FLAT slice at the "
                         "median-attractor value ('attractor') or 0 ('zero')")
    ap.add_argument("--slow_tol", type=float, default=1e-7,
                    help="max squared speed ‖F‖² kept by the brainpy finder (raise e.g. 1e-3 to keep "
                         "slow / near-marginal / ring structure a root-finder misses)")
    ap.add_argument("--marg", type=float, default=0.04,
                    help="|Re eigenvalue| below this → 'marginal' (a slow-manifold direction)")
    ap.add_argument("--noise", action="store_true",
                    help="render the INPUT-NOISE mean field at the run's own σ (exact Gaussian "
                         "resummation φ(a/√(1+c·s²))) — shows which fixed points the noise destabilizes")
    args = ap.parse_args()

    sweep = os.path.basename(os.path.normpath(args.sweep_dir))
    rids = args.run_ids or discover_run_ids(args.sweep_dir)
    suffix = "_noise" if args.noise else ""
    print(f"rank3_flow: {sweep}  stage={args.stage}  conditions={args.conditions}  noise={args.noise}  runs={rids}")
    for rid in rids:
        out = os.path.join(args.out_root, sweep, "individual", rid, "flow", f"rank3_{args.stage}{suffix}")
        try:
            k = render_run(args.sweep_dir, rid, out, args.conditions, args.xlim, args.n_seeds,
                           args.stage, args.slice_mode, args.slow_tol, args.marg, use_run_noise=args.noise)
            print(f"  {rid}: {k} fixed points  ->  {out}.png")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {rid}: ERROR {e}")


if __name__ == "__main__":
    main()
