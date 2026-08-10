"""Genuine simulated-trajectory κ-flow portraits — CLI. Integrates real network trajectories
from a grid of initial κ and plots the paths (rank-2 stages×conditions / rank-3 conditions×planes;
--noise = noisy trajectories). Implementation in src/flow_traj.py; primitive
integrate_kappa_trajectories in src/flow_field.py.

Usage:
  python traj_flow.py --sweep_dir results/dual/sweep_r2go --run_ids s2_r2go10 [--noise]
  python traj_flow.py --sweep_dir results/dual/sweep_r3o  --run_ids s0_r3o10 --stage expert [--noise]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.flow_traj import render_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--out_root", default="results/figures")
    ap.add_argument("--conditions", nargs="+",
                    default=["Autonomous", "A", "B", "Go", "NoGo", "Cue", "C", "D"])
    ap.add_argument("--stage", default="expert", choices=["dpa", "naive", "expert"],
                    help="rank-3 only: which stage to integrate (rank-2 always does all three)")
    ap.add_argument("--xlim", type=float, default=2.0)
    ap.add_argument("--T", type=int, default=500, help="integration steps per trajectory")
    ap.add_argument("--ngt", type=int, default=None, help="grid per axis (default 11 rank-2 / 6 rank-3)")
    ap.add_argument("--noise", action="store_true", help="inject the run's training input noise → NOISY trajectories")
    args = ap.parse_args()

    sweep = os.path.basename(os.path.normpath(args.sweep_dir))
    from plot_sweep import _load_sweep_meta as _lsm
    rids = args.run_ids or [m.run_id for m in _lsm(args.sweep_dir)]
    suff = ".noise" if args.noise else ""
    for rid in rids:
        meta = [m for m in _lsm(args.sweep_dir) if m.run_id == rid][0]
        ngt = args.ngt or (11 if meta.rank == 2 else 6)
        base = "traj_stages" if meta.rank == 2 else f"traj_{args.stage}"
        out = os.path.join(args.out_root, sweep, "individual", rid, "flow", base + suff)
        print(f"traj_flow: {rid} (rank {meta.rank}, noise={args.noise})")
        try:
            render_run(args.sweep_dir, rid, out, args.conditions, args.xlim, args.T, ngt, args.stage, args.noise)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {rid}: ERROR {e}")


if __name__ == "__main__":
    main()
