"""Genuine simulated-trajectory κ-flow portraits (rank-2 AND rank-3).

Unlike the analytic reduced field (plot_sweep / rank3_flow) or plot_sweep's --use_sim_field (a one-step
adiabatic map that collapses to ≈β·analytic), this INTEGRATES the full two-timescale network from a grid
of initial κ and plots the REAL trajectory paths — the honest "sim flow". Faint magma = analytic |field|
for context; white = integrated paths; cyan = start, lime = settled endpoint.

  rank-2: rows = stages (dpa/naive/expert) × cols = input conditions   (κ0–κ1 plane)   → traj_stages[.noise]
  rank-3: rows = input conditions × cols = 3 pairwise planes (one --stage)             → traj_<stage>[.noise]

--noise injects the run's own training input noise (σ = noise·√(1−e^{−α})²) each step → NOISY trajectories.

Usage:
  python traj_flow.py --sweep_dir results/dual/sweep_r2go --run_ids s2_r2go10 [--noise]
  python traj_flow.py --sweep_dir results/dual/sweep_r3o  --run_ids s0_r3o10 --stage expert [--noise]
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt
from src.flow_field import (make_input, _canonical_flow_panels, integrate_kappa_trajectories,
                            low_rank_numpy_params, low_rank_field_np)

KLAB = {0: "κ0", 1: "κ1", 2: "κ2"}
PLANES3 = [(0, 1), (0, 2), (1, 2)]


def run_noise_sigma(meta):
    a = getattr(meta, "alpha", None)
    if a is None:
        DT = meta.dt_base * meta.tau_rec_frac; a = DT / meta.tau
    return float(meta.noise * np.sqrt(1.0 - np.exp(-a) ** 2))


def build_ff(meta, panel):
    ff = make_input(meta.input_size, panel["dims"], panel.get("value", 1.0), dtype=torch.float32)
    if meta.attention_input:
        ff[-1] = getattr(meta, "attention_scale", 1.0)
    return ff.detach().cpu().numpy().astype(np.float64)


def bg_speed(p, ff_np, dx, dy, df_val, xlim, n=56):
    g = np.linspace(-xlim, xlim, n); GX, GY = np.meshgrid(g, g)
    rank = p["M"].shape[1]
    K = np.zeros((n, n, rank)); K[..., dx] = GX; K[..., dy] = GY
    if rank == 3:
        K[..., ({0, 1, 2} - {dx, dy}).pop()] = df_val
    F = low_rank_field_np(p, K, ff_input=ff_np[None, :])
    return g, np.hypot(F[..., dx], F[..., dy])


def draw(ax, g, sp, trj, dx, dy, vmax):
    ax.pcolormesh(g, g, sp, shading="auto", cmap="magma", vmax=vmax, alpha=0.85, rasterized=True, zorder=0)
    for b in range(trj.shape[0]):
        ax.plot(trj[b, :, dx], trj[b, :, dy], color="white", lw=0.55, alpha=0.7, zorder=2)
        k = max(1, trj.shape[1] // 12)
        ax.annotate("", xy=trj[b, k + 1, [dx, dy]], xytext=trj[b, k, [dx, dy]],
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.55, alpha=0.7), zorder=2)
    ax.scatter(trj[:, 0, dx], trj[:, 0, dy], s=5, c="cyan", alpha=0.6, zorder=3)
    ax.scatter(trj[:, -1, dx], trj[:, -1, dy], s=20, c="lime", edgecolors="k", lw=0.4, zorder=4)


def render_run(sweep_dir, rid, out_path, conditions, xlim, T, ngt, stage, noise):
    meta = [m for m in _load_sweep_meta(sweep_dir) if m.run_id == rid][0]
    rank = meta.rank
    sigma = run_noise_sigma(meta) if noise else 0.0
    panels = [p for p in _canonical_flow_panels(meta.cue_on_go_input, meta.cue_scale)
              if p["name"] in conditions]
    tag = f"  ·  NOISY trajectories σ={sigma:.2f}" if noise else ""

    if rank == 2:
        stages = ["dpa", "naive", "expert"]
        gt = np.linspace(-xlim * 0.92, xlim * 0.92, ngt); TX, TY = np.meshgrid(gt, gt)
        k0 = np.stack([TX.ravel(), TY.ravel()], -1)
        cells, speeds = {}, []
        for r, stg in enumerate(stages):
            model = _build_model(meta, "cpu")
            if not _load_ckpt(model, sweep_dir, stg, rid, "cpu"):
                continue
            p = low_rank_numpy_params(model)
            for c, pan in enumerate(panels):
                ff = build_ff(meta, pan)
                trj = integrate_kappa_trajectories(model, ff, k0, n_steps=T, noise_sigma=sigma,
                                                   record_every=max(1, T // 120))
                g, sp = bg_speed(p, ff, 0, 1, 0.0, xlim)
                cells[(r, c)] = (g, sp, trj, pan["name"]); speeds.append(sp.ravel())
            print("  stage done:", stg)
        vmax = float(np.percentile(np.concatenate(speeds), 97))
        nr, nc = len(stages), len(panels)
        fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc, 3.6 * nr), squeeze=False)
        for (r, c), (g, sp, trj, name) in cells.items():
            ax = axes[r][c]; draw(ax, g, sp, trj, 0, 1, vmax)
            ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim); ax.set_aspect("equal")
            ax.axhline(0, color="0.6", lw=0.4, zorder=1)
            if r == 0: ax.set_title(name, fontsize=10)
            if c == 0: ax.set_ylabel(f"{stages[r]}\nκ1", fontsize=9)
            if r == nr - 1: ax.set_xlabel("κ0")
        title = f"{os.path.basename(os.path.normpath(sweep_dir))} · {rid} — SIMULATED trajectory flow " \
                f"(integrated T={T} from {ngt}×{ngt} κ-grid; cyan=start lime=settled){tag}"
    else:  # rank-3: one stage, rows=conditions × cols=3 planes
        model = _build_model(meta, "cpu")
        if not _load_ckpt(model, sweep_dir, stage, rid, "cpu"):
            print("no ckpt", stage); return 0
        p = low_rank_numpy_params(model)
        ax1 = np.linspace(-xlim * 0.9, xlim * 0.9, ngt)
        G = np.stack([a.ravel() for a in np.meshgrid(ax1, ax1, ax1)], -1)   # (ngt³, 3)
        cells, speeds = {}, []
        for r, pan in enumerate(panels):
            ff = build_ff(meta, pan)
            trj = integrate_kappa_trajectories(model, ff, G, n_steps=T, noise_sigma=sigma,
                                               record_every=max(1, T // 120))
            settled = trj[:, -1, :]
            for c, (dx, dy) in enumerate(PLANES3):
                dfx = ({0, 1, 2} - {dx, dy}).pop()
                g, sp = bg_speed(p, ff, dx, dy, float(np.median(settled[:, dfx])), xlim)
                cells[(r, c)] = (g, sp, trj, dx, dy, pan["name"]); speeds.append(sp.ravel())
            print("  cond done:", pan["name"])
        vmax = float(np.percentile(np.concatenate(speeds), 97))
        nr, nc = len(panels), 3
        fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 4.0 * nr), squeeze=False)
        for (r, c), (g, sp, trj, dx, dy, name) in cells.items():
            ax = axes[r][c]; draw(ax, g, sp, trj, dx, dy, vmax)
            ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim); ax.set_aspect("equal")
            if dy == 2: ax.axhline(0, color="0.6", lw=0.5, ls="--", zorder=1)
            ax.set_xlabel(KLAB[dx]); ax.set_ylabel((f"{name}\n" if c == 0 else "") + KLAB[dy], fontsize=9)
        title = f"{os.path.basename(os.path.normpath(sweep_dir))} · {rid} · {stage} — SIMULATED trajectory " \
                f"flow (integrated T={T} from {ngt}³ κ-grid; cyan=start lime=settled){tag}"

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out_path}.{ext}", dpi=110, bbox_inches="tight")
    plt.close(fig); print("  saved", out_path + ".png")
    return 1


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
