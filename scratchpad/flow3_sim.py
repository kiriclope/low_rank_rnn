"""SIMULATION-based rank-3 flow slices + attractors — the ground-truth check for artifacts.

The analytic reader (wells3.py) root-finds the reduced field F(κ)=Ψ(κ)−κ (adiabatic; ignores W_fixed
and the two timescales). The worry: some of those fixed points may be artifacts of the reduction.

This script never root-finds the κ-map (that map is degenerate — at large κ the nonlinearity saturates
so Δκ→0 trivially, minting spurious FPs). Instead it finds attractors the only way that cannot lie:
long-integrate the REAL model (`update_dynamics`, incl. W_fixed + both timescales) from a dense κ-grid
of initial conditions and cluster the settled endpoints. A trajectory that settles somewhere IS a real
attractor. It then renders the three pairwise κ-slices (κ0κ1, κ0κ2, κ1κ2) with the true κ-flow field
(`_sim_step_single`, n_warmup=0 — well-behaved inside the plot box) and the settled cloud + cluster
centres overlaid, and compares each memory well's lick coord (κ2) to the analytic reader. Read-only.

Usage:  python scratchpad/flow3_sim.py <sweep> [run_id_substr]   e.g. flow3_sim.py sweep_r3o s0
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt, make_input
from src.dynamics import _sim_step_single
try:
    import wells3   # analytic cross-check (optional)
except Exception:
    wells3 = None

DEV  = "cpu"
KLAB = {0: "κ0 (sample)", 1: "κ1 (gng rule)", 2: "κ2 (lick)"}
PLANES = [(0, 1), (0, 2), (1, 2)]


def sim_step(model, ff, k, nw=0):
    return _sim_step_single(model, ff, np.asarray(k, float), n_warmup=nw)


@torch.no_grad()
def sim_settle(model, ff, box=2.2, ngrid=7, T=800):
    """Long-integrate the real model from a κ-grid of initial conditions; return settled κ (G, rank)."""
    rank, N = model.m.shape[1], model.hidden_size
    m, n = model.m, model.n
    axes = [torch.linspace(-box, box, ngrid, dtype=m.dtype)] * rank
    grid = torch.stack([g.reshape(-1) for g in torch.meshgrid(*axes, indexing="ij")], 1)   # (G, rank)
    fft  = torch.as_tensor(ff, dtype=m.dtype).unsqueeze(0).expand(grid.shape[0], -1).contiguous()
    h    = grid @ m.T
    drive = model.Ai * model.wi(fft) if model.wi is not None else torch.zeros_like(h)
    rates = model.nonlinearity(model.gain * (drive + h))
    for _ in range(T):
        rates, h = model.update_dynamics(fft, h, rates)
    return (rates @ n / N).cpu().numpy()


def cluster(pts, tol=0.12):
    centres, counts = [], []
    for p in pts:
        for i, c in enumerate(centres):
            if np.linalg.norm(p - c) < tol:
                centres[i] = (c * counts[i] + p) / (counts[i] + 1); counts[i] += 1; break
        else:
            centres.append(p.astype(float).copy()); counts.append(1)
    order = np.argsort(centres and [c[0] for c in centres] or [])
    return np.array(centres)[order], np.array(counts)[order]


def render(sweep, sub=None):
    sd = f"results/dual/{sweep}"
    metas = [m for m in _load_sweep_meta(sd) if (sub is None or sub in m.run_id)]
    metas = sorted(metas, key=lambda x: x.run_id)[:4]
    saved = []
    for meta in metas:
        model = _build_model(meta, DEV)
        if not _load_ckpt(model, sd, "expert", meta.run_id, DEV):
            print(f"{meta.run_id}: no expert ckpt"); continue
        dt = next(model.parameters()).dtype
        ff = make_input(meta.input_size, None, 1.0, device=DEV, dtype=dt)
        if meta.attention_input:
            ff[-1] = getattr(meta, "attention_scale", 1.0)
        ff = ff.detach().cpu().numpy().astype(np.float64)

        settled = sim_settle(model, ff)
        cen, cnt = cluster(settled)
        keep = cnt >= 2                                 # ignore singleton stragglers
        cen, cnt = cen[keep], cnt[keep]
        wells = [(c, n_) for c, n_ in zip(cen, cnt) if abs(c[0]) > 0.5]

        # analytic cross-check
        aw = None
        if wells3 is not None:
            try: aw = wells3.all_wells(meta, sd)
            except Exception: aw = None

        print(f"\n{meta.run_id}: {len(cen)} sim attractors ({len(wells)} memory wells). "
              f"SIM memory wells (κ0,κ1,κ2=lick, basin):")
        for c, n_ in wells:
            print(f"    κ0={c[0]:+.2f} κ1={c[1]:+.2f} κ2(lick)={c[2]:+.2f}   basin={n_}")
        if aw:
            print("  analytic wells (κ0,κ1,κ2):  " +
                  "  ".join("(" + ",".join(f"{v:+.2f}" for v in w) + ")" for w in aw))
        n_up = sum(c[2] > 0.05 for c, _ in wells)
        print(f"  → SIM: {n_up} memory wells with lick>0  (GOAL all-down = {n_up == 0 and len(wells) > 0})")

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.7))
        for ax, (dx, dy) in zip(axs, PLANES):
            dfx = ({0, 1, 2} - {dx, dy}).pop()
            df_val = float(np.median(cen[:, dfx])) if len(cen) else 0.0
            g = np.linspace(-2, 2, 22); GX, GY = np.meshgrid(g, g)
            U = np.zeros_like(GX); V = np.zeros_like(GX)
            for i in range(GX.shape[0]):
                for j in range(GX.shape[1]):
                    k = np.zeros(3); k[dx], k[dy], k[dfx] = GX[i, j], GY[i, j], df_val
                    d = sim_step(model, ff, k); U[i, j], V[i, j] = d[dx], d[dy]
            ax.streamplot(g, g, U, V, color=np.hypot(U, V), cmap="viridis",
                          density=1.1, linewidth=0.7, arrowsize=0.8)
            ax.scatter(settled[:, dx], settled[:, dy], s=7, c="orange", alpha=0.45, zorder=4,
                       label="settled (T=800)")
            for c, n_ in zip(cen, cnt):
                well = abs(c[0]) > 0.5
                ax.scatter(c[dx], c[dy], s=40 + 6 * n_, marker="*" if well else "o",
                           facecolors="red" if well else "deepskyblue", edgecolors="k",
                           linewidths=1.0, zorder=6)
            if dy == 2 or dx == 2:
                (ax.axhline if dy == 2 else ax.axvline)(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel(KLAB[dx]); ax.set_ylabel(KLAB[dy])
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
            ax.set_title(f"{KLAB[dx].split()[0]}–{KLAB[dy].split()[0]}  (slice κ{dfx}={df_val:+.2f})", fontsize=9)
        axs[0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{meta.run_id} — SIMULATION flow (real model, W_fixed + timescales) — "
                     f"★=memory well  ●=other attractor  orange=settled endpoints", fontsize=11)
        fig.tight_layout()
        p = f"/home/leon/.claude/jobs/29e2993a/tmp/simflow_{meta.run_id}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
        print("  saved", p); saved.append(p)
    return saved


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
