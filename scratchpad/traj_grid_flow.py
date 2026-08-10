"""GENUINE simulated flow: seed the full network at a grid of initial κ, INTEGRATE real trajectories
(T steps of the actual two-timescale dynamics), and plot the paths — NOT a one-step adiabatic map.
Same 3×8 layout as the fp_stages figures (rows dpa/naive/expert, cols input conditions). Faint magma
= local speed for context; white = real trajectory paths; cyan = start, lime = settled endpoint. RO.

Usage: python scratchpad/traj_grid_flow.py [sweep] [run_id]   (default sweep_r2go s2_r2go10)
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt
from src.dynamics import make_input, _canonical_flow_panels

DEV = "cpu"
sweep = sys.argv[1] if len(sys.argv) > 1 else "sweep_r2go"
rid   = sys.argv[2] if len(sys.argv) > 2 else "s2_r2go10"
sd    = f"results/dual/{sweep}"
meta  = [m for m in _load_sweep_meta(sd) if m.run_id == rid][0]
XLIM, NG_BG, NGT, T = 2.0, 46, 11, 500
panels = _canonical_flow_panels(meta.cue_on_go_input, meta.cue_scale)
stages = ["dpa", "naive", "expert"]


def build_ff(model, panel):
    dt = next(model.parameters()).dtype
    ff = make_input(meta.input_size, panel["dims"], panel.get("value", 1.0), device=DEV, dtype=dt)
    if meta.attention_input:
        ff[-1] = getattr(meta, "attention_scale", 1.0)
    return ff


@torch.no_grad()
def seed(model, ff, kap):
    """Full network state on the κ-manifold: h = κ mᵀ, rates = φ(gain(drive+h))."""
    m = model.m.detach()
    h = kap.to(m.dtype) @ m.T
    ffb = ff[None, :].expand(kap.shape[0], -1).contiguous()
    drive = model.Ai * model.wi(ffb) if model.wi is not None else torch.zeros_like(h)
    rates = model.nonlinearity(model.gain * (drive + h))
    return h, rates, ffb


@torch.no_grad()
def field_1step(model, ff, GX, GY):
    n, N = model.n.detach(), model.hidden_size
    kap = torch.tensor(np.stack([GX.ravel(), GY.ravel()], -1), dtype=model.m.dtype)
    h, rates, ffb = seed(model, ff, kap)
    base = rates @ n / N
    rates2, _ = model.update_dynamics(ffb, h, rates)
    d = (rates2 @ n / N - base).cpu().numpy()
    return np.hypot(d[:, 0], d[:, 1]).reshape(GX.shape)


@torch.no_grad()
def integrate(model, ff, k0, T):
    n, N = model.n.detach(), model.hidden_size
    h, rates, ffb = seed(model, ff, k0)
    traj = [(rates @ n / N)]
    for _ in range(T):
        rates, h = model.update_dynamics(ffb, h, rates)
        traj.append(rates @ n / N)
    return torch.stack(traj, 1).cpu().numpy()      # (B, T+1, 2)


g = np.linspace(-XLIM, XLIM, NG_BG); GX, GY = np.meshgrid(g, g)
gt = np.linspace(-XLIM * 0.92, XLIM * 0.92, NGT); TX, TY = np.meshgrid(gt, gt)
k0 = torch.tensor(np.stack([TX.ravel(), TY.ravel()], -1), dtype=torch.float32)

cells, speeds = {}, []
for r, stage in enumerate(stages):
    model = _build_model(meta, DEV)
    if not _load_ckpt(model, sd, stage, rid, DEV):
        print("no ckpt", stage); continue
    for c, panel in enumerate(panels):
        ff = build_ff(model, panel)
        sp = field_1step(model, ff, GX, GY)
        trj = integrate(model, ff, k0, T)
        cells[(r, c)] = dict(sp=sp, trj=trj, name=panel["name"])
        speeds.append(sp.ravel())
    print("stage done:", stage)
vmax = float(np.percentile(np.concatenate(speeds), 97))

fig, axes = plt.subplots(3, 8, figsize=(30, 10.6))
for (r, c), cell in cells.items():
    ax = axes[r][c]
    ax.pcolormesh(GX, GY, cell["sp"], shading="auto", cmap="magma", vmax=vmax, alpha=0.85,
                  rasterized=True, zorder=0)
    trj = cell["trj"]
    for b in range(trj.shape[0]):
        ax.plot(trj[b, :, 0], trj[b, :, 1], color="white", lw=0.6, alpha=0.7, zorder=2)
        mid = trj.shape[1] // 12
        ax.annotate("", xy=trj[b, mid + 1], xytext=trj[b, mid],
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.6, alpha=0.7), zorder=2)
    ax.scatter(trj[:, 0, 0], trj[:, 0, 1], s=5, c="cyan", alpha=0.6, zorder=3)
    ax.scatter(trj[:, -1, 0], trj[:, -1, 1], s=22, c="lime", edgecolors="k", lw=0.4, zorder=4)
    ax.axhline(0, color="0.6", lw=0.4, zorder=1)
    ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-XLIM, XLIM); ax.set_aspect("equal")
    if r == 0: ax.set_title(cell["name"], fontsize=11)
    if c == 0: ax.set_ylabel(f"{stages[r]} (stage)\nκ1", fontsize=9)
    if r == 2: ax.set_xlabel("κ0")
fig.suptitle(f"{sweep} · {rid} — SIMULATED trajectory flow: full network integrated T={T} from an "
             f"{NGT}×{NGT} κ-grid of initial conditions (cyan=start, lime=settled). rows dpa/naive/expert.",
             fontsize=13)
fig.tight_layout()
p = f"/home/leon/.claude/jobs/29e2993a/tmp/trajflow_{rid}.png"
fig.savefig(p, dpi=105, bbox_inches="tight"); print("saved", p)
