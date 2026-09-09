"""Is the cue reshaping the manifold rather than moving wells? (Leon 2026-09-03)
Quantify the SLOW SET (|F| < eps, exact analytic autonomous field) of each expert checkpoint
across the cue dose ladder, split by lick side; check transversal stability; overlay the actual
mean nogo trajectory (cue-on -> test-on) to see whether it travels along the slow structure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bifurcation_probe import load_run, run_dt_alpha, autonomous_ff
from src.flow_field import low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np, kappa_from_rates
from src.tasks import make_timings
from traj_verdict import _gen, _labels

SWEEPS = [("results/dual/sweep_r2nocue", "nocue", 0.0),
          ("results/dual/sweep_r2sgd2",  "sgd2",  2.0),
          ("results/dual/sweep_r2cue1",  "cue1",  1.0),
          ("results/dual/sweep_r2cue2",  "cue2",  2.0)]
LIM, N, EPS = 2.5, 301, 0.05
g = np.linspace(-LIM, LIM, N); GX, GY = np.meshgrid(g, g)
K = np.stack([GX, GY], -1)
cell = (2*LIM/(N-1))**2

print(f"slow set = |F| < {EPS} (exact field, attention on)  box ±{LIM}")
print(f"{'run':10s} {'area':>6s} {'%κ₁<0':>6s}  {'+κ₀ side κ₁-extent':>22s}  {'−κ₀ side κ₁-extent':>22s}")
fig, axes = plt.subplots(4, 4, figsize=(16.5, 15.5), squeeze=False)
for c, (sw, tag, dose) in enumerate(SWEEPS):
    for s in range(4):
        rid = f"s{s}_{tag}"
        m, cfg = load_run(sw, rid, stage="expert")
        p = low_rank_numpy_params(m); ff = autonomous_ff(cfg)
        F = np.asarray(low_rank_field_np(p, K, ff))
        sp = np.hypot(F[..., 0], F[..., 1])
        slow = sp < EPS
        area = slow.sum() * cell
        frac_dn = (slow & (GY < 0)).sum() / max(slow.sum(), 1)
        exts = []
        for sgn in (+1, -1):
            side = slow & (sgn*GX > 0.5)
            exts.append(f"[{GY[side].min():+.2f},{GY[side].max():+.2f}]" if side.any() else "  none  ")
        print(f"{rid:10s} {area:6.3f} {100*frac_dn:5.0f}%  {exts[0]:>22s}  {exts[1]:>22s}")
        # transversal stability at slow points (sample): eig ratio |λ_slow|/|λ_fast|, sign of fast
        pts = np.argwhere(slow)[::max(1, slow.sum()//40)]
        n_attr = 0; n_tot = 0
        for iy, ix in pts:
            J = np.asarray(low_rank_jacobian_flow_np(p, K[iy, ix][None], ff)).reshape(2, 2)
            ev = np.sort(np.linalg.eigvals(J).real)
            n_tot += 1; n_attr += (ev[0] < -0.2 and abs(ev[1]) < 0.15)
        # mean nogo trajectory on the DUAL task, cue-on -> test-on
        _, alpha, _ = run_dt_alpha(cfg); sig = float(np.sqrt(1-np.exp(-2*alpha)))
        T = make_timings(cfg["dt_base"]*cfg["tau_rec_frac"])["dual"]
        X, y, names = _gen("dual", cfg, T, 384, 0, cfg["noise"]*sig)
        lab = _labels("dual", cfg, T, X, names)
        m.noise = 0.0
        with torch.no_grad(): _, r, _ = m(X, y, ret_rates=True)
        kap = kappa_from_rates(m, r).numpy()
        cu, te = int(T.n_stim_on[2]), int(T.n_stim_on[3])
        ng = kap[lab["nogo"] & lab["A"], cu:te, :2].mean(0)   # A-side nogo mean path
        ax = axes[s][c]
        ax.pcolormesh(g, g, np.log10(sp + 1e-4), cmap="magma", vmin=-3, vmax=0.5, shading="auto", rasterized=True)
        ax.contour(g, g, sp, levels=[EPS], colors="cyan", linewidths=1.0)
        ax.plot(ng[:, 0], ng[:, 1], color="lime", lw=1.8)
        ax.scatter(ng[0, 0], ng[0, 1], c="white", s=25, zorder=5)
        ax.scatter(ng[-1, 0], ng[-1, 1], c="lime", edgecolors="k", s=45, zorder=5)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
        ax.set_title(f"{rid}  slow-area {area:.2f} ({100*frac_dn:.0f}% κ₁<0)  attr {n_attr}/{n_tot}", fontsize=9)
        if s == 3: ax.set_xlabel("κ₀")
        if c == 0: ax.set_ylabel("κ₁")
fig.suptitle("EXPERT autonomous speed |F| (log) · cyan = slow set |F|<0.05 · lime = mean A-nogo path cue-on→test-on\n"
             "columns: cue 0 / sgd2 (cue 2 + no-lick rule) / cue 1 / cue 2 (same DPA start per row-seed)", fontsize=11)
fig.tight_layout()
out = "/home/leon/rnn/results/figures/sweep_r2sgd2/slow_manifold_sgd2"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=110, bbox_inches="tight")
print("saved", out + ".png")
