"""w5: the A/B hold WINDOW (whole delay vs last 0.5 s) x phi, on the DPA stage: the memory-axis field profile F₀/κ₀ (well test = sign change + → −),
the κ₀ trajectory over the DPA delay, and the rate scale, for nocue / fdrelu / rr01 / rr1 / rrb01."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bifurcation_probe import load_run, autonomous_ff, run_dt_alpha
from src.flow_field import low_rank_numpy_params, low_rank_field_np
from src.tasks import make_timings
import traj_verdict as tv
from traj_verdict import kappa_from_rates
ARMS = [("nocue",  "results/dual/sweep_r2nocue", "Φ, foundation\nλ₀ init 3.0 (supercrit)"),
        ("w5lif",  "results/dual/sweep_r2w5lif", "Φ, terminal window\nλ₀ init 3.0 (supercrit)"),
        ("scl12",  "results/dual/sweep_r2scl",   "Φ, from BELOW\nλ₀ init 1.2"),
        ("scl16",  "results/dual/sweep_r2scl",   "Φ, from BELOW\nλ₀ init 1.6"),
        ("scl18",  "results/dual/sweep_r2scl",   "Φ, from BELOW\nλ₀ init 1.8")]
ks = np.logspace(np.log10(0.2), 2, 60)
fig, axes = plt.subplots(3, len(ARMS), figsize=(4.0 * len(ARMS), 10.5), squeeze=False)
for c, (a, sw, lab) in enumerate(ARMS):
    for s in range(4):
        m, cfg = load_run(sw, f"s{s}_{a}", stage="dpa", device="cpu"); m.eval()
        p = low_rank_numpy_params(m); ff = autonomous_ff(cfg)
        F = np.asarray(low_rank_field_np(p, np.stack([ks, 0 * ks], -1), ff)); prof = F[:, 0] / ks
        axes[0][c].plot(ks, prof, lw=1.6, label=f"s{s}")
        _, alpha, _ = run_dt_alpha(cfg); sig = float(np.sqrt(1 - np.exp(-2 * alpha)))
        T = make_timings(cfg.get("dt_base", 0.03) * cfg.get("tau_rec_frac", 0.75))["dpa"]
        X, y, names = tv._gen("dpa", cfg, T, 256, 0, cfg["noise"] * sig); lb = tv._labels("dpa", cfg, T, X, names); m.noise = 0.0
        with torch.no_grad(): _, r, _ = m(X, y, ret_rates=True)
        k = kappa_from_rates(m, r).numpy(); r = r.numpy(); t = np.arange(k.shape[1]) * T.dt
        axes[1][c].plot(t, k[lb["A"], :, 0].mean(0), lw=1.4); axes[1][c].plot(t, k[lb["B"], :, 0].mean(0), lw=1.4, ls="--")
        axes[2][c].plot(t, np.sqrt((r ** 2).mean(axis=(0, 2))), lw=1.4)
    ax = axes[0][c]; ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-1.05, 0.7)
    ax.set_title(f"{lab}\n{a}", fontsize=10); ax.set_xlabel("κ₀ (log)"); ax.set_ylabel("F₀/κ₀  (autonomous, κ₁=0)") if c == 0 else None
    if c == 0: ax.legend(fontsize=8, loc="lower left")
    ax = axes[1][c]
    _ymax = max(abs(l.get_ydata()).max() for l in ax.get_lines())
    if _ymax > 3: ax.set_yscale("symlog", linthresh=2)
    else: ax.set_ylim(-1.6, 1.6)
    ax.axhline(0, color="k", lw=0.6)
    for on, off in zip(T.stim_on, T.stim_off): ax.axvspan(on, off, color="0.85", zorder=0)
    ax.set_xlabel("t (s)"); ax.set_ylabel("κ₀  (A solid, B dashed; symlog)") if c == 0 else None
    ax = axes[2][c]; ax.set_yscale("log"); ax.set_ylim(0.1, 100)
    for on, off in zip(T.stim_on, T.stim_off): ax.axvspan(on, off, color="0.85", zorder=0)
    ax.set_xlabel("t (s)"); ax.set_ylabel("RMS rate over units (log)") if c == 0 else None
fig.suptitle("Subcritical init, Phi (lif) — pitchfork at lambda0 = 2.9; arms 3-5 START BELOW it. Memory-axis field profile (row 1: a well = sign change + → −; escape = stays > 0), κ₀ over the trial (row 2), rate scale (row 3)", fontsize=11)
fig.tight_layout()
out = "/home/leon/rnn/results/figures/sweep_r2scl/field_profiles_scl"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=110, bbox_inches="tight"); print("saved", out + ".png")
