"""Fig. 4c of the dual project, for an RNN sweep: per seed and per sample (A/B), the change (expert − naive)
in the depth of the memory WELL — κ₁ of the attractor of the input-noise-averaged field (find_wells, rule 11) nearest
to where that sample's GNG-free DPA trials sit in the late delay (d ≤ 0.6; else NaN and the point is dropped) — against the
change in accuracy — left: DPA accuracy on DPA-only trials; right: NoGo accuracy on dual trials (the side the cue pushes toward the lick; go is at ceiling).
Circles = the two samples per seed, joined; ρ/p (Spearman) and the regression band on the per-seed means.
Usage: fig4c_rnn.py <sweep_dir> <arm> [out.png]   (trained input noise, recurrent 0)"""
import sys, os, math, numpy as np, torch, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/leon/rnn")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
from bifurcation_probe import load_run, run_dt_alpha, find_wells
from src.tasks import make_timings
import traj_verdict as tv
from traj_verdict import kappa_from_rates
sw, arm = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else f"results/figures/{os.path.basename(sw)}/summary/fig4c_depth_vs_accuracy.png"
def measure(seed, stage):
    m, cfg = load_run(sw, f"s{seed}_{arm}", stage=stage, device="cpu"); m.eval(); m.noise = 0.0
    dt, a, _ = run_dt_alpha(cfg); sig = cfg["noise"] * math.sqrt(1 - math.exp(-2 * a)); T = make_timings(dt)["dual"]
    torch.manual_seed(0); X, y, names = tv._gen("dual", cfg, T, 1024, 0, sig)
    with torch.no_grad(): _, r, _ = m(X, y, ret_rates=True)
    k = kappa_from_rates(m, r).numpy()[:, :, 1]; names = np.asarray(names)
    t_on, t_off = int(T.n_stim_on[3]), int(T.n_stim_off[3]); c_off = int(T.n_stim_off[2]); half = int(round(0.5 / T.dt)); one = int(round(1.0 / T.dt))
    late = k[:, t_on - one:t_on].mean(1); resp = k[:, t_off - half:t_off].mean(1); cue = k[:, c_off - half:c_off].mean(1)
    is_pair = np.array([(n[0] == "A" and n[-1] == "C") or (n[0] == "B" and n[-1] == "D") for n in names])
    none = np.array([("_go_" not in n and "_nogo_" not in n) for n in names]); go = np.array(["_go_" in n for n in names]); nogo = np.array(["_nogo_" in n for n in names])
    k0 = kappa_from_rates(m, r).numpy()[:, :, 0]; late0 = k0[:, t_on - one:t_on].mean(1)
    wells = [f for f, t in find_wells(m, cfg, xlim=2.5, n_seeds=61, noise_sigma=sig)
             if str(t).lower().startswith(("stable", "attract")) and abs(f[0]) > 0.5]
    res = {}
    for s in "AB":
        samp = np.array([n[0] == s for n in names])
        st = (late0[samp & none].mean(), late[samp & none].mean())          # where the memory state sits
        d = [math.hypot(f[0] - st[0], f[1] - st[1]) for f in wells]
        depth = wells[int(np.argmin(d))][1] if wells and min(d) <= 0.6 else float("nan")   # κ₁ of the occupied well (from the flow)
        dpa = ((resp > 0) == is_pair)[samp & none].mean()
        go_acc = (cue > 0)[samp & go].mean(); nogo_acc = (cue <= 0)[samp & nogo].mean()
        res[s] = (depth, dpa, nogo_acc, go_acc)      # right panel = NoGo (the side the cue pushes into the lick)
    return res
seeds = range(4); N = {s: measure(s, "naive") for s in seeds}; E = {s: measure(s, "expert") for s in seeds}
sig_eff = 0.373 * load_run(sw, f"s0_{arm}", stage="expert", device="cpu")[1]["noise"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), dpi=150)
for ax, j, ylab in ((axes[0], 1, "Δ DPA accuracy (DPA-only trials)"), (axes[1], 2, "Δ NoGo accuracy (dual trials)")):
    xm, ym = [], []
    for s in seeds:
        xs = [E[s][c][0] - N[s][c][0] for c in "AB"]; ys = [E[s][c][j] - N[s][c][j] for c in "AB"]
        if any(np.isnan(xs)):
            ok = [i for i in range(2) if not np.isnan(xs[i])]
            for i in ok: ax.scatter(xs[i], ys[i], s=48, facecolor="C0" if i == 0 else "white", edgecolor="C0", zorder=3, marker="s")
            if ok: xm.append(np.mean([xs[i] for i in ok])); ym.append(np.mean([ys[i] for i in ok]))
            continue
        ax.plot(xs, ys, "-", color="0.6", lw=1, zorder=1)
        ax.scatter(xs[0], ys[0], s=48, facecolor="C0", edgecolor="k", zorder=3, label="sample A" if s == 0 else None)
        ax.scatter(xs[1], ys[1], s=48, facecolor="white", edgecolor="C0", zorder=3, label="sample B" if s == 0 else None)
        ax.annotate(f"s{s}", (np.mean(xs), np.mean(ys)), fontsize=7, color="0.3", xytext=(3, 3), textcoords="offset points")
        xm.append(np.mean(xs)); ym.append(np.mean(ys))
    xm, ym = np.array(xm), np.array(ym)
    rho, p = spearmanr(xm, ym) if (len(xm) > 2 and np.ptp(ym) > 0 and np.ptp(xm) > 0) else (float("nan"), float("nan"))
    if len(xm) == 0: ax.text(0.5, 0.5, "no seed has an attractor at BOTH stages\n(naive geometry is a shelf: no well to measure)", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    if len(xm) > 2 and np.ptp(xm) > 0 and np.ptp(ym) > 0:
        lr = linregress(xm, ym); xx = np.linspace(xm.min(), xm.max(), 50); ax.plot(xx, lr.intercept + lr.slope * xx, "-", color="C3", lw=1.2, zorder=2)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    ax.set_xlabel("Δ well depth (flow: κ₁ of the occupied attractor)"); ax.set_ylabel(ylab)
    ax.set_title(f"Spearman ρ = {rho:+.2f}, p = {p:.2f}  (per-seed means, n = {len(xm)})", fontsize=9)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle(f"{arm}: Δ well depth vs Δ accuracy, expert − naive, per seed × sample\n(circles A / open B, joined per seed; squares: the other sample had no attractor at one stage; σ_eff = {sig_eff:.2f})", fontsize=9); fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
for s in seeds:
    for c in "AB": print(f"s{s} {c}: depth {N[s][c][0]:+.2f}→{E[s][c][0]:+.2f} (Δ{E[s][c][0]-N[s][c][0]:+.2f})  DPA {N[s][c][1]:.3f}→{E[s][c][1]:.3f}  NoGo {N[s][c][2]:.3f}→{E[s][c][2]:.3f}  Go {N[s][c][3]:.3f}→{E[s][c][3]:.3f}")
