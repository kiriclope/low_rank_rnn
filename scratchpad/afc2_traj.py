"""κ(t) for the 2AFC runs (§39): L and R trials, individual traces + mean, and the κ-plane path.
Usage: afc2_traj.py <sweep_dir> <out.png> run_id [run_id ...]"""
import sys, os, json, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/leon/rnn"); from bifurcation_probe import load_run, run_dt_alpha
from src.tasks import make_timings, generate_gng_trials
sw, out, rids = sys.argv[1], sys.argv[2], sys.argv[3:]
fig, axes = plt.subplots(len(rids), 3, figsize=(13, 2.6 * len(rids)), squeeze=False)
for r, rid in enumerate(rids):
    m, cfg = load_run(sw, rid, stage="naive", device="cpu")
    dt, alpha, _ = run_dt_alpha(cfg); T = make_timings(dt)[cfg.get("rule_timing", "gng")]
    m.noise = float(cfg.get("model_noise", 0.0) * np.sqrt(1 - np.exp(-alpha) ** 2))
    torch.manual_seed(0)
    X, y = generate_gng_trials(64, T, cfg["input_size"], noise=cfg["noise"] * float(np.sqrt(1 - np.exp(-alpha) ** 2)), target_rank=2, cue_on_go_input=cfg["cue_on_go_input"],
        cue_scale=cfg["cue_scale"], nogo_target=cfg["nogo_target"], go_target=cfg["go_target"], go_on_rwd_input=cfg["go_on_rwd_input"],
        windowed_targets=cfg["windowed_targets"], decay_to_zero=cfg["decay_to_zero"], gng_response=cfg["gng_response"],
        response_in_cue=cfg["response_in_cue"], response_to_end=cfg.get("afc_response_to_end", False))
    with torch.no_grad(): k = m(X).numpy()
    isR = X[:, T.n_stim_on[0]:T.n_stim_off[0], 4].mean(1).numpy() > 0.5   # R on channel 4, L on 5
    t = np.arange(k.shape[1]) * dt
    for c, (lab, sel) in enumerate([("R", isR), ("L", ~isR)]):
        ax = axes[r, c]
        for i in np.where(sel)[0][:12]: ax.plot(t, k[i, :, 0], color="tab:blue", lw=0.5, alpha=0.4); ax.plot(t, k[i, :, 1], color="tab:red", lw=0.5, alpha=0.4)
        ax.plot(t, k[sel, :, 0].mean(0), color="tab:blue", lw=2, label="κ₀"); ax.plot(t, k[sel, :, 1].mean(0), color="tab:red", lw=2, label="κ₁")
        for a, b in zip(T.n_stim_on, T.n_stim_off): ax.axvspan(a * dt, b * dt, color="0.9")
        ax.axhline(0, color="0.5", lw=0.5); ax.set_ylim(-1.6, 1.6); ax.set_title(f"{rid}: {lab} trials", fontsize=9)
        if c == 0: ax.legend(fontsize=7, loc="upper left")
    ax = axes[r, 2]
    for i in range(24): ax.plot(k[i, :, 0], k[i, :, 1], color="tab:orange" if isR[i] else "tab:green", lw=0.6, alpha=0.6)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axhline(0, color="0.5", lw=0.5); ax.axvline(0, color="0.5", lw=0.5); ax.set_aspect("equal"); ax.set_title("κ-plane (orange R, green L)", fontsize=9)
    print(rid, "end-of-trial mean κ: R", k[isR, -1].mean(0).round(2), "L", k[~isR, -1].mean(0).round(2), "| end of delay: R", k[isR, T.n_stim_on[1]-1].mean(0).round(2), "L", k[~isR, T.n_stim_on[1]-1].mean(0).round(2), flush=True)
fig.tight_layout(); fig.savefig(out, dpi=110); print("wrote", out)
