"""Target time-courses for the CUE-DRIVEN arms (response_in_cue=True, decay_to_zero=False).
Pulls the real r2cue / r3cue RunConfigs from make_configs so the plotted windows/params are exactly
what training sees. One figure per rank; rows = tasks (DPA / GNG / Dual), cols = κ channels.
The readout is κ[-1]; the ±1 RESPONSE now sits in the LAST 0.5 s of its triggering stimulus (shaded),
and there is NO decay target after it (free ⇒ NaN ⇒ line gap = emergent relaxation). Read-only."""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.tasks import make_timings, generate_dpa_trials, generate_gng_trials, generate_dual_trials
from sweep import make_configs

cfgs = {c.run_id: c for c in make_configs("results/dual/_tmp")}
DT = cfgs["s0_r3cue"].dt_base * cfgs["s0_r3cue"].tau_rec_frac
T  = make_timings(DT)
def tt(y): return np.arange(y.shape[0]) * DT


def gen(c):
    """Generate one deterministic (noise=0) batch per task with the config's exact kwargs."""
    aI, aG, aS = c.attention_input, c.attention_gated, c.attention_scale
    Xd, yd = generate_dpa_trials(64, T["dpa"], input_size=c.input_size, target_rank=c.target_rank,
        noise=0.0, input_scale=c.input_scale, attention_input=aI, attention_gated=aG, attention_scale=aS,
        windowed_targets=c.windowed_targets, decay_to_zero=c.decay_to_zero, decay_onesided=c.decay_onesided,
        response_in_cue=c.response_in_cue)
    Xg, yg = generate_gng_trials(64, T["gng"], input_size=c.input_size, target_rank=c.target_rank,
        noise=0.0, cue_on_go_input=c.cue_on_go_input, cue_scale=c.cue_scale, nogo_target=c.nogo_target,
        go_target=c.go_target, go_on_rwd_input=c.go_on_rwd_input, input_scale=c.input_scale,
        attention_input=aI, attention_gated=aG, attention_scale=aS, ramping_gng=c.ramping_gng,
        windowed_targets=c.windowed_targets, decay_to_zero=c.decay_to_zero, gng_response=c.gng_response,
        decay_onesided=c.decay_onesided, response_in_cue=c.response_in_cue)
    Xu, yu, _, names = generate_dual_trials(288, T["dual"], input_size=c.input_size, target_rank=c.target_rank,
        noise=0.0, cue_on_go_input=c.cue_on_go_input, cue_scale=c.cue_scale, nogo_target=c.nogo_target,
        go_target=c.go_target, go_on_rwd_input=c.go_on_rwd_input, input_scale=c.input_scale,
        attention_input=aI, attention_gated=aG, attention_scale=aS, ramping_gng=c.ramping_gng,
        windowed_targets=c.windowed_targets, decay_to_zero=c.decay_to_zero, gng_response=c.gng_response,
        gng_memory=c.dual_gng_memory, decay_onesided=c.decay_onesided, response_in_cue=c.response_in_cue)
    return (Xd, yd), (Xg, yg), (Xu, yu, np.asarray(names).astype(str))


def style(ax, timing, spans, resp):
    for a, b, lab in spans:
        ax.axvspan(a, b, color="0.86", zorder=0)
        ax.text((a + b) / 2, 1.6, lab, ha="center", va="bottom", fontsize=7, color="0.4")
    for a, b in resp:      # response windows (last 0.5 s of the stimulus) — light green
        ax.axvspan(a, b, color="#c8e6c9", alpha=0.7, zorder=0)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylim(-1.75, 1.95); ax.set_xlim(0, timing.t_steps)


def plot_rank(rid, path):
    c = cfgs[rid]
    (Xd, yd), (Xg, yg), (Xu, yu, names) = gen(c)
    R = c.target_rank
    chan = ["κ0 memory", "κ1 rule", "κ2 lick (readout)"] if R == 3 else ["κ0 memory", "κ1 decision (rule+lick)"]
    fig, axes = plt.subplots(3, R, figsize=(5.2 * R, 9), squeeze=False)

    # ---- DPA ----
    isA = Xd[:, int(2.5 / DT), 0] > 0.5
    isC = Xd[:, int(8.5 / DT), 2] > 0.5
    match = (isA & isC) | (~isA & ~isC)
    exs = [("A·match", isA & match, "tab:blue", "-"), ("B·match", ~isA & match, "tab:red", "-"),
           ("A·nonmatch", isA & ~match, "tab:cyan", "--")]
    spans = [(2, 3, "sample"), (8, 9, "test")]
    resp  = [(8.5, 9.0)]        # last 0.5 s of test
    for k in range(R):
        ax = axes[0][k]
        for lab, mask, col, ls in exs:
            i = torch.where(mask)[0][0]; tr = yd[i].numpy()
            ax.plot(tt(tr), tr[:, k], color=col, ls=ls, lw=1.8, label=lab)
        style(ax, T["dpa"], spans, resp); ax.set_title(f"DPA — {chan[k]}", fontsize=10)
        if k == R - 1: ax.legend(fontsize=7, loc="lower left")

    # ---- GNG ----
    isgo = Xg[:, int(2.5 / DT), 4] > 0.5
    spans = [(2, 3, "go/nogo stim"), (4, 4.5, "resp cue")]
    resp  = [(4.0, 4.5)]        # last 0.5 s of cue
    for k in range(R):
        ax = axes[1][k]
        for lab, mask, col, ls in [("go", isgo, "tab:green", "-"), ("nogo", ~isgo, "tab:red", "--")]:
            i = torch.where(mask)[0][0]; tr = yg[i].numpy()
            ax.plot(tt(tr), tr[:, k], color=col, ls=ls, lw=1.8, label=lab)
        style(ax, T["gng"], spans, resp); ax.set_title(f"GNG — {chan[k]}", fontsize=10)
        if k == R - 1: ax.legend(fontsize=7, loc="upper left")

    # ---- DUAL ----
    def pick(gng, match):
        for i, n in enumerate(names):
            parts = n.split("_"); g = parts[1] if len(parts) == 3 else "none"
            is_m = (n[0] == "A" and n[-1] == "C") or (n[0] == "B" and n[-1] == "D")
            if g == gng and is_m == match: return yu[i].numpy()
        return None
    spans = [(2, 3, "sample"), (4, 5, "go/nogo"), (6, 6.5, "resp cue"), (8, 9, "test")]
    resp  = [(6.0, 6.5), (8.5, 9.0)]    # last 0.5 s of cue & of test
    exs = [("go·match", "go", True, "tab:green", "-"), ("nogo·match", "nogo", True, "tab:red", "--"),
           ("none·nonmatch", "none", False, "0.45", ":")]
    for k in range(R):
        ax = axes[2][k]
        for lab, gng, m, col, ls in exs:
            tr = pick(gng, m)
            if tr is not None: ax.plot(tt(tr), tr[:, k], color=col, ls=ls, lw=1.8, label=lab)
        style(ax, T["dual"], spans, resp); ax.set_title(f"DUAL — {chan[k]}", fontsize=10)
        if k == R - 1: ax.legend(fontsize=7, loc="upper left", ncol=1)

    for ax in axes[-1]: ax.set_xlabel("time (s)")
    fig.suptitle(f"{rid}  targets — response_in_cue=True, decay_to_zero=False  "
                 f"(green = response window = last 0.5 s of stimulus; gaps = free/NaN)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", path)


TMP = "/home/leon/.claude/jobs/29e2993a/tmp"
plot_rank("s0_r2cue", f"{TMP}/targets_r2cue.png")
plot_rank("s0_r3cue", f"{TMP}/targets_r3cue.png")
