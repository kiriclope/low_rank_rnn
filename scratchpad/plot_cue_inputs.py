"""Input time-courses per task/trial-type — companion to plot_cue_targets.py (same style).
Pulls the real config so channels/scales/attention match training. noise=0 shown (clean signal);
the trained nets add σ·N(0,1) on EVERY channel each step. Rows = tasks, cols = exemplar trials;
one line per ACTIVE input channel. input_size=7: 0/1 sample A/B · 2/3 test C/D · 4 go+cue ·
5 nogo · 6 attention (tonic, gated to the retention delay). Read-only."""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.tasks import make_timings, generate_dpa_trials, generate_gng_trials, generate_dual_trials
from sweep import make_configs

c = {r.run_id: r for r in make_configs("results/dual/_tmp")}["s0_r3cue"]
DT = c.dt_base * c.tau_rec_frac
T  = make_timings(DT)
alpha = DT / c.tau
SIGMA = float(c.noise * np.sqrt(1.0 - np.exp(-alpha) ** 2))    # per-channel per-step input std used in training
def tt(y): return np.arange(y.shape[0]) * DT

CHAN = {0: ("sample A", "tab:blue"), 1: ("sample B", "tab:red"),
        2: ("test C", "tab:green"),  3: ("test D", "tab:purple"),
        4: ("go / cue", "tab:orange"), 5: ("nogo", "tab:brown"),
        6: ("attention", "0.5")}

akw = dict(target_rank=c.target_rank, attention_input=c.attention_input,
           attention_gated=c.attention_gated, attention_scale=c.attention_scale)
gkw = dict(cue_on_go_input=c.cue_on_go_input, cue_scale=c.cue_scale,
           go_on_rwd_input=c.go_on_rwd_input, input_scale=c.input_scale, **akw)

def pair(fn, T_, n, **kw):
    """Same RNG for both → identical trial identities; clean gives structure, noisy is what's plotted."""
    torch.manual_seed(0); a = fn(n, T_, input_size=c.input_size, noise=0.0,   **kw)
    torch.manual_seed(0); b = fn(n, T_, input_size=c.input_size, noise=SIGMA, **kw)
    return a, b

(Xdc, _), (Xdn, _)             = pair(generate_dpa_trials, T["dpa"], 64, input_scale=c.input_scale, **akw)
(Xgc, _), (Xgn, _)             = pair(generate_gng_trials, T["gng"], 64, **gkw)
(Xuc, *_, names), (Xun, *_)    = pair(generate_dual_trials, T["dual"], 288, **gkw)
names = np.asarray(names).astype(str)

fig, axes = plt.subplots(3, 3, figsize=(15.5, 9), sharey=True)
plt.rcParams.update({"font.size": 10})
seen = set()

def panel(ax, Xc, Xn, i, timing, spans, title):
    clean, noisy = Xc[i].numpy(), Xn[i].numpy()
    for ch in range(clean.shape[1]):
        if np.max(np.abs(clean[:, ch])) > 0.05:                # channel is active in this trial
            lab, col = CHAN[ch]
            ax.plot(tt(noisy), noisy[:, ch], color=col, lw=0.8, alpha=0.85,   # actual noisy input
                    label=(lab if lab not in seen else None))
            ax.plot(tt(clean), clean[:, ch], color=col, lw=2.2, alpha=0.35)   # clean signal reference
            seen.add(lab)
    for a, b, l in spans:
        ax.axvspan(a, b, color="0.9", zorder=0)
        ax.text((a + b) / 2, 3.15, l, ha="center", va="bottom", fontsize=7.5, color="0.4")
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.set_ylim(-1.4, 3.5); ax.set_xlim(0, timing.t_steps); ax.set_title(title, fontsize=10)

# ---- DPA ----
isA = Xdc[:, int(2.5/DT), 0] > 0.5; isC = Xdc[:, int(8.5/DT), 2] > 0.5
sp_d = [(2, 3, "sample"), (8, 9, "test")]
for k, (mask, ttl) in enumerate([(isA & isC, "DPA — A→C (match)"), (~isA & ~isC, "DPA — B→D (match)"),
                                  (isA & ~isC, "DPA — A→D (nonmatch)")]):
    panel(axes[0][k], Xdc, Xdn, int(torch.where(mask)[0][0]), T["dpa"], sp_d, ttl)

# ---- GNG ----
isgo = Xgc[:, int(2.5/DT), 4] > 0.5
sp_g = [(2, 3, "go/nogo stim"), (4, 4.5, "resp cue")]
for k, (mask, ttl) in enumerate([(isgo, "GNG — go"), (~isgo, "GNG — nogo")]):
    panel(axes[1][k], Xgc, Xgn, int(torch.where(mask)[0][0]), T["gng"], sp_g, ttl)
axes[1][2].axis("off")

# ---- DUAL ----
def pick(gng, match):
    for i, n in enumerate(names):
        parts = n.split("_"); g = parts[1] if len(parts) == 3 else "none"
        is_m = (n[0] == "A" and n[-1] == "C") or (n[0] == "B" and n[-1] == "D")
        if g == gng and is_m == match: return i
    return 0
sp_u = [(2, 3, "sample"), (4, 5, "go/nogo"), (6, 6.5, "resp cue"), (8, 9, "test")]
for k, (g, mt) in enumerate([("go", True), ("nogo", True), ("none", False)]):
    idx = pick(g, mt)
    s, gg, t = names[idx].split("_") if len(names[idx].split("_")) == 3 else (names[idx][0], "none", names[idx][-1])
    ttl = f"DUAL — {s}·{gg}·{t} ({'match' if mt else 'nonmatch'})"
    panel(axes[2][k], Xuc, Xun, idx, T["dual"], sp_u, ttl)

for ax in axes[-1]: ax.set_xlabel("time (s)")
for r in range(3): axes[r][0].set_ylabel("input amplitude")
fig.legend(loc="upper center", ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.005))
fig.suptitle(f"Inputs per trial type — ACTUAL noisy input (thin) + clean signal (thick faded); "
             f"σ={SIGMA:.2f}·N(0,1) on every channel each step (noise={c.noise}) — cue_scale={c.cue_scale}",
             fontsize=12, y=0.965)
fig.tight_layout(rect=(0, 0, 1, 0.94))
p = "/home/leon/.claude/jobs/29e2993a/tmp/task_inputs.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); print("saved", p)
