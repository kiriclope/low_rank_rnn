"""Target time courses per trial type — windowed_targets=True, decay_to_zero=True, gng_response=True.
3 tasks x 2 channels (kappa0 memory, kappa1 decision). One exemplar per condition (targets are
deterministic). NaN = free window -> line gap. Shaded spans = stimulus inputs."""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
from src.tasks import make_timings, generate_dpa_trials, generate_gng_trials, generate_dual_trials

DT = 0.0225
T = make_timings(DT)
KW_DPA = dict(input_size=7, target_rank=2, noise=0.0, input_scale=1.0, attention_input=True,
              windowed_targets=True, decay_to_zero=True)
KW = dict(**KW_DPA, cue_on_go_input=True, cue_scale=2.0, nogo_target=0.0)

torch.manual_seed(0)
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="row")
plt.rcParams.update({"font.size": 10})

def style(ax, timing, spans, title):
    for (a, b, lab) in spans:
        ax.axvspan(a, b, color="0.85", zorder=0)
        ax.text((a+b)/2, 1.55, lab, ha="center", va="bottom", fontsize=8, color="0.35")
    ax.axhline(0, color="0.7", lw=0.5, zorder=1)
    ax.set_ylim(-1.7, 1.9); ax.set_xlim(0, timing.t_steps)
    ax.set_title(title, fontsize=11)

def tt(y):  # time axis
    return np.arange(y.shape[0]) * DT

# ---------------- DPA ----------------
X, y = generate_dpa_trials(64, T["dpa"], **KW_DPA)
isA = X[:, int(2.5/DT), 0] > 0.5
mtc = torch.nan_to_num(y[:, :, 1], nan=0.0).sum(1) > 0    # match: pairing expr +1
a  = y[isA].numpy();  b = y[~isA].numpy()
m  = y[mtc].numpy();  nm = y[~mtc].numpy()
spans = [(2, 3, "sample"), (8, 9, "test")]
ax = axes[0, 0]
ax.plot(tt(a[0]), a[0][:, 0], color="tab:blue", lw=2, label="A")
ax.plot(tt(b[0]), b[0][:, 0], color="tab:red", lw=2, ls="--", label="B")
style(ax, T["dpa"], spans, "DPA — κ₀ (memory) target"); ax.legend(loc="lower right", fontsize=8)
ax = axes[0, 1]
ax.plot(tt(m[0]), m[0][:, 1], color="tab:green", lw=2, label="match (A→C/B→D)")
ax.plot(tt(nm[0]), nm[0][:, 1], color="tab:purple", lw=2, ls="--", label="nonmatch")
style(ax, T["dpa"], spans, "DPA — κ₁ (decision) target"); ax.legend(loc="lower left", fontsize=8)

# ---------------- GNG ----------------
X, y = generate_gng_trials(64, T["gng"], gng_response=True, **KW)
isgo = X[:, int(2.5/DT), 4] > 0.5
g  = y[isgo].numpy(); ng = y[~isgo].numpy()
spans = [(2, 3, "go/nogo stim"), (4, 4.5, "cue")]
ax = axes[1, 0]
ax.plot(tt(g[0]), g[0][:, 0], color="0.3", lw=2, label="all trials")
style(ax, T["gng"], spans, "GNG — κ₀ target (baseline only, then free)")
ax.legend(loc="lower right", fontsize=8)
ax = axes[1, 1]
ax.plot(tt(g[0]), g[0][:, 1], color="tab:green", lw=2, label="go")
ax.plot(tt(ng[0]), ng[0][:, 1], color="tab:red", lw=2, ls="--", label="nogo")
style(ax, T["gng"], spans, "GNG — κ₁ target (hold · rwd · decay)")
for x0, lab in [(3.5, "hold"), (4.5, "rwd"), (5.0, "decay")]:
    ax.annotate(lab, (x0 + 0.13, -1.45), fontsize=8, color="0.3")
ax.legend(loc="upper left", fontsize=8)

# ---------------- DUAL ----------------
X, y, _, names = generate_dual_trials(288, T["dual"], gng_response=True, **KW)
names = np.asarray(names).astype(str)
def pick(gng, match):
    for i, n in enumerate(names):
        parts = n.split("_")
        g = parts[1] if len(parts) == 3 else "none"
        is_m = (n[0] == "A" and n[-1] == "C") or (n[0] == "B" and n[-1] == "D")
        if g == gng and is_m == match:
            return y[i].numpy()
    raise ValueError
spans = [(2, 3, "sample"), (4, 5, "go/nogo"), (6, 6.5, "cue"), (8, 9, "test")]
ax = axes[2, 0]
ax.plot(tt(y[0].numpy()), y[0].numpy()[:, 0], color="0.3", lw=2, label="all trials")
style(ax, T["dual"], spans, "DUAL — κ₀ target (baseline only; memory free, frozen rank-0)")
ax.legend(loc="lower right", fontsize=8)
ax = axes[2, 1]
colors = {"go": "tab:green", "nogo": "tab:red", "none": "0.45"}
for gng in ["go", "nogo", "none"]:
    for match, ls in [(True, "-"), (False, "--")]:
        tr = pick(gng, match)
        lab = f"{gng} · {'match' if match else 'nonmatch'}"
        ax.plot(tt(tr), tr[:, 1], color=colors[gng], ls=ls, lw=1.8, alpha=0.9, label=lab)
style(ax, T["dual"], spans, "DUAL — κ₁ target (hold · rwd · decay · pair · decay)")
for x0, lab in [(5.5, "hold"), (6.6, "rwd"), (7.15, "decay"), (9.1, "pair"), (10.1, "decay")]:
    ax.annotate(lab, (x0, -1.45), fontsize=8, color="0.3")
ax.legend(loc="upper left", fontsize=7, ncol=2)

for ax in axes[-1]: ax.set_xlabel("time (s)")
fig.suptitle("Targets per trial type — windowed + decay_to_zero + gng_response "
             "(gaps = free/NaN; unified loss: ±1 hinge, 0 pin)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "results/figures/task_targets"
os.makedirs(out, exist_ok=True)
fig.savefig(f"{out}/targets_decay_rwd.png", dpi=150)
print("saved", f"{out}/targets_decay_rwd.png")
