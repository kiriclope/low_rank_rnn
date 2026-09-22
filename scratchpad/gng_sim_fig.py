"""Figure 4 of the note: rows C and D of the GNG theory figure in the simulations — per column (σ₁ kept through
GNG, τ tied on GNG alone, σ₃ kept then released, free one-sided GNG): the autonomous field, the go-driven field,
the nogo-driven field (seed 0, other seeds' attractors overlaid) and the (m0, m1) populations, all at the checkpoint
after the GNG stage. Usage: gng_sim_fig.py <out.png>"""
import sys, os, glob, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/leon/rnn"); sys.path.insert(0, "/home/leon/rnn/scratchpad"); from symtools import *
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from bifurcation_probe import find_all_fixed_points, classify_fixed_points, autonomous_ff
out = sys.argv[1]; XL = (-1.5, 1.5); MK = ["o", "s", "D", "^", "v", "<", ">", "p"]
QC = {(1, 1): "#4c72b0", (-1, 1): "#dd8452", (-1, -1): "#55a868", (1, -1): "#c44e52"}
COLS = [("results/dual/sweep_lif_symdpa", "symdpa_pair", [0, 1, 2, 3], "σ₁ tied in DPA, released for GNG", "s1", "residual σ₁"),
        ("results/dual/sweep_lif_rulesym", "rulesym_afc_dec", [0, 1, 2, 3], "τ tied as diag(+1, −1), two-sided GNG alone", "s3", "residual τ (= σ₃)"),
        ("results/dual/sweep_lif_symdpa", "symdpa_test", [0, 1, 2, 3], "σ₃ tied in DPA, released for GNG", "s3", "residual σ₃"),
        ("results/dual/sweep_lif_recipe7", "recipe7", [0, 1, 2, 3], "free, one-sided GNG (the recipe)", None, "residuals σ₁ · σ₂ · σ₃")]
ROWS = [("auto", "autonomous field"), ("mn", "populations (m₀, m₁)")]   # DRIVEN=1 adds the go- and nogo-driven fields
if os.environ.get("DRIVEN", "0") == "1": ROWS = [ROWS[0], ("go", "go-driven field"), ("nogo", "nogo-driven field"), ROWS[1]]
def ff_for(cfg, kind):
    ff = autonomous_ff(cfg)
    if kind == "go": ff[4] = 1.0
    if kind == "nogo": ff[5] = 1.0
    return ff
def wells(model, cfg, ff, sig):
    fps, _ = find_all_fixed_points(model, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), ff_input=ff, n_seeds=41, noise_sigma=sig)
    stabs, _ = classify_fixed_points(model, fps, ff_input=ff, noise_sigma=sig)
    return [f for f, kd in zip(fps, stabs) if str(kd).lower().startswith(("stable", "attract")) and np.linalg.norm(f) > 0.15]
fig = plt.figure(figsize=(16, 4.4 * len(ROWS)), dpi=140); gs = fig.add_gridspec(len(ROWS), 4)
for c, (sw, arm, seeds, label, rkey, rlab) in enumerate(COLS):
    models = {s_: load_run(sw, f"s{s_}_{arm}", stage="naive", device="cpu") for s_ in seeds}
    m0, cfg = models[seeds[0]]; sig = sig_of(cfg)
    F_of, _ = field_fn(m0, sig); res = {s_: residuals(field_fn(models[s_][0], sig)[0], disk(21)) for s_ in seeds}
    for r, (kind, rowlab) in enumerate(ROWS):
        ax = fig.add_subplot(gs[r, c])
        if kind == "mn":
            M = m0.m.detach().numpy(); Nv = m0.n.detach().numpy(); N = len(M); J = Nv.T @ M / N
            axu = (np.abs(M[:, 0]) < 0.8) & (np.abs(M[:, 1]) > 2.5); lat = ~axu
            for q, col in QC.items():
                sel = lat & (np.sign(M[:, 0]) == q[0]) & (np.sign(M[:, 1]) == q[1]); ax.scatter(M[sel, 0], M[sel, 1], s=5, color=col, alpha=0.65, edgecolors="none")
            ax.scatter(M[axu, 0], M[axu, 1], s=6, color="0.45", alpha=0.8, edgecolors="none")
            ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8); ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect("equal")
            ax.set_title(f"seed {seeds[0]} · J₀₁ {J[0, 1]:+.2f}  J₁₀ {J[1, 0]:+.2f} · axis units {axu.sum()}", fontsize=8)
            ax.set_xlabel("m₀", fontsize=9); ax.set_ylabel(("m₁\n" + rowlab) if c == 0 else "m₁", fontsize=9); continue
        ff = ff_for(cfg, kind); ffx = torch.tensor(ff, dtype=torch.float32).view(1, 1, -1)
        cache, spd = _flow_panel_cache(m0, dict(name=rowlab, dims=None, conds=[]), cfg["input_size"], ffx, XL, XL, 81, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
        _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m0)
        ax.axhline(0, color="w", lw=0.9, ls=(0, (5, 4)), alpha=0.9)
        counts = []
        for s_ in seeds:
            w_ = wells(models[s_][0], models[s_][1], ff_for(models[s_][1], kind), sig); counts.append(len(w_))
            if s_ != seeds[0]:
                for f in w_: ax.plot(f[0], f[1], MK[s_ % 8], color="w", ms=5.5, mew=1.1, mfc="none", zorder=6)
        if rkey: rtxt = f"{rlab} (median of {len(seeds)}): {np.median([res[s_][rkey] for s_ in seeds]):.2f}"
        else: rtxt = f"{rlab}: " + " · ".join(f"{np.median([res[s_][k] for s_ in seeds]):.2f}" for k in ("s1", "s2", "s3"))
        ax.set_title((label + "\n" if r == 0 else "") + f"{rtxt}\nattractors {counts}", fontsize=8)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel("κ₀", fontsize=9); ax.set_ylabel(("κ₁\n" + rowlab) if c == 0 else "κ₁", fontsize=9)
        print(arm, kind, "attractors", counts, flush=True)
fig.suptitle("The GNG stage in the simulations, after GNG: the autonomous field and the populations (○ attractors of seed 0; white markers: the other seeds)" + (" — with the go- and nogo-driven fields" if len(ROWS) == 4 else ""), fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
