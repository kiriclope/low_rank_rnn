#!/usr/bin/env python
"""
perturb_well_sweep.py — causal d'(well location) curve from ONE network by perturbing the rank-2
memory/decision modes, for each training stage (dpa / naive / expert).

The seed-scatter tuning curve (dpa_well_accuracy.py) is weak (R²≈0.2) because every seed differs in
many ways besides its well location. Here we hold ONE network fixed and slide its A/B memory wells
along κ₁ by adding a controlled κ₀→κ₁ coupling, re-measuring the well location and the DPA
match/nonmatch decision d' at each step. Location is then the ONLY variable → a clean causal curve.

Two perturbation modes (both antisymmetric: A and B move in opposite κ₁ directions, together they
sweep the whole axis):

  n1  : n₁ ← n₁ + ε·m̂₀   — tilts the decision READOUT toward the memory pattern. Moves the well AND
        redefines κ₁ (= rates·n₁/N), so d' mixes location with readout re-alignment. Full counterfactual.
  m0  : m₀ ← m₀ + ε·(n₁⊥n₀) — adds the κ₀→κ₁ coupling through the DYNAMICS only, along the part of n₁
        orthogonal to n₀ (so κ₀ memory depth and the readout n are left intact). d' then reflects the
        PURE location dependence of discriminability. Recommended for the causal read.

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python perturb_well_sweep.py \
      --sweep_dir results/dual/sweep_n1024_mem5 --run_id s0_n1024m5 \
      [--perturb n1 m0] [--stages dpa naive expert] [--eps -1.2 1.2 --n_eps 13]
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation_probe import load_run
from dpa_well_accuracy import dpa_stats
from plot_sweep import TIMINGS, _noise_sigma
from src.tasks import generate_dpa_trials

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
CA, CB = "#c0392b", "#2c6fbb"   # sample A / sample B
STAGE_TITLE = {"dpa": "after DPA", "naive": "after GNG (naive)", "expert": "after Dual (expert)"}


def _unit(v):
    n = torch.linalg.norm(v)
    return v / n if n > 0 else v


def _perturb_dirs(model, mode):
    """Return (param_tensor, col, base_col, direction) for the chosen perturbation. `direction` is
    scaled to the original column's norm so ε is a dimensionless fraction of that mode's magnitude."""
    m0 = model.m.detach()[:, 0].clone()
    n0, n1 = model.n.detach()[:, 0].clone(), model.n.detach()[:, 1].clone()
    if mode == "n1":                                   # readout+dynamics: n₁ += ε·m̂₀ (·|n₁|)
        return model.n, 1, n1, _unit(m0) * torch.linalg.norm(n1)
    if mode == "m0":                                   # dynamics only: m₀ += ε·(n₁ ⊥ n₀) (·|m₀|)
        perp = n1 - (torch.dot(n1, _unit(n0))) * _unit(n0)   # component of n₁ orthogonal to n₀
        return model.m, 0, m0, _unit(perp) * torch.linalg.norm(m0)
    raise ValueError(mode)


@torch.no_grad()
def sweep_stage(model, cfg, mode, eps_vals, n_trials, device):
    """Full-dynamics simulated sweep: physically perturb the weights at each ε, re-simulate, measure."""
    param, col, base, direction = _perturb_dirs(model, mode)
    recs = []
    for eps in eps_vals:
        param[:, col] = base + float(eps) * direction
        s = dpa_stats(model, cfg, n_trials=n_trials, device=device)
        recs.append((eps, s["A"], s["B"]))
    param[:, col] = base                               # restore
    return recs


@torch.no_grad()
def _dpa_measure(model, cfg, n_trials, device, inp_ch=None, inp_amp=0.0):
    """Run DPA trials (optionally adding a tonic drive inp_amp on input channel inp_ch from sample
    onset to trial end) and return per-sample well (κ0,κ1) over the delay + graded d'. WEIGHTS and
    PAIRING untouched — only the stimulus changes, so d' is the real trained network's discrimination."""
    t = TIMINGS["dpa"]
    on, off = [int(x) for x in t.n_stim_on], [int(x) for x in t.n_stim_off]
    torch.manual_seed(0)
    X, y = generate_dpa_trials(n_trials, timing=t, input_size=cfg["input_size"],
                               noise=_noise_sigma(cfg["noise"]),
                               attention_input=cfg.get("attention_input", False))
    if inp_ch is not None and inp_amp != 0.0:
        X[:, on[0]:, inp_ch] = X[:, on[0]:, inp_ch] + float(inp_amp)   # tonic drive from sample onset
    model.noise = 0.0
    k = model(X.to(device), y.to(device)).cpu().numpy()
    is_A = (X[:, on[0]:off[0], 0].mean(1) > X[:, on[0]:off[0], 1].mean(1)).numpy()
    is_match = y[:, -1, -1].numpy() > 0
    well, dec = k[:, off[0]:on[1], :].mean(1), k[:, off[1]:, 1].mean(1)
    out = {}
    for lab, mask in (("A", is_A), ("B", ~is_A)):
        p, n = dec[mask & is_match], dec[mask & ~is_match]
        sd = np.sqrt(0.5 * (p.var() + n.var()))
        out[lab] = dict(k0=float(well[mask, 0].mean()), k1=float(well[mask, 1].mean()),
                        dprime=float((p.mean() - n.mean()) / sd) if sd > 0 else float("nan"))
    return out


@torch.no_grad()
def sweep_stage_input(model, cfg, ch, amp_vals, n_trials, device):
    """Sweep the SAMPLE INPUT drive (tonic amp on channel `ch`) instead of the weights. Moves both
    wells' κ₁ together (symmetric — pairing preserved); the network is never modified."""
    return [(a, *(_dpa_measure(model, cfg, n_trials, device, inp_ch=ch, inp_amp=a)[g] for g in ("A", "B")))
            for a in amp_vals]


@torch.no_grad()
def analytical_curve(model, cfg, eps_dense, n_trials, device):
    """Closed-form d'(κ₁-location) for the READOUT-ROTATION counterfactual w = n₁ + ε·m₀. Simulate the
    UNPERTURBED network once, project post-test rates onto (n₁, m₀), and evaluate the LDA SNR
    d'(ε) = (Δμ_n + ε·Δμ_m)/√([1,ε]·V·[1,ε]ᵀ) with V = ½(Cov_match+Cov_nonmatch) — matching the
    simulated d' definition. Location κ₁(ε) = r̄_delay·w/N is linear in ε. No re-simulation → smooth."""
    t = TIMINGS["dpa"]
    on, off = [int(x) for x in t.n_stim_on], [int(x) for x in t.n_stim_off]
    torch.manual_seed(0)
    X, y = generate_dpa_trials(n_trials, timing=t, input_size=cfg["input_size"],
                               noise=_noise_sigma(cfg["noise"]),
                               attention_input=cfg.get("attention_input", False))
    model.noise = 0.0
    _, rates, _ = model(X.to(device), y.to(device), ret_rates=True)
    R = rates.cpu().numpy()
    N = R.shape[-1]
    n1 = model.n.detach().cpu().numpy()[:, 1]
    m0 = model.m.detach().cpu().numpy()[:, 0]
    is_A     = (X[:, on[0]:off[0], 0].mean(1) > X[:, on[0]:off[0], 1].mean(1)).numpy()
    is_match = y[:, -1, -1].numpy() > 0
    pn_del = R[:, off[0]:on[1], :].mean(1) @ n1 / N       # per-trial delay projection onto n₁
    pm_del = R[:, off[0]:on[1], :].mean(1) @ m0 / N       #                              onto m₀
    pn_pst = R[:, off[1]:, :].mean(1) @ n1 / N            # post-test decision projections
    pm_pst = R[:, off[1]:, :].mean(1) @ m0 / N
    e = np.asarray(eps_dense)
    out = {}
    for lab, mask in (("A", is_A), ("B", ~is_A)):
        mm, mn = mask & is_match, mask & ~is_match
        dmu = np.array([pn_pst[mm].mean() - pn_pst[mn].mean(),
                        pm_pst[mm].mean() - pm_pst[mn].mean()])
        V = 0.5 * (np.cov(np.stack([pn_pst[mm], pm_pst[mm]])) +
                   np.cov(np.stack([pn_pst[mn], pm_pst[mn]])))       # within-class 2×2
        q = V[0, 0] + 2 * e * V[0, 1] + e * e * V[1, 1]
        dp = (dmu[0] + e * dmu[1]) / np.sqrt(np.maximum(q, 1e-30))
        k1 = pn_del[mask].mean() + e * pm_del[mask].mean()
        out[lab] = (k1, dp)
    return out


def render(sweep_dir, rid, stages, mode, eps_vals, n_trials, out_root, device, input_channel=4):
    sweep = os.path.basename(os.path.normpath(sweep_dir))
    eps_dense = np.linspace(eps_vals.min(), eps_vals.max(), 400)   # smooth analytical grid
    fig, axes = plt.subplots(1, len(stages), figsize=(4.6 * len(stages), 4.6), squeeze=False)
    axes = axes[0]
    for ax, stage in zip(axes, stages):
        model, cfg = load_run(sweep_dir, rid, stage=stage, device=device)
        if mode == "input":
            recs = sweep_stage_input(model, cfg, input_channel, eps_vals, n_trials, device)
            ana = None
        else:
            recs = sweep_stage(model, cfg, mode, eps_vals, n_trials, device)
            ana  = analytical_curve(model, cfg, eps_dense, n_trials, device)   # readout-rotation
        e = np.array([r[0] for r in recs])
        for grp, col, lab in ((1, CA, "A"), (2, CB, "B")):         # A = idx 1, B = idx 2
            k1 = np.array([r[grp]["k1"] for r in recs])
            dp = np.array([r[grp]["dprime"] for r in recs])
            ax.scatter(k1, dp, c=col, s=26, edgecolor="k", linewidth=0.4, zorder=3,
                       label=f"sample {lab} (sim.)")
            if ana is not None:                                    # weight modes: closed-form overlay
                ak1, adp = ana[lab]
                o = np.argsort(ak1)
                ax.plot(ak1[o], adp[o], "-", color=col, lw=1.6, alpha=0.9, zorder=2)
            elif len(k1) >= 5:                                     # input mode: smooth poly thru points
                o = np.argsort(k1)
                coef = np.polyfit(k1, dp, min(4, len(k1) - 1))
                gx = np.linspace(k1.min(), k1.max(), 200)
                ax.plot(gx, np.polyval(coef, gx), "-", color=col, lw=1.6, alpha=0.9, zorder=2)
            i0 = int(np.argmin(np.abs(e)))                         # unperturbed ε≈0 marker
            ax.scatter(k1[i0], dp[i0], marker="*", s=190, c=col, edgecolor="k",
                       linewidth=0.6, zorder=4)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.axvspan(*sorted([ax.get_xlim()[0], 0]), color="0.85", alpha=0.35, zorder=0)  # no-lick κ1<0
        ax.axvline(0, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel(r"well location  $\kappa_1$  (decision / lick axis)")
        ax.set_title(STAGE_TITLE.get(stage, stage), fontsize=11)
    axes[0].set_ylabel(r"DPA decision $d'$  (match vs nonmatch)")
    mode_desc = {"n1": r"$n_1\!\leftarrow\!n_1+\epsilon\,\hat m_0$ (readout+dynamics)",
                 "m0": r"$m_0\!\leftarrow\!m_0+\epsilon\,(n_1\!\perp\!n_0)$ (dynamics only, readout fixed)",
                 "input": rf"tonic drive $\epsilon$ on input ch {input_channel} (weights & pairing intact)"}[mode]
    tail = ("points = full weight perturbation · line = analytic readout-rotation SNR · gap = dynamics"
            if mode != "input" else
            "wells moved by the STIMULUS only — network untouched · line = deg-4 fit · ★ = unperturbed")
    fig.suptitle(f"{sweep} · {rid} · perturb {mode}: {mode_desc}\n{tail}", fontsize=10, y=1.04)
    handles = [Line2D([0], [0], marker="o", ls="", color=CA, mec="k", label="sample A (sim.)"),
               Line2D([0], [0], marker="s", ls="", color=CB, mec="k", label="sample B (sim.)"),
               Line2D([0], [0], color="0.5", lw=1.6,
                      label="deg-4 fit" if mode == "input" else "analytic readout-rotation"),
               Line2D([0], [0], marker="*", ls="", color="0.4", mec="k", label="unperturbed")]
    axes[-1].legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_dir = os.path.join(out_root, sweep, "summary")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"perturb_well_{rid}_{mode}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {base}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_root", default="results/figures")
    ap.add_argument("--perturb", nargs="+", default=["input"], choices=["input", "n1", "m0"],
                    help="input = tonic sample-input drive (default, weights/pairing intact); "
                         "n1/m0 = weight perturbations")
    ap.add_argument("--stages", nargs="+", default=["dpa", "naive", "expert"],
                    choices=["dpa", "naive", "expert"])
    ap.add_argument("--input_channel", type=int, default=4, help="input channel for --perturb input (4=go)")
    ap.add_argument("--eps", nargs=2, type=float, default=None,
                    help="perturbation range (default: input ±2.5, weight modes ±1.2)")
    ap.add_argument("--n_eps", type=int, default=21)
    ap.add_argument("--n_trials", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for mode in args.perturb:
        lo, hi = args.eps if args.eps else ((-2.5, 2.5) if mode == "input" else (-1.2, 1.2))
        ctr = 0.8 if mode == "input" else 0.4                    # width of the fine central region
        eps_vals = np.unique(np.concatenate([                    # denser near κ₁≈0 (small ε)
            np.linspace(lo, hi, args.n_eps), np.linspace(-ctr, ctr, 11)]))
        print(f"[perturb {mode}]  stages={args.stages}  eps∈[{lo},{hi}] ({len(eps_vals)} pts)")
        render(args.sweep_dir, args.run_id, args.stages, mode, eps_vals,
               args.n_trials, args.out_root, args.device, input_channel=args.input_channel)


if __name__ == "__main__":
    main()
