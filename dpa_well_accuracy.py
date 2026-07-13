#!/usr/bin/env python
"""
dpa_well_accuracy.py — DPA accuracy vs memory-well location, split by sample (A vs B).

For each seed of a sweep it runs a batch of DPA trials, recovers the sample identity
(A = input channel 0, B = channel 1) from the first stimulus, and measures, per sample type:

  • WELL LOCATION — the memory state the trial settles into during the DELAY (mean κ over the
    delay window [n_stim_off[0] : n_stim_on[1]]), both κ₀ (memory axis, where A/B separate) and
    κ₁ (decision/lick axis — how low/no-lick the resting memory sits).
  • DPA ACCURACY — exactly the production metric (`sweep._dpa_accuracy`): the match/nonmatch
    decision read on κ₁ after the test offset, sign-compared to the ±1 target, restricted to that
    sample's trials.

Left panel plots accuracy vs κ₀-location; right panel vs κ₁-location. Each seed contributes one
A point and one B point per panel (faint line links a seed's A/B pair). This shows whether DPA
retention depends on where the A/B memory wells land in the κ-plane.

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python dpa_well_accuracy.py \
      --sweep_dir results/dual/sweep_n1024_mem5 --out_root results/figures [--n_trials 512]
"""
import argparse, os, sys
import numpy as np
import torch
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation_probe import load_run, discover_run_ids
from plot_sweep import TIMINGS, _noise_sigma
from src.tasks import generate_dpa_trials, generate_gng_trials

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 0.9})
CA, CB = "#c0392b", "#2c6fbb"   # sample A / sample B (DPA)  ·  go / nogo (GNG) colours


def _auc(pos, neg):
    """ROC-AUC = P(a random match trial's decision > a random nonmatch trial's). Parameter-free
    graded 'accuracy' (0.5=chance, 1=perfect separation); ties averaged via mid-ranks."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _overlap(a, b, bins=64):
    """Histogram overlap coefficient of two decision distributions ∈[0,1]: shared probability mass
    (1=identical/indistinguishable, 0=disjoint). error-region proxy the hard threshold ignores."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, edges, density=True)
    hb, _ = np.histogram(b, edges, density=True)
    return float(np.minimum(ha, hb).sum() * (edges[1] - edges[0]))


def _metrics(pos, neg):
    """Graded separation of two decision clouds: hard sign-acc, AUC, overlap, and (unsaturated) d'."""
    pooled_sd = float(np.sqrt(0.5 * (pos.var() + neg.var()))) if len(pos) and len(neg) else float("nan")
    return dict(auc=_auc(pos, neg), overlap=_overlap(pos, neg),
                dprime=float((pos.mean() - neg.mean()) / pooled_sd) if pooled_sd > 0 else float("nan"))


@torch.no_grad()
def dpa_stats(model, cfg, n_trials=512, device="cpu", seed=0):
    """Per-sample (A,B): mean well (κ0,κ1) over the DPA delay + graded match/nonmatch decision
    metrics. Decision variable = mean κ1 after the test; within each sample, match (+1) vs
    nonmatch (−1) form the two clouds. Accuracy is a property of each SAMPLE."""
    t = TIMINGS["dpa"]
    on, off = [int(x) for x in t.n_stim_on], [int(x) for x in t.n_stim_off]
    torch.manual_seed(seed)
    X, y = generate_dpa_trials(n_trials, timing=t, input_size=cfg["input_size"],
                               noise=_noise_sigma(cfg["noise"]),
                               attention_input=cfg.get("attention_input", False))
    model.noise = 0.0
    kap = model(X.to(device), y.to(device)).cpu().numpy()          # (T,n_steps,2): κ0,κ1
    is_A     = (X[:, on[0]:off[0], 0].mean(1) > X[:, on[0]:off[0], 1].mean(1)).numpy()
    well     = kap[:, off[0]:on[1], :].mean(1)                     # mean κ over the memory-hold window
    pred_dec = kap[:, off[1]:, 1].mean(1)                          # κ1 decision after test
    is_match = y[:, -1, -1].numpy() > 0

    out = {}
    for lab, mask in (("A", is_A), ("B", ~is_A)):
        out[lab] = dict(k0=float(well[mask, 0].mean()), k1=float(well[mask, 1].mean()),
                        k0_sd=float(well[mask, 0].std()), k1_sd=float(well[mask, 1].std()),
                        **_metrics(pred_dec[mask & is_match], pred_dec[mask & ~is_match]))
    out["_raw"] = dict(k0=well[:, 0], k1=well[:, 1], dec=pred_dec, pos=is_match)  # per-trial, for binning
    return out


@torch.no_grad()
def gng_stats(model, cfg, n_trials=512, device="cpu", seed=0):
    """Per-trial-type (go,nogo): mean well (κ0,κ1) over the pre-cue GNG delay + graded go/nogo
    decision metric. Decision variable = mean κ1 after the cue; go vs nogo ARE the two clouds, so
    the SAME separation (auc/overlap/d') is attached to both points (it is a seed-level score)."""
    t = TIMINGS["gng"]
    on, off = [int(x) for x in t.n_stim_on], [int(x) for x in t.n_stim_off]
    torch.manual_seed(seed)
    X, y = generate_gng_trials(n_trials, timing=t, input_size=cfg["input_size"],
                               noise=_noise_sigma(cfg["noise"]),
                               cue_on_go_input=cfg.get("cue_on_go_input", True),
                               cue_scale=cfg.get("cue_scale", 1.0),
                               nogo_target=cfg.get("nogo_target", 0.0),
                               attention_input=cfg.get("attention_input", False))
    model.noise = 0.0
    kap = model(X.to(device), y.to(device)).cpu().numpy()
    is_go    = (X[:, on[0]:off[0], 4].mean(1) > X[:, on[0]:off[0], 5].mean(1)).numpy()
    well     = kap[:, off[0]:on[1], :].mean(1)                     # mean κ over pre-cue delay
    pred_dec = kap[:, off[1]:, 1].mean(1)                          # κ1 decision after cue
    m = _metrics(pred_dec[is_go], pred_dec[~is_go])                # go vs nogo separation (seed-level)

    out = {}
    for lab, mask in (("go", is_go), ("nogo", ~is_go)):
        out[lab] = dict(k0=float(well[mask, 0].mean()), k1=float(well[mask, 1].mean()),
                        k0_sd=float(well[mask, 0].std()), k1_sd=float(well[mask, 1].std()), **m)
    out["_raw"] = dict(k0=well[:, 0], k1=well[:, 1], dec=pred_dec, pos=is_go)  # per-trial, for binning
    return out


def _bin_means(xs, ys, nbins=6, min_n=2):
    """Bin the per-seed points by location and return (bin_center, mean d', sem) for populated bins."""
    edges = np.linspace(xs.min(), xs.max(), nbins + 1)
    bx, by, be = [], [], []
    for i in range(nbins):
        mb = (xs >= edges[i]) & (xs <= edges[i + 1]) if i == nbins - 1 else \
             (xs >= edges[i]) & (xs < edges[i + 1])
        if mb.sum() >= min_n:
            bx.append(xs[mb].mean()); by.append(ys[mb].mean())
            be.append(ys[mb].std() / np.sqrt(mb.sum()))
    return np.array(bx), np.array(by), np.array(be)


def _add_tuning(ax, xs, ys, nbins=6, deg=3):
    """Overlay a tuning curve on the SAME per-seed points that are scattered on this panel: binned
    means (black, ±sem) + a low-order polynomial fit through the raw per-seed points, so the curve
    runs THROUGH the dots (not a separately-pooled binning on a different d' scale)."""
    if len(xs) < deg + 1:
        return
    bx, by, be = _bin_means(xs, ys, nbins=nbins)
    if len(bx):
        ax.errorbar(bx, by, yerr=be, fmt="o", color="k", ms=5, lw=0, ecolor="k",
                    elinewidth=1.0, capsize=2, zorder=5, label="binned mean $d'$")
    d = min(deg, len(xs) - 1)
    coef = np.polyfit(xs, ys, d)
    gx = np.linspace(xs.min(), xs.max(), 200)
    gy = np.polyval(coef, gx)
    xpk = gx[int(np.argmax(gy))]                                   # location of the fitted peak
    r2  = 1.0 - np.sum((ys - np.polyval(coef, xs)) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-12)
    ax.plot(gx, gy, "--", color="magenta", lw=1.7, zorder=6,
            label=r"deg-%d poly ($R^2$=%.2f, peak $\approx$%.2f)" % (d, r2, xpk))
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)


def _row_panels(axes, rows, task, g0, g1, ylab, title, legend_col=1):
    """Plot one task's row: d' vs κ0 location (left) and vs κ1 location (right); two trial-type
    groups (g0 marker=o, g1 marker=s) per seed, faint line linking the same-seed pair. The group
    legend goes on `legend_col` (the panel that does NOT get a binned/fit tuning curve overlay)."""
    for ax, key, xlab in ((axes[0], "k0", r"well location  $\kappa_0$  (memory axis)"),
                          (axes[1], "k1", r"well location  $\kappa_1$  (decision / lick axis)")):
        for _, st in rows:
            s = st[task]
            ax.plot([s[g0][key], s[g1][key]], [s[g0]["dprime"], s[g1]["dprime"]],
                    color="0.75", lw=0.6, zorder=1)
            ax.errorbar(s[g0][key], s[g0]["dprime"], xerr=s[g0][key + "_sd"], fmt="o",
                        color=CA, ms=7, mec="k", mew=0.5, ecolor="0.6", elinewidth=0.7, zorder=3)
            ax.errorbar(s[g1][key], s[g1]["dprime"], xerr=s[g1][key + "_sd"], fmt="s",
                        color=CB, ms=7, mec="k", mew=0.5, ecolor="0.6", elinewidth=0.7, zorder=3)
        if key == "k1":
            ax.axvspan(ax.get_xlim()[0], 0, color="0.85", alpha=0.4, zorder=0)  # no-lick half
            ax.axvline(0, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    axes[0].set_title(title, loc="left", fontsize=10.5)
    axes[legend_col].legend(handles=[Line2D([0], [0], marker="o", ls="", color=CA, mec="k", label=g0),
                            Line2D([0], [0], marker="s", ls="", color=CB, mec="k", label=g1),
                            Line2D([0], [0], color="0.75", lw=1.2, label="same seed")],
                   loc="lower left", fontsize=8, framealpha=0.9)


def render(sweep_dir, out_root, n_trials=512, device="cpu", poly_deg=3):
    sweep = os.path.basename(os.path.normpath(sweep_dir))
    rids = discover_run_ids(sweep_dir)
    rows = []
    hdr = f"{'run':<13}" + "".join(f"{c:>7}" for c in
          ("A_k0", "A_k1", "A_d'", "B_k0", "B_k1", "B_d'",
           "go_k0", "go_k1", "ng_k0", "ng_k1", "gng_auc", "gng_d'"))
    print(hdr)
    for rid in rids:
        try:
            m, cfg = load_run(sweep_dir, rid, stage="expert", device=device)
            d = dpa_stats(m, cfg, n_trials, device)
            g = gng_stats(m, cfg, n_trials, device)
            rows.append((rid, {"dpa": d, "gng": g}))
            A, B, go, ng = d["A"], d["B"], g["go"], g["nogo"]
            print(f"{rid:<13}" + "".join(f"{v:>7.2f}" for v in
                  (A['k0'], A['k1'], A['dprime'], B['k0'], B['k1'], B['dprime'],
                   go['k0'], go['k1'], ng['k0'], ng['k1'], go['auc'], go['dprime'])))
        except Exception as e:
            print(f"  {rid}: SKIP ({e})")
    if not rows:
        print("no runs"); return

    # y-axis = d' (separation in σ units). AUC & overlap SATURATE (1.0 / 0.0) once the two decision
    # clouds are disjoint, so they can't show a well-location gradient; d' keeps growing with margin.
    # Top row = DPA (match/nonmatch per sample A/B); bottom row = GNG (go vs nogo, seed-level score).
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.8))
    _row_panels(axes[0], rows, "dpa", "A", "B",
                r"DPA decision $d'$  (match vs nonmatch)",
                f"{sweep} · DPA — margin $d'$ vs well location  (AUC & overlap saturate 1.0 / 0.0)",
                legend_col=0)                          # tuning curve goes on the κ1 panel (col 1)
    _row_panels(axes[1], rows, "gng", "go", "nogo",
                r"GNG decision $d'$  (go vs nogo)",
                f"{sweep} · GNG — go/nogo margin $d'$ vs pre-cue well location",
                legend_col=1)                          # tuning curve goes on the κ0 panel (col 0)

    # Requested tuning curves — fit the SAME per-seed points scattered on each panel (both trial-type
    # groups), so the curve runs through the dots. x = each seed/group well location, y = its d'.
    def _pts(task, groups, key):
        xs = np.array([st[task][g][key]     for _, st in rows for g in groups])
        ys = np.array([st[task][g]["dprime"] for _, st in rows for g in groups])
        return xs, ys
    _add_tuning(axes[0][1], *_pts("dpa", ("A", "B"), "k1"), deg=poly_deg)      # DPA — κ1 panel
    _add_tuning(axes[1][0], *_pts("gng", ("go", "nogo"), "k0"), deg=poly_deg)  # GNG — κ0 panel
    fig.tight_layout()
    out_dir = os.path.join(out_root, sweep, "summary")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dpa_accuracy_vs_well")
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {base}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out_root", default="results/figures")
    ap.add_argument("--n_trials", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--poly_deg", type=int, default=3, help="degree of the tuning-curve polynomial fit")
    args = ap.parse_args()
    render(args.sweep_dir, args.out_root, args.n_trials, args.device, args.poly_deg)


if __name__ == "__main__":
    main()
