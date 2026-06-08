#!/usr/bin/env python3
"""
plot_mn_inputs.py — Visualize covariance structure of m, n, and input weights.

For each (run, stage), produces:
  mn_scatter.pdf          — m0,m1 vs n0,n1 scatter (2×2)
  inputs_vs_n.pdf         — all input channels vs n0,n1 (n_inputs × 2 grid)
  inputs_vs_m.pdf         — all input channels vs m0,m1 (n_inputs × 2 grid)
  corr_matrix.pdf         — Pearson-r heatmap: m0,m1, n0,n1, all inputs

Vectors are hidden-unit weight vectors of length N=512.
Correlations are Pearson r computed over those N units.

Usage:
    LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 \\
        python plot_mn_inputs.py \\
            --sweep_dir /home/leon/results/dual/sweep_random2 \\
            --run_id s0_random \\
            [--stages dpa naive expert] \\
            [--out_dir ./results/figures/sweep_random2/mn_inputs] \\
            [--device cuda:0]

If --run_id is omitted, runs for all runs in results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models import LowRankModel
from src.init   import init_dpa_internal_readout_prepost


# ---------------------------------------------------------------------------
# Loading helpers (mirrors plot_sweep.py)
# ---------------------------------------------------------------------------

def _load_all_meta(sweep_dir: str) -> list[dict]:
    rows = []
    path = os.path.join(sweep_dir, "results.jsonl")
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            rows.append(r)
    return rows


def _build_model(cfg: dict, device: str) -> LowRankModel:
    tau          = float(cfg.get("tau", 0.3))
    dt_base      = float(cfg.get("dt_base", 0.03))
    tau_rec_frac = float(cfg.get("tau_rec_frac", 0.75))
    dt           = dt_base * tau_rec_frac
    alpha        = dt / tau
    alpha_rec    = dt / (tau * tau_rec_frac)
    return LowRankModel(
        input_size  = int(cfg["input_size"]),
        hidden_size = 512,
        output_size = 0,
        rank        = 2,
        gain        = float(cfg.get("gain", 2.0)),
        alpha       = alpha,
        alpha_rec   = alpha_rec,
        noise       = 0.0,
        rwd         = bool(cfg.get("rwd", True)),
        rwd_scale   = float(cfg.get("rwd_scale", 1.0)),
        device      = device,
    )


def _build_init_model(cfg: dict, device: str) -> LowRankModel:
    """Reproduce the model state right after initialization (before any training).

    Mirrors run_single: set seeds, build model, apply structured init if requested.
    The seed is set once before model construction so the default random init is
    identical to what the sweep produced.
    """
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = _build_model(cfg, device)

    if cfg.get("init_style", "random") == "structured":
        init_dpa_internal_readout_prepost(
            model, mem=0, out=1,
            memory_lambda   = float(cfg.get("memory_lambda",    0.8)),
            decision_lambda = float(cfg.get("decision_lambda",  0.5)),
            target_mn_corr  = float(cfg.get("target_mn_corr",  0.8)),
            target_out_mn_corr = cfg.get("target_out_mn_corr", 0.8),
            sample_scale    = float(cfg.get("sample_scale",    1.0)),
            test_scale      = float(cfg.get("test_scale",      1.0)),
            mix_strength    = float(cfg.get("mix_strength",    0.0)),
            noise_scale_mn  = 1.0,
            noise_scale_in  = 1.0,
            seed            = seed,
            verbose         = False,
        )

    model.eval()
    return model


def _load_ckpt(model: LowRankModel, sweep_dir: str, stage: str,
               run_id: str, device: str) -> bool:
    path = os.path.join(sweep_dir, run_id, f"{stage}_{run_id}.pth")
    if not os.path.exists(path):
        path = os.path.join(sweep_dir, "models", f"{stage}_{run_id}.pth")
    if not os.path.exists(path):
        return False
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return True


def _input_labels(input_size: int, cue_on_go_input: bool, rwd: bool) -> list[str]:
    if cue_on_go_input:
        base = ["sample A", "sample B", "test C", "test D", "go+cue", "nogo"]
    else:
        base = ["sample A", "sample B", "test C", "test D", "go", "nogo", "cue"]
    if rwd:
        base.append("reward")
    return base[:input_size]


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _scatter_panel(ax, xvec, yvec, xlabel, ylabel):
    r = pearson_r(xvec, yvec)
    ax.scatter(xvec, yvec, s=3, alpha=0.25, linewidths=0, rasterized=True)
    ax.axhline(0, color="k", lw=0.5, ls="--", zorder=0)
    ax.axvline(0, color="k", lw=0.5, ls="--", zorder=0)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"r = {r:.3f}", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Figure 1: m vs n  (2 × 2 grid)
# ---------------------------------------------------------------------------

def plot_m_n(m: np.ndarray, n: np.ndarray, out_path: str) -> None:
    rank = m.shape[1]
    fig, axes = plt.subplots(rank, rank, figsize=(rank * 2.8, rank * 2.8))
    if rank == 1:
        axes = np.array([[axes]])
    for i in range(rank):       # n index (rows)
        for j in range(rank):   # m index (cols)
            _scatter_panel(axes[i, j], m[:, j], n[:, i],
                           f"$m_{j}$", f"$n_{i}$")
    fig.suptitle("m vs n", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Figure 2/3: each input vs n or m  (n_inputs × rank grid)
# ---------------------------------------------------------------------------

def plot_inputs_vs_vec(wi: np.ndarray, vec: np.ndarray,
                       vec_name: str, input_labels: list[str],
                       out_path: str) -> None:
    """wi: (hidden, input_size). vec: (hidden, rank)."""
    n_in = len(input_labels)
    rank = vec.shape[1]
    fig, axes = plt.subplots(n_in, rank,
                             figsize=(rank * 2.8, n_in * 2.5),
                             squeeze=False)
    for k, lab in enumerate(input_labels):
        for j in range(rank):
            _scatter_panel(axes[k, j], wi[:, k], vec[:, j],
                           lab, f"${vec_name}_{j}$")
    fig.suptitle(f"inputs vs {vec_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: correlation matrix (single run/stage)
# ---------------------------------------------------------------------------

def _corr_matrix(m: np.ndarray, n: np.ndarray,
                 wi: np.ndarray, input_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (C, names) where C is the Pearson-r matrix."""
    rank  = m.shape[1]
    names = [f"$m_{j}$" for j in range(rank)] + \
            [f"$n_{j}$" for j in range(rank)] + \
            input_labels
    vecs  = [m[:, j] for j in range(rank)] + \
            [n[:, j] for j in range(rank)] + \
            [wi[:, k] for k in range(len(input_labels))]
    N = len(names)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = pearson_r(vecs[i], vecs[j])
    return C, names


def _draw_corr_ax(ax, C: np.ndarray, names: list[str], rank: int,
                  title: str = "", fontsize: int = 7,
                  show_yticks: bool = True) -> object:
    """Draw one correlation matrix panel into ax. Returns the imshow object."""
    N  = len(names)
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(N))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=fontsize)
    if show_yticks:
        ax.set_yticks(range(N))
        ax.set_yticklabels(names, fontsize=fontsize)
    else:
        ax.set_yticks([])
    for s in [rank, 2 * rank]:
        ax.axhline(s - 0.5, color="k", lw=1.0)
        ax.axvline(s - 0.5, color="k", lw=1.0)
    for i in range(N):
        for j in range(N):
            color = "w" if abs(C[i, j]) > 0.65 else "k"
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    fontsize=fontsize - 1, color=color)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    return im


def plot_corr_matrix(m: np.ndarray, n: np.ndarray,
                     wi: np.ndarray, input_labels: list[str],
                     out_path: str) -> None:
    C, names = _corr_matrix(m, n, wi, input_labels)
    rank = m.shape[1]
    N    = len(names)
    cell = 0.72
    fig, ax = plt.subplots(figsize=(N * cell + 2.0, N * cell + 1.5))
    im = _draw_corr_ax(ax, C, names, rank,
                       title="Correlation matrix: m, n, input weights")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Summary figure: mean corr matrix across runs, one panel per stage
# ---------------------------------------------------------------------------

def _get_weights(r: dict, stage: str, sweep_dir: str, device: str):
    """Return (m, n, wi) arrays for a run/stage, or None if unavailable."""
    cfg = r["config"]
    if stage == "init":
        model = _build_init_model(cfg, device)
    else:
        model = _build_model(cfg, device)
        if not _load_ckpt(model, sweep_dir, stage, r["run_id"], device):
            return None
    m  = model.m.detach().cpu().numpy()
    n  = model.n.detach().cpu().numpy()
    wi = model.wi.weight.detach().cpu().numpy()
    return m, n, wi


def plot_summary_corr_mean_std(all_meta: list[dict], stages: list[str],
                               sweep_dir: str, out_path: str,
                               device: str = "cpu") -> None:
    """
    One panel per stage.  Color = mean Pearson r across seeds.
    A white circle is overlaid in each cell whose radius scales with std across seeds:
      radius = 0 → seeds fully agree; radius = 0.45 → maximum observed std.
    A reference legend in the last panel shows what three circle sizes mean.
    """
    if not all_meta:
        return

    cfg0   = all_meta[0]["config"]
    cue    = bool(cfg0.get("cue_on_go_input", False))
    rwd    = bool(cfg0.get("rwd", True))
    labels = _input_labels(int(cfg0["input_size"]), cue, rwd)
    rank   = 2
    stage_labels = {"init": "Init", "dpa": "After DPA",
                    "naive": "After GNG", "expert": "After Dual"}

    # Collect per-run matrices
    run_mats: dict[str, list[np.ndarray]] = {s: [] for s in stages}
    for r in all_meta:
        for stage in stages:
            result = _get_weights(r, stage, sweep_dir, device)
            if result is not None:
                m, n, wi = result
                C, names = _corr_matrix(m, n, wi, labels)
                run_mats[stage].append(C)

    valid_stages = [s for s in stages if run_mats[s]]
    if not valid_stages:
        return

    # Compute mean and std per stage
    C_means = {s: np.mean(run_mats[s], axis=0) for s in valid_stages}
    C_stds  = {s: np.std(run_mats[s],  axis=0) for s in valid_stages}

    # Global max std (for consistent circle scaling across stages)
    global_max_std = max(C_stds[s].max() for s in valid_stages)
    MAX_RADIUS     = 0.45   # data units; fills ~90% of a cell

    N     = len(names)
    n_stg = len(valid_stages)
    cell  = 0.72
    panel = N * cell          # physical width/height of one data panel (inches)

    # n_stg data columns + 1 legend column (same width), then colorbar space
    fig, axes = plt.subplots(
        1, n_stg + 1,
        figsize=((n_stg + 1) * panel + 1.8, panel + 1.8),
        gridspec_kw={"width_ratios": [1] * n_stg + [1]},
        squeeze=False,
    )

    im_last = None
    for col, stage in enumerate(valid_stages):
        ax     = axes[0, col]
        C_mean = C_means[stage]
        C_std  = C_stds[stage]
        n_runs = len(run_mats[stage])

        im = ax.imshow(C_mean, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        im_last = im

        # Separator lines between m / n / input blocks
        for s in [rank, 2 * rank]:
            ax.axhline(s - 0.5, color="k", lw=1.0, zorder=5)
            ax.axvline(s - 0.5, color="k", lw=1.0, zorder=5)

        # White circle overlay: radius ∝ std
        if global_max_std > 0:
            for i in range(N):
                for j in range(N):
                    r = MAX_RADIUS * C_std[i, j] / global_max_std
                    if r > 0.01:
                        ax.add_patch(plt.Circle(
                            (j, i), r,
                            color="w", alpha=0.75, zorder=4, linewidth=0,
                        ))

        ax.set_xticks(range(N))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        if col == 0:
            ax.set_yticks(range(N))
            ax.set_yticklabels(names, fontsize=7)
        else:
            ax.set_yticks([])
        ax.set_title(f"{stage_labels.get(stage, stage)}\n(n={n_runs})", fontsize=8)

    # Legend panel: same physical size as data panels, same data coordinate scale.
    # Three rows of circles (std = 0, 0.5×max, max) centred in the panel.
    ax_leg = axes[0, n_stg]
    ax_leg.set_xlim(-0.5, N - 0.5)
    ax_leg.set_ylim(-0.5, N - 0.5)
    ax_leg.set_aspect("equal")
    ax_leg.axis("off")
    ax_leg.set_title("std (circle size)", fontsize=7)

    if global_max_std > 0:
        legend_fracs  = [0.0, 0.5, 1.0]
        legend_stds   = [f * global_max_std for f in legend_fracs]
        legend_labels = ["0", f"{global_max_std*0.5:.2f}", f"{global_max_std:.2f}"]
        # Space three circles evenly along the vertical centre of the panel
        cx = (N - 1) / 2.0
        ys = np.linspace(N * 0.2, N * 0.8, len(legend_fracs))
        for y, frac, std_val, lab in zip(ys, legend_fracs, legend_stds, legend_labels):
            r = MAX_RADIUS * frac
            # Blue background tile so white circle is visible
            ax_leg.add_patch(plt.Rectangle(
                (cx - 0.5, y - 0.5), 1, 1,
                color="tab:blue", alpha=0.25, zorder=1, linewidth=0,
            ))
            if r > 0.01:
                ax_leg.add_patch(plt.Circle(
                    (cx, y), r,
                    color="w", alpha=0.75, linewidth=0.6,
                    edgecolor="gray", zorder=3,
                ))
            ax_leg.text(cx + 0.7, y, lab,
                        ha="left", va="center", fontsize=7)

    # Colorbar to the right of legend panel
    fig.colorbar(im_last, ax=axes[0, n_stg], shrink=0.6, label="mean Pearson r",
                 fraction=0.08, pad=0.15)

    fig.suptitle("Mean correlation ± seed variability (circle = std)",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_summary_corr_stages(all_meta: list[dict], stages: list[str],
                             sweep_dir: str, out_path: str,
                             device: str = "cpu") -> None:
    """Mean Pearson-r matrix across all runs, one panel per stage."""
    if not all_meta:
        return

    # Use first run's config for labels (all runs share input layout)
    cfg0   = all_meta[0]["config"]
    cue    = bool(cfg0.get("cue_on_go_input", False))
    rwd    = bool(cfg0.get("rwd", True))
    labels = _input_labels(int(cfg0["input_size"]), cue, rwd)
    rank   = 2

    stage_labels = {"init": "Init", "dpa": "After DPA",
                    "naive": "After GNG", "expert": "After Dual"}

    # Accumulate correlation matrices per stage
    sums   = {s: None for s in stages}
    counts = {s: 0    for s in stages}

    for r in all_meta:
        for stage in stages:
            result = _get_weights(r, stage, sweep_dir, device)
            if result is None:
                continue
            m, n, wi = result
            C, names = _corr_matrix(m, n, wi, labels)
            if sums[stage] is None:
                sums[stage] = C.copy()
            else:
                sums[stage] += C
            counts[stage] += 1

    valid_stages = [s for s in stages if counts[s] > 0]
    if not valid_stages:
        print("  [skip] no data for any stage")
        return

    N      = len(names)
    n_stg  = len(valid_stages)
    cell   = 0.68
    pad    = 1.8   # extra width for colorbar + left yticks
    fig, axes = plt.subplots(
        1, n_stg,
        figsize=(n_stg * (N * cell + 0.3) + pad, N * cell + 1.5),
        squeeze=False,
    )

    im_last = None
    for col, stage in enumerate(valid_stages):
        C_mean = sums[stage] / counts[stage]
        title  = stage_labels.get(stage, stage)
        if counts[stage] > 1:
            title += f"\n(n={counts[stage]})"
        im_last = _draw_corr_ax(
            axes[0, col], C_mean, names, rank,
            title=title, fontsize=7,
            show_yticks=(col == 0),
        )

    fig.colorbar(im_last, ax=axes[0, -1], shrink=0.85, label="Pearson r")
    fig.suptitle("Mean correlation structure across stages", fontsize=11, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")




def run_one(r: dict, stage: str, sweep_dir: str,
            out_root: str, device: str) -> None:
    run_id = r["run_id"]
    cfg    = r["config"]
    cue    = bool(cfg.get("cue_on_go_input", False))
    rwd    = bool(cfg.get("rwd", True))
    labels = _input_labels(int(cfg["input_size"]), cue, rwd)

    if stage == "init":
        model = _build_init_model(cfg, device)
    else:
        model = _build_model(cfg, device)
        if not _load_ckpt(model, sweep_dir, stage, run_id, device):
            print(f"  [skip] {run_id}/{stage}: checkpoint not found")
            return

    m  = model.m.detach().cpu().numpy()   # (hidden, rank)
    n  = model.n.detach().cpu().numpy()   # (hidden, rank)
    wi = model.wi.weight.detach().cpu().numpy()   # (hidden, input_size)

    out_dir = os.path.join(out_root, run_id, stage)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{run_id} / {stage}  →  {out_dir}")
    plot_m_n(m, n, os.path.join(out_dir, "mn_scatter.pdf"))
    plot_inputs_vs_vec(wi, n, "n", labels,
                       os.path.join(out_dir, "inputs_vs_n.pdf"))
    plot_inputs_vs_vec(wi, m, "m", labels,
                       os.path.join(out_dir, "inputs_vs_m.pdf"))
    plot_corr_matrix(m, n, wi, labels,
                     os.path.join(out_dir, "corr_matrix.pdf"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", required=True,
                   help="Sweep directory containing results.jsonl")
    p.add_argument("--run_id", default=None,
                   help="Specific run ID (default: all runs)")
    p.add_argument("--stages", nargs="+", default=["init", "dpa", "naive", "expert"],
                   help="Stages to plot (default: init dpa naive expert)")
    p.add_argument("--out_dir", default=None,
                   help="Output root (default: ./results/figures/<sweep>/mn_inputs)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    sweep_name = os.path.basename(args.sweep_dir.rstrip("/"))
    out_root = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results", "figures", sweep_name, "mn_inputs"
    )

    all_meta = _load_all_meta(args.sweep_dir)
    if args.run_id:
        all_meta = [r for r in all_meta if r["run_id"] == args.run_id]
        if not all_meta:
            raise SystemExit(f"run_id {args.run_id!r} not found in results.jsonl")

    for r in all_meta:
        for stage in args.stages:
            run_one(r, stage, args.sweep_dir, out_root, args.device)

    # Summary figures (only when >1 run)
    if len(all_meta) > 1:
        print("\nSummary: mean corr matrix across stages...")
        plot_summary_corr_stages(
            all_meta, args.stages, args.sweep_dir,
            os.path.join(out_root, "summary_corr_stages.pdf"),
            device=args.device,
        )
        print("Summary: mean + std overlay...")
        plot_summary_corr_mean_std(
            all_meta, args.stages, args.sweep_dir,
            os.path.join(out_root, "summary_corr_mean_std.pdf"),
            device=args.device,
        )

    print(f"\nDone. Output in {out_root}")


if __name__ == "__main__":
    main()
