"""Measure the sample-memory wells as the AUTONOMOUS fixed points of the trained net — NO assumption
about how MANY wells there are. For each expert checkpoint we clamp the input to the 'Autonomous'
condition (attention at the trained amplitude), find ALL fixed points (analytic finder for saturating
φ; grid-sim ground truth for non-saturating φ), classify stability, and list EVERY memory-well
attractor (|κ₀| large) with its (κ₀, κ₁), tallying up (κ₁>0) vs down (κ₁<0). Read-only.

Why the rewrite (2026-08-04, §20): the previous version hard-coded a TWO-well model — it picked one
attractor per κ₀ sign (`_side_well`, outermost |κ₀|) and averaged them. When a memory bifurcates into
FOUR wells (each sample A/B splitting into an UP κ₁>0 and a DOWN κ₁<0 attractor — exactly what the
lif + decision-readout-DC runs do), that collapse threw away half the fixed points and averaged an
up-well with a down-well into a meaningless number, repeatedly mis-reporting the well locations. Now
we report all of them. The GOAL metric is "all memory wells DOWN" (memory held in the no-lick plane).
"""
import os, sys, collections
import numpy as np
import torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
from plot_sweep import (_load_sweep_meta, _build_model, _load_ckpt, XLIM, YLIM, TIMINGS,
                        find_all_fixed_points, classify_fixed_points, make_input,
                        _model_has_backbone, _sim_fps_for_conditions)
from src.tasks import generate_dual_trials

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
K0_WELL_FRAC = 0.4   # a memory well has |κ₀| ≥ this fraction of the run's κ₀ scale (vs a κ₁ pole)


@torch.no_grad()
def _kappa_box(model, meta):
    """Adaptive FP-search box: ± 1.3× the κ range the trained net actually visits on dual trials
    (floored at the default XLIM). Essential for non-saturating φ (softplus/relu/elu) whose κ runs
    to ±5..±10 — the default ±1.5 grid misses the wells entirely. Also returns the κ₀ scale used to
    tell memory wells (large |κ₀|) from κ₁ decision poles."""
    dt = TIMINGS["dual"]
    X, y, _, _ = generate_dual_trials(
        256, timing=dt, input_size=meta.input_size, noise=meta.noise, target_rank=meta.rank,
        cue_on_go_input=meta.cue_on_go_input, cue_scale=meta.cue_scale, nogo_target=meta.nogo_target,
        input_scale=1.0, attention_input=meta.attention_input, attention_gated=meta.attention_gated)
    out = model(X.to(DEV), y.to(DEV)).cpu().numpy()
    k0max = max(abs(out[:, :, 0]).max(), abs(XLIM[1]))
    k1max = max(abs(out[:, :, -1]).max(), abs(YLIM[1]))
    return (-1.3 * k0max, 1.3 * k0max), (-1.3 * k1max, 1.3 * k1max), float(k0max)


# Saturating, odd φ where the analytic κ-reduction + eigenvalue classification is reliable. Every
# other φ (relu/softplus/elu/gelu) is non-saturating: its wells sit at wide, awkward κ and are SLOW
# attractors (map |λ|≈0.99) that both the root-finder misses and the classifier mislabels "marginal".
# Those go to the grid-sim ground truth instead (forward-integrate a κ-grid, cluster settled points).
_SATURATING = {"tanh", "erf", "lif", "lif_sc", "tanh_asym"}


@torch.no_grad()
def _grid_sim_wells(model, meta, k0_scale):
    """Ground-truth wells via forward simulation: integrate a κ-grid autonomously (attention at the
    trained amplitude), take settled endpoints, and cluster them into ALL distinct attractors (greedy
    merge within tol) — NO per-side / 2-well assumption. Stability from per-cluster spread."""
    import ei_flow
    R  = 1.25 * max(k0_scale, abs(YLIM[1]))
    ff = make_input(meta.input_size, active_dims=None, value=1.0, device=DEV,
                    dtype=next(model.parameters()).dtype)
    if meta.attention_input:
        ff[-1] = getattr(meta, "attention_scale", 1.0)   # clamp attention at the TRAINED amplitude
    traj, _ = ei_flow.run_grid(model, ff, R=R, gsize=30, set_w=40, T=2000, I0=1.0, device=DEV)
    end = traj[:, -150:, :].mean(1)          # settled endpoint per grid start [B, 2]
    tol = 0.12 * max(1.0, k0_scale)
    centers, counts = [], []                 # greedy 1-pass clustering of settled endpoints
    for p in end:
        for i, c in enumerate(centers):
            if np.linalg.norm(p - c) < tol:
                counts[i] += 1; centers[i] = centers[i] + (p - centers[i]) / counts[i]; break
        else:
            centers.append(p.copy()); counts.append(1)
    fps, stabs = [], []
    for c, n in zip(centers, counts):
        if n < 5:                            # negligible basin
            continue
        members = end[np.linalg.norm(end - c, axis=1) < tol]
        spread  = float(np.linalg.norm(members - c, axis=1).std())
        fps.append(c)
        stabs.append("attractor" if spread < 0.15 * max(1.0, k0_scale) else "marginal")
    return (np.asarray(fps) if fps else np.empty((0, 2))), stabs, k0_scale


@torch.no_grad()
def autonomous_fps(meta, sweep_dir):
    """(fps, stabs, k0_scale) for the Autonomous input condition, or None if no expert ckpt."""
    model = _build_model(meta, DEV)
    if not _load_ckpt(model, sweep_dir, "expert", meta.run_id, DEV):
        return None
    dtype = next(model.parameters()).dtype
    if _model_has_backbone(model):   # EISTP / fixed-backbone: analytic κ-reduction invalid → sim finder
        cond = [("Autonomous", None, "black", "o")]
        _, _, _, fps, stabs = _sim_fps_for_conditions(model, meta.input_size, cond, DEV)[0]
        return np.asarray(fps), list(stabs), 1.0
    xlim, ylim, k0_scale = _kappa_box(model, meta)   # adaptive to the run's κ scale
    if getattr(model, "nonlinearity_str", "tanh") not in _SATURATING:
        return _grid_sim_wells(model, meta, k0_scale)   # non-saturating → ground-truth grid-sim
    ff = make_input(meta.input_size, active_dims=None, value=1.0, device=DEV, dtype=dtype)
    if meta.attention_input:         # autonomous = only tonic attention on (matches the flow / fp figures)
        ff[-1] = getattr(meta, "attention_scale", 1.0)   # clamp attention at the TRAINED amplitude
    fps, _   = find_all_fixed_points(model, xlim=xlim, ylim=ylim, ff_input=ff,
                                     n_seeds=25, residual_tol=1e-8, merge_tol=5e-2)
    # marginal_tol=2e-3 (not the 1e-2 default): shallow subcritical wells are genuine but SLOW
    # attractors (map |λ|≈0.99); the loose 1e-2 band mislabels them "marginal" and drops them. 2e-3
    # still catches true line/ring manifolds (|λ|=1.000) but keeps slow point attractors as clean.
    stabs, _ = classify_fixed_points(model, fps, ff_input=ff, marginal_tol=2e-3, slow_tol=5e-2)
    return np.asarray(fps), list(stabs), k0_scale


K1_DEADBAND = 0.05   # |κ₁| < this ⇒ "on the line" (neither clearly up nor down)


def all_wells(meta, sweep_dir):
    """EVERY autonomous memory-well fixed point — NO two-well assumption. Returns a list of
    (κ₀, κ₁, stab, is_attractor) for all FPs with |κ₀| ≥ K0_WELL_FRAC·k0_scale (a memory attractor,
    vs a decision pole at small |κ₀|), sorted by κ₀. `None` if no expert ckpt."""
    r = autonomous_fps(meta, sweep_dir)
    if r is None:
        return None
    fps, stabs, k0_scale = r
    thr = K0_WELL_FRAC * k0_scale
    ws = [(float(f[0]), float(f[1]), s, ("attractor" in s))
          for f, s in zip(fps, stabs)
          if abs(f[0]) > thr and ("attractor" in s or "marginal" in s)]
    return sorted(ws, key=lambda w: w[0])


def _lvl(k1):
    return "dn" if k1 < -K1_DEADBAND else ("up" if k1 > K1_DEADBAND else "0")


for sweep in sys.argv[1:]:
    sd = f"results/dual/{sweep}"
    metas = _load_sweep_meta(sd)
    print(f"\n== {sweep} :: ALL autonomous memory wells (no 2-well assumption; "
          f"dn = κ₁<{-K1_DEADBAND:+.2f} no-lick, up = κ₁>{K1_DEADBAND:+.2f}; * = marginal) ==")
    up_k1 = collections.defaultdict(list); dn_k1 = collections.defaultdict(list)
    n_up  = collections.defaultdict(list); n_dn  = collections.defaultdict(list)
    anydn = collections.defaultdict(int);  alldn = collections.defaultdict(int)
    ntot  = collections.defaultdict(int)
    for m in sorted(metas, key=lambda x: x.run_id):
        try:
            ws = all_wells(m, sd)
        except Exception as e:                       # incompatible/old ckpt in the dir → skip, don't crash
            print(f"  {m.run_id:16s} (load failed: {type(e).__name__})"); continue
        if ws is None:
            print(f"  {m.run_id:16s} (no ckpt)"); continue
        tag = m.run_id.split("_", 1)[1]
        up  = [w for w in ws if w[1] >  K1_DEADBAND]
        dn  = [w for w in ws if w[1] < -K1_DEADBAND]
        ntot[tag] += 1
        if dn: anydn[tag] += 1
        if ws and not up and dn: alldn[tag] += 1     # GOAL: every memory well in the no-lick plane
        n_up[tag].append(len(up)); n_dn[tag].append(len(dn))
        up_k1[tag] += [w[1] for w in up]; dn_k1[tag] += [w[1] for w in dn]
        pts = " ".join(f"({w[0]:+.2f},{w[1]:+.2f}){_lvl(w[1])}{'' if w[3] else '*'}" for w in ws)
        print(f"  {m.run_id:16s} [{len(ws)}w {len(up)}up {len(dn)}dn]  {pts}")
    print("  " + "-" * 72)
    for tag in sorted(ntot):
        nt = ntot[tag]
        du = np.array(dn_k1[tag]) if dn_k1[tag] else np.array([np.nan])
        uu = np.array(up_k1[tag]) if up_k1[tag] else np.array([np.nan])
        print(f"  arm {tag:12s} per-seed avg {np.mean(n_up[tag]):.1f}up + {np.mean(n_dn[tag]):.1f}dn wells | "
              f"≥1 down: {anydn[tag]}/{nt}   ALL-down (GOAL): {alldn[tag]}/{nt}")
        print(f"      down-well κ₁ mean={np.nanmean(du):+.3f} (n={len(dn_k1[tag])})   "
              f"up-well κ₁ mean={np.nanmean(uu):+.3f} (n={len(up_k1[tag])})")
