"""Flow VERDICT — the canonical scoring of a rank-2 sweep against the project goal.

The ONLY success criterion: the SAMPLE-MEMORY wells (the ± κ₀ attractor pair that carries
the A/B bit) sit at κ₁ < 0. Everything else — response-window behaviour, deep memory-less
attractors, per-arm eval boundaries — is NOT pushdown and is reported separately so it can
never be conflated again.

Per run (expert stage by default):
  - scipy FP finder on a WIDE box (±4.5 — softplus/inflated runs park structure out to ±3.8),
    every root re-verified directly via |F(κ*)| (the brainpy path can return spurious FPs);
  - attractors split into MEMORY WELLS (|κ₀| ≥ --mem_k0, the ± pair) vs EXTRAS
    (κ₀≈0 basement/lick states carry NO sample bit; up/down copies = parking);
  - ALL-DOWN = every memory well κ₁ < 0 (the goal), plus spiral count (complex Jacobian eigs);
  - behaviour at the FIXED boundary 0 (never each arm's own lenient midpoint) + dual_dpa
    from results.jsonl.

Which field (2026-09-25): these nets are trained with INPUT noise, which lowers each unit's effective gain by
1/sqrt(1 + g²σ²‖w_i‖²) (median 0.55–0.72 in the trained nets). The trials follow the INPUT-NOISE-AVERAGED field
(σ = σ_eff, analytic Gaussian average): end-of-delay states sit on its wells, not on the deterministic ones
(s1_log: trials (+0.94,−0.56), averaged well (+0.93,−0.58), deterministic (+1.23,−0.34)). The verdict therefore
scores the averaged field (--field noise, default) and prints the deterministic one as a second line
(--field both, default) — they disagree on "all down" in 3/8 recipe7_bfix and 8/8 logsub nets.
Each memory well is tagged with the Dual trial types that occupy it at the end of the delay
(N = no Go/NoGo, G = Go, X = NoGo; ≥ 50 % of that type nearest to the well).

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python flow_verdict.py \
      --sweep_dir results/dual/sweep_r2sign2 [--stage expert] [--run_ids ...] [--xlim 4.5]
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation_probe import load_run, discover_run_ids, run_dt_alpha
from src.flow_field import (low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np,
                            kappa_from_rates)
from src.flow_fixedpoints import find_all_fixed_points, classify_fixed_points
from src.tasks import make_timings, generate_dual_trials


def analyse_run(sweep_dir, rid, stage, xlim, mem_k0, resid_tol=1e-4, noise_sigma=0.0, loaded=None):
    model, cfg = loaded if loaded is not None else load_run(sweep_dir, rid, stage=stage)
    params = low_rank_numpy_params(model)
    ff = torch.zeros(cfg["input_size"])
    if cfg.get("attention_input", False):   # the last channel is attention ONLY when the run has one;
        ff[-1] = cfg.get("attention_scale", 1.0) * cfg.get("input_scale", 1.0)   # otherwise it is NoGo

    fps, _ = find_all_fixed_points(model, (-xlim, xlim), (-xlim, xlim), ff, noise_sigma=noise_sigma)
    fps = np.atleast_2d(np.asarray(fps, dtype=float))
    labels, eigvals = classify_fixed_points(model, fps, ff, noise_sigma=noise_sigma)
    labels = [str(l) for l in labels]

    wells, extras, spirals = [], [], 0
    for f, lab, ev in zip(fps, labels, eigvals):
        if lab != "attractor":
            continue
        k = np.array([float(f[0]), float(f[1])])
        # direct verification — never trust a finder unchecked
        F = np.asarray(low_rank_field_np(params, k[None, :], ff.numpy(), noise_sigma=noise_sigma)).ravel()
        if np.linalg.norm(F) > resid_tol:
            continue  # spurious
        ev = np.asarray(ev).ravel()
        sp = bool(np.abs(ev.imag).max() > 1e-3)
        spirals += sp
        (wells if abs(k[0]) >= mem_k0 else extras).append((round(k[0], 2), round(k[1], 2), sp))

    # a memory PAIR needs both κ₀ signs represented; otherwise flag it
    signs = {np.sign(w[0]) for w in wells}
    pair_ok = len(wells) >= 2 and len(signs) == 2
    all_down = pair_ok and all(w[1] < 0 for w in wells)
    return model, cfg, wells, extras, spirals, pair_ok, all_down


def _dual_trials(cfg, n=1024, seed=0):
    """The run's own Dual trials + timing. Shared by behaviour_at_zero and reach_at_delay_end."""
    DT, alpha, _ = run_dt_alpha(cfg)
    T = make_timings(DT)["dual"]
    torch.manual_seed(seed)
    X, y, _, names = generate_dual_trials(
        n, T, input_size=cfg["input_size"], target_rank=cfg["target_rank"],
        noise=cfg["noise"] * float(np.sqrt(1 - np.exp(-2 * alpha))),
        cue_on_go_input=True, cue_scale=cfg.get("cue_scale", 2.0),
        input_scale=cfg.get("input_scale", 1.0), attention_input=cfg.get("attention_input", False),
        attention_gated=cfg.get("attention_gated", True),
        windowed_targets=cfg.get("windowed_targets", True),
        decay_to_zero=cfg.get("decay_to_zero", False),
        gng_response=cfg.get("gng_response", True),
        gng_memory=cfg.get("dual_gng_memory", False),
        response_in_cue=cfg.get("response_in_cue", False),
        gng_rwd_after_cue=cfg.get("gng_rwd_after_cue", False))
    return X, y, np.asarray(names).astype(str), T, DT


def behaviour_at_zero(model, cfg):
    """Dual go/nogo at the FIXED boundary 0, in the run's own trained response window."""
    X, y, names, T, DT = _dual_trials(cfg)
    half = int(round(0.5 / DT)); co = int(T.n_stim_off[2])
    in_cue = bool(cfg.get("response_in_cue", False)) and not bool(cfg.get("gng_rwd_after_cue", False))
    model.noise = 0.0
    dev = next(model.parameters()).device
    pred = model(X.to(dev), y.to(dev))[..., -1].detach().cpu()
    is_ng = torch.as_tensor(["_nogo_" in n for n in names])
    is_go = torch.as_tensor(["_go_" in n for n in names])
    m = (pred[:, co - half:co] if in_cue else pred[:, co:co + half]).mean(1)
    return ((m[is_ng] <= 0).float().mean().item(), m[is_ng].mean().item(),
            (m[is_go] > 0).float().mean().item())


def reach_at_delay_end(model, cfg, wells):
    """Do the A/B trials actually REACH the memory wells the geometry advertises?

    Geometry alone cannot answer this: a κ₀ well pair can exist and be unreachable, or the sample
    bit can live on a rotated axis. Measured at the last step before the TEST stimulus (end of the
    memory delay), reports
      sep     — sign(κ₀) separation of the two sample classes (0.5 = chance, 1.0 = perfect),
      in_well — fraction of trials sitting within 0.5 of one of the reported memory wells,
      |κ₀|    — mean memory amplitude at delay end (compare against --mem_k0).
    Diagnostic only: NOT part of ALL-DOWN, so verdicts stay comparable with the existing log."""
    X, _, names, T, _ = _dual_trials(cfg)
    model.noise = 0.0
    dev = next(model.parameters()).device
    with torch.no_grad():
        _, rates, _ = model(X.to(dev), ret_rates=True)
    kap = kappa_from_rates(model, rates).cpu().numpy()          # (B, T, rank)
    t_end = int(T.n_stim_on[3]) - 1                             # end of the memory delay
    k = kap[:, t_end, :2]
    sample = np.array([n.split("_")[0] for n in names])
    cls = np.unique(sample)
    if len(cls) < 2:
        return float("nan"), float("nan"), float(np.abs(k[:, 0]).mean()), [""] * len(wells)
    frac_pos = (k[sample == cls[0], 0] > 0).mean()
    sep = max(frac_pos, 1.0 - frac_pos)
    tags = [""] * len(wells)
    if len(wells):
        W = np.asarray([[w[0], w[1]] for w in wells], dtype=float)
        dist = np.linalg.norm(k[:, None, :] - W[None], axis=2)
        in_well = float((dist.min(1) < 0.5).mean())
        near = dist.argmin(1)
        typ = np.array(["G" if "_go_" in n else ("X" if "_nogo_" in n else "N") for n in names])
        for j in range(len(wells)):   # per (type, sample): a type occupies well j if ≥ 50 % of that sample's trials of that type sit nearest to it
            tags[j] = "".join(t for t in "NGX" if any(((typ == t) & (sample == c)).any() and
                                                      (near[(typ == t) & (sample == c)] == j).mean() >= 0.5 for c in cls))
    else:
        in_well = 0.0
    return float(sep), in_well, float(np.abs(k[:, 0]).mean()), tags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--stage", default="expert", choices=["dpa", "naive", "expert"])
    ap.add_argument("--xlim", type=float, default=4.5)
    ap.add_argument("--mem_k0", type=float, default=0.5,
                    help="|κ₀| above this = sample-coding attractor (memory well)")
    ap.add_argument("--no_behaviour", action="store_true")
    ap.add_argument("--field", default="both", choices=["noise", "clean", "both"],
                    help="noise = input-noise-averaged field at the run's σ_eff (what the trials follow; the verdict); "
                         "clean = deterministic field; both = verdict on noise + a second line for clean (default)")
    args = ap.parse_args()

    rows = {}
    jl = os.path.join(args.sweep_dir, "results.jsonl")
    if os.path.exists(jl):
        for line in open(jl):
            r = json.loads(line)
            rows[r["run_id"]] = r.get("accuracy", {}).get("after_dual", {})

    rids = args.run_ids or discover_run_ids(args.sweep_dir)
    n_down, n_down_clean = 0, 0
    fld = "input-noise-averaged field" if args.field != "clean" else "deterministic field"
    print(f"VERDICT — {os.path.basename(os.path.normpath(args.sweep_dir))}  stage={args.stage}  {fld}  "
          f"(memory well = attractor with |κ₀| ≥ {args.mem_k0}; goal = ALL memory wells κ₁ < 0; "
          f"tags = trial types occupying the well at delay end: N none, G Go, X NoGo)")
    fmt = lambda ws, tg: " ".join(f"({w[0]:+.2f},{w[1]:+.2f}){'S' if w[2] else ''}{'[' + t + ']' if t else ''}"
                                  for w, t in zip(ws, tg)) or "NONE"
    for rid in rids:
        loaded = load_run(args.sweep_dir, rid, stage=args.stage)
        _, alpha, _ = run_dt_alpha(loaded[1])
        sig = float(loaded[1]["noise"]) * float(np.sqrt(1 - np.exp(-2 * alpha))) if args.field != "clean" else 0.0
        model, cfg, wells, extras, spirals, pair_ok, all_down = analyse_run(
            args.sweep_dir, rid, args.stage, args.xlim, args.mem_k0, noise_sigma=sig, loaded=loaded)
        n_down += all_down
        acc = rows.get(rid, {})
        beh, tags = "", [""] * len(wells)
        if not args.no_behaviour:
            ng0, ngm, go0 = behaviour_at_zero(model, cfg)
            sep, inw, k0m, tags = reach_at_delay_end(model, cfg, wells)
            beh = (f"  nogo@0={ng0:.2f}(μ{ngm:+.2f}) go@0={go0:.2f}"
                   f"  reach: sep={sep:.2f} in_well={inw:.2f} |κ₀|={k0m:.2f}")
        estr = f"  extras={len(extras)}:" + " ".join(f"({e[0]:+.1f},{e[1]:+.1f})" for e in extras) if extras else ""
        flag = "ALL-DOWN ✓" if all_down else ("NO PAIR ✗" if not pair_ok else "not down")
        print(f"{rid}: {flag}  mem wells: {fmt(wells, tags)}{estr}  spirals={spirals}"
              f"  dual_dpa={acc.get('dual_dpa', float('nan')):.3f}{beh}", flush=True)
        if args.field == "both":
            _, _, wc, xc, spc, okc, downc = analyse_run(args.sweep_dir, rid, args.stage, args.xlim, args.mem_k0,
                                                        noise_sigma=0.0, loaded=loaded)
            n_down_clean += downc
            flagc = "ALL-DOWN" if downc else ("NO PAIR" if not okc else "not down")
            print(f"    deterministic field: {flagc}  mem wells: {fmt(wc, [''] * len(wc))}", flush=True)
    print(f"SUMMARY: {n_down}/{len(rids)} seeds with all memory wells below the line ({fld})"
          + (f"; deterministic field: {n_down_clean}/{len(rids)}" if args.field == "both" else ""))


if __name__ == "__main__":
    main()
