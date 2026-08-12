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

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python flow_verdict.py \
      --sweep_dir results/dual/sweep_r2sign2 [--stage expert] [--run_ids ...] [--xlim 4.5]
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation_probe import load_run, discover_run_ids
from src.flow_field import low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np
from src.flow_fixedpoints import find_all_fixed_points, classify_fixed_points
from src.tasks import make_timings, generate_dual_trials

DT = 0.0225


def analyse_run(sweep_dir, rid, stage, xlim, mem_k0, resid_tol=1e-4):
    model, cfg = load_run(sweep_dir, rid, stage=stage)
    params = low_rank_numpy_params(model)
    ff = torch.zeros(cfg["input_size"])
    ff[-1] = cfg.get("attention_scale", 1.0) * cfg.get("input_scale", 1.0)

    fps, _ = find_all_fixed_points(model, (-xlim, xlim), (-xlim, xlim), ff)
    fps = np.atleast_2d(np.asarray(fps, dtype=float))
    labels, eigvals = classify_fixed_points(model, fps, ff)
    labels = [str(l) for l in labels]

    wells, extras, spirals = [], [], 0
    for f, lab, ev in zip(fps, labels, eigvals):
        if lab != "attractor":
            continue
        k = np.array([float(f[0]), float(f[1])])
        # direct verification — never trust a finder unchecked
        F = np.asarray(low_rank_field_np(params, k[None, :], ff.numpy())).ravel()
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


def behaviour_at_zero(model, cfg):
    """Dual go/nogo at the FIXED boundary 0, in the run's own trained response window."""
    T = make_timings(DT)["dual"]
    half = int(round(0.5 / DT)); co = int(T.n_stim_off[2])
    in_cue = bool(cfg.get("response_in_cue", False)) and not bool(cfg.get("gng_rwd_after_cue", False))
    torch.manual_seed(0)
    X, y, _, names = generate_dual_trials(
        1024, T, input_size=cfg["input_size"], target_rank=cfg["target_rank"],
        noise=cfg["noise"] * float(np.sqrt(1 - np.exp(-2 * 0.075))),
        cue_on_go_input=True, cue_scale=cfg.get("cue_scale", 2.0),
        input_scale=cfg.get("input_scale", 1.0), attention_input=True,
        attention_gated=cfg.get("attention_gated", True),
        windowed_targets=cfg.get("windowed_targets", True),
        decay_to_zero=cfg.get("decay_to_zero", False),
        gng_response=cfg.get("gng_response", True),
        gng_memory=cfg.get("dual_gng_memory", False),
        response_in_cue=cfg.get("response_in_cue", False),
        gng_rwd_after_cue=cfg.get("gng_rwd_after_cue", False))
    model.noise = 0.0
    dev = next(model.parameters()).device
    pred = model(X.to(dev), y.to(dev))[..., -1].detach().cpu()
    names = np.asarray(names).astype(str)
    is_ng = torch.as_tensor(["_nogo_" in n for n in names])
    is_go = torch.as_tensor(["_go_" in n for n in names])
    m = (pred[:, co - half:co] if in_cue else pred[:, co:co + half]).mean(1)
    return ((m[is_ng] <= 0).float().mean().item(), m[is_ng].mean().item(),
            (m[is_go] > 0).float().mean().item())


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
    args = ap.parse_args()

    rows = {}
    jl = os.path.join(args.sweep_dir, "results.jsonl")
    if os.path.exists(jl):
        for line in open(jl):
            r = json.loads(line)
            rows[r["run_id"]] = r.get("accuracy", {}).get("after_dual", {})

    rids = args.run_ids or discover_run_ids(args.sweep_dir)
    n_down = 0
    print(f"VERDICT — {os.path.basename(os.path.normpath(args.sweep_dir))}  stage={args.stage}  "
          f"(memory well = attractor with |κ₀| ≥ {args.mem_k0}; goal = ALL memory wells κ₁ < 0)")
    for rid in rids:
        model, cfg, wells, extras, spirals, pair_ok, all_down = analyse_run(
            args.sweep_dir, rid, args.stage, args.xlim, args.mem_k0)
        n_down += all_down
        acc = rows.get(rid, {})
        beh = ""
        if not args.no_behaviour:
            ng0, ngm, go0 = behaviour_at_zero(model, cfg)
            beh = f"  nogo@0={ng0:.2f}(μ{ngm:+.2f}) go@0={go0:.2f}"
        wstr = " ".join(f"({w[0]:+.2f},{w[1]:+.2f}){'S' if w[2] else ''}" for w in wells) or "NONE"
        estr = f"  extras={len(extras)}:" + " ".join(f"({e[0]:+.1f},{e[1]:+.1f})" for e in extras) if extras else ""
        flag = "ALL-DOWN ✓" if all_down else ("NO PAIR ✗" if not pair_ok else "not down")
        print(f"{rid}: {flag}  mem wells: {wstr}{estr}  spirals={spirals}"
              f"  dual_dpa={acc.get('dual_dpa', float('nan')):.3f}{beh}")
    print(f"SUMMARY: {n_down}/{len(rids)} seeds with all memory wells below the line")


if __name__ == "__main__":
    main()
