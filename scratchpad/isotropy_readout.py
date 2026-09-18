"""Does DPA training preserve the ISOTROPY of the rank-2 covariance (i.e. the ring)?
Per run: the trained overlap matrix J = g·nᵀm/N, the per-mode factor scales σ(m_i), σ(n_i), the angular
anisotropy of the autonomous input-noise-averaged field (range of the radial component around circles,
compared with the 1/√N finite-size floor), the fixed points with their relaxation times in seconds
(protocol rule 12), and DPA accuracy.  Usage: isotropy_readout.py <sweep_dir> [<run_id> ...]"""
import sys, os, json, math, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch
from bifurcation_probe import load_run, run_dt_alpha, autonomous_ff, find_wells
from src.flow_field import low_rank_numpy_params, low_rank_field_np
sw = sys.argv[1]; STAGE = os.environ.get("STAGE", "dpa")
rids = sys.argv[2:] or sorted(os.path.basename(d) for d in glob.glob(f"{sw}/s*") if os.path.isdir(d))
TH = np.linspace(0, 2 * np.pi, 32, endpoint=False)
def acc(rid):
    for ln in open(f"{sw}/results.jsonl"):
        r = json.loads(ln)
        if r["run_id"] == rid: return r["accuracy"]["after_dpa"]["dpa"]
    return float("nan")
print(f"stage = {STAGE}")
print(f"{'run':18s} {'λ init':>6s} | {'J00':>5s} {'J11':>5s} {'J01':>6s} {'J10':>6s} | "
      f"{'σm0':>5s} {'σm1':>5s} {'σn0':>5s} {'σn1':>5s} {'iso':>5s} | {'anis@R':>8s} {'floor':>6s} | {'dpa':>5s}")
for rid in rids:
    try: m, cfg = load_run(sw, rid, stage=STAGE, device="cpu")
    except Exception as e: print(f"{rid:18s} load failed: {str(e)[:60]}"); continue
    m.eval(); dt, alpha, _ = run_dt_alpha(cfg); sig = cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha))
    P = low_rank_numpy_params(m); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); N = M.shape[0]
    J = cfg["gain"] * (Nv.T @ M) / N
    sm, sn = M.std(0), Nv.std(0)
    iso = max(sm[0] / sm[1], sm[1] / sm[0], sn[0] / sn[1], sn[1] / sn[0])      # 1.0 = isotropic factors
    ff = autonomous_ff(cfg); ff = ff.detach().cpu().numpy() if hasattr(ff, "detach") else np.asarray(ff)
    wells = [(f, k, t) for f, k, t in find_wells(m, cfg, xlim=2.5, n_seeds=61, noise_sigma=sig, with_eigs=True)
             if str(k).lower().startswith(("stable", "attract"))]
    R = float(np.mean([math.hypot(*f) for f, _, _ in wells])) if wells else 1.0
    pts = np.stack([R * np.cos(TH), R * np.sin(TH)], 1)
    F = np.stack([np.asarray(low_rank_field_np(P, p, ff_input=ff, noise_sigma=sig)).ravel()[:2] for p in pts])
    anis = float(np.ptp((F * pts).sum(1) / R))
    floor = 0.022 * math.sqrt(8192 / N)                                        # §33d scaling, N=8192 → 0.022
    print(f"{rid:18s} {cfg['memory_lambda']:6.1f} | {J[0,0]:5.2f} {J[1,1]:5.2f} {J[0,1]:+6.2f} {J[1,0]:+6.2f} | "
          f"{sm[0]:5.2f} {sm[1]:5.2f} {sn[0]:5.2f} {sn[1]:5.2f} {iso:5.2f} | {anis:8.3f} {floor:6.3f} | {acc(rid):5.3f}")
    delay = float(cfg.get("stim_on", [0, 0])[1]) if isinstance(cfg.get("stim_on"), list) else 5.0
    for f, k, (ts, tf) in sorted(wells, key=lambda w: -math.hypot(*w[0]))[:4]:
        print(f"{'':18s}   attractor ({f[0]:+.2f},{f[1]:+.2f})  r {math.hypot(*f):.2f}  "
              f"τ {ts:.1f}/{tf:.2f}s  ratio {ts/max(tf,1e-9):.0f}×{'   RING-LIKE (τ_slow > delay)' if ts > delay else ''}")
