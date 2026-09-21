"""§38 readout: for every rulesym run (naive checkpoint = after the rule stage), the three Z2 residuals of
the autonomous field, the attractors with their orbit structure, J, the go/nogo/cue overlaps, the GMM
population count, and the accuracies. Usage: rulesym_readout.py <sweep_dir> [out.tsv]"""
import sys, os, json, math, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/leon/rnn"); sys.path.insert(0, "/home/leon/rnn/scratchpad")
import numpy as np
from symtools import *
from bifurcation_probe import find_wells
sw = sys.argv[1]; out = open(sys.argv[2], "w") if len(sys.argv) > 2 else None
acc = {}
if os.path.exists(f"{sw}/results.jsonl"):
    for l in open(f"{sw}/results.jsonl"):
        r = json.loads(l); acc[r["run_id"]] = r.get("accuracy", {}).get("after_gng", {})
PAT = os.environ.get("PAT", "rulesym"); rids = sorted(os.path.basename(d) for d in glob.glob(f"{sw}/s*_{PAT}_*") if os.path.exists(f"{d}/naive_{os.path.basename(d)}.pth"))
hdr = "run\tgng_acc\tv_s1\tv_s2\tv_s3\tJ00\tJ01\tJ10\tJ11\tn0.go\tn0.nogo\tn1.go\tn1.nogo\tn1.cue\tbicK\tattractors\torbits"
print(hdr); out and out.write(hdr + "\n")
def orbit_report(w, tol=0.12):
    """Pair up attractors under each candidate action; report which actions map the set onto itself."""
    rep = []
    for name, Dk in D.items():
        ok = all(any(np.linalg.norm(Dk @ f - g) < tol for g in w) for f in w) if w else False
        rep.append(f"{name}:{'yes' if ok else 'no'}")
    return " ".join(rep)
for rid in rids:
    m, cfg = load_run(sw, rid, stage="naive", device="cpu"); sig = sig_of(cfg)
    F_of, P = field_fn(m, sig); v = residuals(F_of, disk(31))
    M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"]); Wi = np.asarray(P["Wi"]); N = M.shape[0]
    J = cfg["gain"] * (Nv.T @ M) / N; ov = (Nv.T @ Wi) / N
    w = [np.array(f) for f, k, t in find_wells(m, cfg, xlim=2.5, n_seeds=61, noise_sigma=sig, with_eigs=True) if str(k).lower().startswith(("stable", "attract"))]
    w = [f for f in w if np.linalg.norm(f) > 0.15]
    try:
        from sklearn.mixture import GaussianMixture
        V = np.c_[M[:, 0], M[:, 1], Nv[:, 0], Nv[:, 1]]
        bic = {k: GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(V).bic(V) for k in (1, 2, 3, 4)}
        K = min(bic, key=bic.get)
    except Exception: K = -1
    a = acc.get(rid, {}); ga = a.get("gng", float("nan")) if isinstance(a, dict) else float("nan")
    line = (f"{rid}\t{ga:.3f}\t{v['s1']:.3f}\t{v['s2']:.3f}\t{v['s3']:.3f}\t{J[0,0]:.2f}\t{J[0,1]:+.2f}\t{J[1,0]:+.2f}\t{J[1,1]:.2f}\t"
            f"{ov[0,4]:+.2f}\t{ov[0,5]:+.2f}\t{ov[1,4]:+.2f}\t{ov[1,5]:+.2f}\t{ov[1,6]:+.2f}\t{K}\t"
            + " ".join(f"({f[0]:+.2f},{f[1]:+.2f})" for f in w) + f"\t{orbit_report(w)}")
    print(line, flush=True); out and (out.write(line + "\n"), out.flush())
