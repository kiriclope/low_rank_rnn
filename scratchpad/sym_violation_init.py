"""Same measurement as sym_violation.py, on the INITIAL network of a run (rebuilt from config.json with the
run's seed, structured init, then symmetrize_init if the run used one). Usage: sym_violation_init.py <sweep> <rid> ..."""
import sys, os, math, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch
from bifurcation_probe import run_dt_alpha
from src.models import LowRankModel
from src.init import init_dpa_internal_readout_prepost
from src.train import symmetrize_init
from src.flow_field import low_rank_numpy_params, low_rank_field_np
R = 1.0; NTH = 36; D = {"s1": np.diag([-1.0, 1.0]), "s2": -np.eye(2), "s3": np.diag([1.0, -1.0])}
th = np.linspace(0, 2 * np.pi, NTH, endpoint=False); circ = R * np.stack([np.cos(th), np.sin(th)], 1)
sw = sys.argv[1]
print("run\tv_s1\tv_s2\tv_s3\tmean_n0\tmean_n1\tJ01\tJ10")
for rid in sys.argv[2:]:
    cfg = json.load(open(os.path.join(sw, rid, "config.json")))
    _, alpha, alpha_rec = run_dt_alpha(cfg); sig = cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha))
    torch.manual_seed(cfg["seed"])
    m = LowRankModel(input_size=cfg["input_size"], hidden_size=cfg["hidden_size"], output_size=0, rank=cfg.get("rank", 2),
                     gain=cfg["gain"], alpha=alpha, alpha_rec=alpha_rec, noise=0.0, nonlinearity=cfg["nonlinearity"],
                     nl_gamma=cfg.get("nl_gamma", 0.0), use_unit_bias=cfg.get("use_unit_bias", False),
                     unit_bias_trainable=cfg.get("unit_bias_trainable", False), use_rec_scale=cfg.get("use_rec_scale", False),
                     integrate=cfg.get("integrate", "both"), device="cpu")
    init_dpa_internal_readout_prepost(m, mem=0, out=1, gng=None, gng_lambda=cfg.get("gng_lambda", 0.0),
        memory_lambda=cfg["memory_lambda"], decision_lambda=cfg["decision_lambda"], target_mn_corr=cfg["target_mn_corr"],
        target_out_mn_corr=cfg["target_out_mn_corr"],
        **({} if cfg.get("readout_scale") is None else {"readout_scale": cfg["readout_scale"]}),
        sample_scale=cfg["sample_scale"], test_scale=cfg["test_scale"], decision_readout_mean=cfg.get("decision_readout_mean", 0.0),
        mix_strength=cfg["mix_strength"], noise_scale_mn=1.0, noise_scale_in=1.0, rwd_input_scale=cfg.get("rwd_input_scale", 1.0),
        seed=cfg["seed"], verbose=False)
    sym = (cfg.get("symmetry") or ("pair" if cfg.get("mirror_tying") else "")).lower()
    if sym: symmetrize_init(m, sym)
    m.eval(); P = low_rank_numpy_params(m); Nv = np.asarray(P["Nvec"]); M = np.asarray(P["M"]); N = M.shape[0]
    J = cfg["gain"] * (Nv.T @ M) / N
    ffz = np.zeros(P["Wi"].shape[1])
    F_of = lambda pts: np.stack([np.asarray(low_rank_field_np(P, p, ff_input=ffz, noise_sigma=sig)).ravel()[:2] for p in pts])
    F = F_of(circ); nF = np.linalg.norm(F)
    v = {k: np.linalg.norm(F_of(circ @ Dk.T) - F @ Dk.T) / nF for k, Dk in D.items()}
    mn = Nv.mean(0)
    print(f"{rid}\t{v['s1']:.3f}\t{v['s2']:.3f}\t{v['s3']:.3f}\t{mn[0]:+.4f}\t{mn[1]:+.4f}\t{J[0,1]:+.3f}\t{J[1,0]:+.3f}", flush=True)
