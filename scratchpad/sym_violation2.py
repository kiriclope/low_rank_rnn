"""Equivariance of the autonomous kappa-field under the Klein group, on the DISK |k| <= 1.5 (so the
normalization is the field's typical strength, not the near-zero field on the ring).
  v_sigma = ||F(D k) - D F(k)|| / ||F(k)||   (RMS over grid points inside the disk)
Also the even part E(k) = (F(k)+F(-k))/2 split into the constant <n>/2 and the rest (bias-driven for lif),
the readout means, the input-bias rms, and J off-diagonals.
Usage: sym_violation2.py <out.tsv> <sweep>:<rid>:<stage> ...   stage may be 'init' (rebuilt from config)."""
import sys, os, math, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch
from bifurcation_probe import load_run, run_dt_alpha
from src.models import LowRankModel
from src.init import init_dpa_internal_readout_prepost
from src.train import symmetrize_init
from src.flow_field import low_rank_numpy_params, low_rank_field_np
D = {"s1": np.diag([-1.0, 1.0]), "s2": -np.eye(2), "s3": np.diag([1.0, -1.0])}
g = np.linspace(-1.5, 1.5, 31); X, Y = np.meshgrid(g, g); pts = np.stack([X.ravel(), Y.ravel()], 1); pts = pts[np.hypot(*pts.T) <= 1.5]
def build_init(sw, rid):
    cfg = json.load(open(os.path.join(sw, rid, "config.json"))); _, alpha, alpha_rec = run_dt_alpha(cfg)
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
    m.eval(); return m, cfg
out = open(sys.argv[1], "w")
hdr = "run\tstage\tv_s1\tv_s2\tv_s3\tmean_n0\tmean_n1\tE_rms\tE_const_rms\tE_rest_rms\tbias_rms\tJ01\tJ10"
print(hdr); out.write(hdr + "\n")
for spec in sys.argv[2:]:
    sw, rid, stage = spec.split(":")
    try: m, cfg = build_init(sw, rid) if stage == "init" else load_run(sw, rid, stage=stage, device="cpu")
    except Exception as e: print(f"{rid}\t{stage}\tSKIP {str(e)[:50]}"); continue
    _, alpha, _ = run_dt_alpha(cfg); sig = cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha))
    P = low_rank_numpy_params(m); Nv = np.asarray(P["Nvec"]); M = np.asarray(P["M"]); N = M.shape[0]
    J = cfg["gain"] * (Nv.T @ M) / N; mean_n = Nv.mean(0); b = m.wi.bias.detach().numpy()
    ffz = np.zeros(P["Wi"].shape[1])
    F_of = lambda q: np.stack([np.asarray(low_rank_field_np(P, p, ff_input=ffz, noise_sigma=sig)).ravel()[:2] for p in q])
    F = F_of(pts); nF = np.linalg.norm(F)
    v = {k: np.linalg.norm(F_of(pts @ Dk.T) - F @ Dk.T) / nF for k, Dk in D.items()}
    E = 0.5 * (F + F_of(-pts)); Ec = np.broadcast_to(0.5 * mean_n[:2], E.shape); Er = E - Ec
    rms = lambda A: float(np.sqrt((A ** 2).sum(1).mean()))
    line = (f"{rid}\t{stage}\t{v['s1']:.3f}\t{v['s2']:.3f}\t{v['s3']:.3f}\t{mean_n[0]:+.4f}\t{mean_n[1]:+.4f}\t"
            f"{rms(E):.4f}\t{rms(Ec):.4f}\t{rms(Er):.4f}\t{np.sqrt((b**2).mean()):.3f}\t{J[0,1]:+.3f}\t{J[1,0]:+.3f}")
    print(line, flush=True); out.write(line + "\n"); out.flush()
