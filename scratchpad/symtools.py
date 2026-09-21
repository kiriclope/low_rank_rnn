"""Shared helpers for the symmetry figures: rebuild an init from config (+ overrides), swap the transfer
function, evaluate the kappa-field on point sets, and the equivariance residuals."""
import sys, os, json, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch
from bifurcation_probe import load_run, run_dt_alpha
from src.models import LowRankModel
from src.init import init_dpa_internal_readout_prepost
from src.train import symmetrize_init
from src.flow_field import low_rank_numpy_params, low_rank_field_np
D = {"s1": np.diag([-1.0, 1.0]), "s2": -np.eye(2), "s3": np.diag([1.0, -1.0])}
from scipy.special import erf as _erf
PHI = {"lif": lambda u: 0.5 * (1 + _erf(u / math.sqrt(2))), "tanh": np.tanh}
def disk(n=31, r=1.5):
    g = np.linspace(-r, r, n); X, Y = np.meshgrid(g, g); p = np.stack([X.ravel(), Y.ravel()], 1)
    return p[np.hypot(*p.T) <= r]
def sig_of(cfg):
    _, alpha, _ = run_dt_alpha(cfg); return cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha))
def make_model(cfg, N=None, nonlinearity=None, device="cpu"):
    _, alpha, alpha_rec = run_dt_alpha(cfg)
    return LowRankModel(input_size=cfg["input_size"], hidden_size=N or cfg["hidden_size"], output_size=0, rank=cfg.get("rank", 2),
                        gain=cfg["gain"], alpha=alpha, alpha_rec=alpha_rec, noise=0.0, nonlinearity=nonlinearity or cfg["nonlinearity"],
                        nl_gamma=cfg.get("nl_gamma", 0.0), use_unit_bias=cfg.get("use_unit_bias", False),
                        unit_bias_trainable=cfg.get("unit_bias_trainable", False), use_rec_scale=cfg.get("use_rec_scale", False),
                        integrate=cfg.get("integrate", "both"), device=device)
def build_init(sw, rid, N=None, seed=None, symmetry="cfg", **over):
    """Structured init exactly as the sweep builds it; N / seed / init kwargs may be overridden."""
    cfg = json.load(open(os.path.join(sw, rid, "config.json")))
    seed = cfg["seed"] if seed is None else seed
    torch.manual_seed(seed); m = make_model(cfg, N=N)
    kw = dict(memory_lambda=cfg["memory_lambda"], decision_lambda=cfg["decision_lambda"], target_mn_corr=cfg["target_mn_corr"],
              target_out_mn_corr=cfg["target_out_mn_corr"], sample_scale=cfg["sample_scale"], test_scale=cfg["test_scale"],
              decision_readout_mean=cfg.get("decision_readout_mean", 0.0), mix_strength=cfg["mix_strength"],
              noise_scale_mn=1.0, noise_scale_in=1.0, rwd_input_scale=cfg.get("rwd_input_scale", 1.0))
    if cfg.get("readout_scale") is not None: kw["readout_scale"] = cfg["readout_scale"]
    kw.update(over)
    init_dpa_internal_readout_prepost(m, mem=0, out=1, gng=None, gng_lambda=cfg.get("gng_lambda", 0.0), seed=seed, verbose=False, **kw)
    sym = (cfg.get("symmetry") or ("pair" if cfg.get("mirror_tying") else "")).lower() if symmetry == "cfg" else (symmetry or "")
    if sym: symmetrize_init(m, sym)
    m.eval(); return m, cfg
def swap_phi(model, cfg, nonlinearity, zero_bias=False):
    """Same parameters, another transfer function (and optionally b = 0)."""
    m2 = make_model(cfg, N=model.m.shape[0], nonlinearity=nonlinearity)
    m2.load_state_dict(model.state_dict(), strict=False)
    if zero_bias:
        with torch.no_grad(): m2.wi.bias.zero_()
    m2.eval(); cfg2 = dict(cfg); cfg2["nonlinearity"] = nonlinearity; return m2, cfg2
def field_fn(model, sig=0.0):
    P = low_rank_numpy_params(model); ffz = np.zeros(P["Wi"].shape[1])
    def F_of(pts):
        return np.stack([np.asarray(low_rank_field_np(P, p, ff_input=ffz, noise_sigma=sig)).ravel()[:2] for p in np.atleast_2d(pts)])
    return F_of, P
def residuals(F_of, pts):
    F = F_of(pts); nF = np.linalg.norm(F)
    return {k: float(np.linalg.norm(F_of(pts @ Dk.T) - F @ Dk.T) / nF) for k, Dk in D.items()}
def raw_field(M, Nv, b, g, phi, pts):
    """Noise-free kappa-field from raw arrays: F(k) = (1/N) n^T phi(g(M k + b)) - k."""
    pts = np.atleast_2d(pts); N = M.shape[0]
    out = np.empty_like(pts)
    for i, k in enumerate(pts):
        out[i] = (Nv.T @ phi(g * (M @ k + b))) / N - k
    return out
def raw_residual(M, Nv, b, g, phi, pts, key="s2"):
    F = raw_field(M, Nv, b, g, phi, pts); Dk = D[key]
    return float(np.linalg.norm(raw_field(M, Nv, b, g, phi, pts @ Dk.T) - F @ Dk.T) / np.linalg.norm(F))
