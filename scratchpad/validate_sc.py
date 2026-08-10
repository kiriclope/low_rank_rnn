"""Validate the self-consistent (DMFT) noise closure against a direct noisy simulation.
Predicts, at a memory well: (i) the noise-shifted fixed point κ*_sc and (ii) the mode-overlap
covariance C = Cov(δκ). Simulates the real model with input noise σ from that well and compares
the stationary mean(κ) and Cov(κ). Read-only.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
from scipy.optimize import root
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt, make_input
from src.dynamics import (low_rank_numpy_params, low_rank_field_np, low_rank_field_sc_np,
                          solve_sc_variance, _sc_a_bar)

DEV = "cpu"; sd = "results/dual/sweep_cue"
meta = [m for m in _load_sweep_meta(sd) if m.run_id == "s0_r3cue"][0]
model = _build_model(meta, DEV); _load_ckpt(model, sd, "expert", meta.run_id, DEV)
p = low_rank_numpy_params(model); sigma = meta.noise_sigma()
N = model.hidden_size; R = model.m.shape[1]
m_t, n_t = model.m.detach(), model.n.detach()
print(f"{meta.run_id}  σ={sigma:.4f}  N={N}  R={R}")

ff = make_input(meta.input_size, None, 1.0, device=DEV, dtype=torch.float32)
if meta.attention_input: ff[-1] = getattr(meta, "attention_scale", 1.0)
ff_np = ff.detach().cpu().numpy().astype(np.float64)


def C_predict(kstar):
    """Recompute the predicted mode covariance C at κ* (mirrors solve_sc_variance internals)."""
    a_bar = _sc_a_bar(p, kstar[None], ff_np)
    Delta, sigt, phip = solve_sc_variance(p, a_bar, sigma)
    M, Nvec, Wi, Ai, g = p["M"], p["Nvec"], p["Wi"], np.asarray(p["Ai"], float), p["gain"]
    if Ai.ndim == 0: Ai = np.full(N, float(Ai))
    Ut = (1.0 / N) * np.einsum('jr,bj,j,jk->brk', Nvec, phip, Ai, Wi)      # (1,R,n_in)
    eig = np.linalg.eigvals(sigt).real
    G = np.eye(R)[None] if eig.max() >= 0.98 else np.linalg.inv(np.eye(R)[None] - sigt)
    Cov_b = (g ** 2 * sigma ** 2) * np.einsum('brk,bsk->brs', Ut, Ut)
    C = np.einsum('brs,bst,but->brt', G, Cov_b, G)[0]
    return C, Delta.mean(), float(eig.max())


def C_predict_sig(kstar, sig):
    a_bar = _sc_a_bar(p, kstar[None], ff_np)
    Delta, sigt, phip = solve_sc_variance(p, a_bar, sig)
    M, Nvec, Wi, Ai, g = p["M"], p["Nvec"], p["Wi"], np.asarray(p["Ai"], float), p["gain"]
    if Ai.ndim == 0: Ai = np.full(N, float(Ai))
    Ut = (1.0 / N) * np.einsum('jr,bj,j,jk->brk', Nvec, phip, Ai, Wi)
    eig = np.linalg.eigvals(sigt).real
    G = np.eye(R)[None] if eig.max() >= 0.98 else np.linalg.inv(np.eye(R)[None] - sigt)
    C = np.einsum('brs,bst,but->brt', G, (g**2*sig**2)*np.einsum('brk,bsk->brs', Ut, Ut), G)[0]
    return C, Delta.mean(), float(eig.max())


@torch.no_grad()
def simulate(kstar, sig, B=400, T=1500, burn=500):
    """B parallel noisy trajectories from κ*; stationary mean(κ), Cov(κ), and fraction still near κ*."""
    dt = m_t.dtype
    h = torch.tensor(kstar, dtype=dt).repeat(B, 1) @ m_t.T
    ffb = torch.as_tensor(ff_np, dtype=dt)[None, :].repeat(B, 1)
    drive = model.Ai * model.wi(ffb) if model.wi is not None else torch.zeros_like(h)
    rates = model.nonlinearity(model.gain * (drive + h))
    samples = []
    for t in range(T):
        ff_t = ffb + sig * torch.randn(B, meta.input_size, dtype=dt)
        rates, h = model.update_dynamics(ff_t, h, rates)
        if t >= burn:
            samples.append((rates @ n_t / N).cpu().numpy())
    K = np.stack(samples).reshape(-1, R)
    held = np.mean(np.sign(K[:, 0]) == np.sign(kstar[0]))          # fraction still on κ0's side
    return K.mean(0), np.cov(K.T), held


kdet = root(lambda k: low_rank_field_np(p, k[None], ff_input=ff_np[None]).ravel(),
            np.array([+1.0, -0.25, -0.70])).x                     # stable +samp nogo DOWN well
print(f"\ndeterministic well κ* = {np.round(kdet,3)}  (validate C on this stable well)")
print(f"{'σ':>7} {'⟨Δ⟩':>7} {'maxeigσ̃':>8} {'held':>6}   Cov_pred diag         Cov_sim diag        relerr")
for frac in [0.125, 0.25, 0.5, 1.0]:
    sig = sigma * frac
    Cpred, dmean, smax = C_predict_sig(kdet, sig)
    msim, Csim, held = simulate(kdet, sig)
    rel = np.linalg.norm(Cpred - Csim) / (np.linalg.norm(Csim) + 1e-9)
    print(f"{sig:7.3f} {dmean:7.2f} {smax:+8.2f} {held:6.2f}   "
          f"{np.round(np.diag(Cpred),4)}  {np.round(np.diag(Csim),4)}  {rel:.3f}")
