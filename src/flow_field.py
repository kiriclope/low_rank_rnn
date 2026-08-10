from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.special
from scipy.optimize import root

from .tasks import TaskTiming

"""Shared rank-general low-rank flow ENGINE: analytic field + Jacobian, input-noise mean
field (exact + self-consistent), κ-projection, sim primitives, integrate_kappa_trajectories.
Imported by flow_fixedpoints / flow_rank2 / flow_rank3 and re-exported by dynamics.py."""

def _model_device(model):
    return next(model.parameters()).device


def _model_dtype(model):
    return next(model.parameters()).dtype


def make_input(input_size, active_dims=None, value=1.0, device=None, dtype=None):
    x = torch.zeros(input_size, device=device, dtype=dtype)
    if active_dims is not None:
        x[list(active_dims)] = value
    return x


# ---------------------------------------------------------------------------
# Low-rank parameter extraction
# ---------------------------------------------------------------------------


def low_rank_numpy_params(model):
    """
    Extract parameters for low-rank vector-field calculations.

    Convention:
        h = M κ
        r = tanh(gain * (A_i(W_i x + b_i) + h))
        Ψ(κ; x) = n^T r / N
        κ⁺ = κ + β(Ψ(κ; x) - κ)
        β = 1 - exp(-alpha)
    """
    alpha = (
        float(model.alpha.detach().cpu())
        if torch.is_tensor(model.alpha) else float(model.alpha)
    )

    if hasattr(model, "_constrained_n"):
        n_eff = model._constrained_n()
    elif hasattr(model, "signed_n"):
        n_eff = model.signed_n()
    else:
        n_eff = model.n

    try:
        Ai = model.Ai.detach().cpu().numpy().astype(np.float64)
    except AttributeError:
        Ai = float(model.Ai) if hasattr(model, "Ai") else 1.0

    gain = (
        float(model.gain.detach().cpu()) if torch.is_tensor(model.gain)
        else float(model.gain) if hasattr(model, "gain") else 1.0
    )

    # phi_pp (φ'') and phi_ppp (φ''') are used ONLY by the noise-corrected mean field
    # (Ψ_σ = Ψ_0 + ½·(1/N)Σ n_i φ''(a_i) s_i²) and its Jacobian (φ''' term). Where a closed
    # φ'' isn't provided they default to 0 → noise correction silently off for that nl.
    nl_str = getattr(model, "nonlinearity_str", "tanh")
    phi_pp_np = phi_ppp_np = lambda u: np.zeros_like(np.asarray(u, dtype=np.float64))
    # noise_compress c: for a Gaussian-CDF φ the Gaussian input-noise average is EXACT —
    # ⟨φ(ā+η)⟩ = φ(ā/√(1+c·s²)) (c=1 lif, 2 erf, 2π lif_sc). None ⇒ fall back to the φ'' Taylor term.
    noise_compress = None
    if nl_str == "relu":
        phi_np       = lambda u: np.maximum(u, 0.0)
        phi_prime_np = lambda u: (u > 0).astype(np.float64)
        # φ''=0 a.e. (kink at 0 ignored) → relu is noise-transparent at leading order
    elif nl_str == "softplus":
        # numerically stable: softplus(u) = u for u>>0, log1p(exp(u)) otherwise
        _sig         = lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -20.0, 20.0)))
        phi_np       = lambda u: np.where(u > 20.0, u, np.log1p(np.exp(np.minimum(u, 20.0))))
        phi_prime_np = _sig
        phi_pp_np    = lambda u: _sig(u) * (1.0 - _sig(u))
        phi_ppp_np   = lambda u: _sig(u) * (1.0 - _sig(u)) * (1.0 - 2.0 * _sig(u))
    elif nl_str == "erf":
        _2_sqrt_pi   = 2.0 / np.sqrt(np.pi)
        phi_np       = scipy.special.erf
        _e           = lambda u: np.exp(-np.clip(u, -20.0, 20.0) ** 2)
        phi_prime_np = lambda u: _2_sqrt_pi * _e(u)
        phi_pp_np    = lambda u: _2_sqrt_pi * (-2.0 * u) * _e(u)
        phi_ppp_np   = lambda u: _2_sqrt_pi * (4.0 * u ** 2 - 2.0) * _e(u)
        noise_compress = 2.0                                    # erf(x)=2Φ(x√2)−1 ⇒ c=2
    elif nl_str == "elu":
        # ELU: x for x>0, exp(x)-1 for x<=0 (alpha=1); φ'= 1 for x>0, exp(x) for x<=0
        _en          = lambda u: np.exp(np.minimum(u, 0.0))
        phi_np       = lambda u: np.where(u > 0, u, _en(u) - 1.0)
        phi_prime_np = lambda u: np.where(u > 0, 1.0, _en(u))
        phi_pp_np    = lambda u: np.where(u > 0, 0.0, _en(u))
        phi_ppp_np   = lambda u: np.where(u > 0, 0.0, _en(u))
    elif nl_str == "lif":
        # Brunel erfc approximation: Gaussian CDF, φ ∈ [0,1], φ'(0) = 1/√(2π) ≈ 0.399
        _sqrt2    = np.sqrt(2.0)
        _sqrt2pi  = np.sqrt(2.0 * np.pi)
        _g        = lambda u: np.exp(-np.clip(u, -20.0, 20.0) ** 2 / 2.0) / _sqrt2pi   # φ' = N(u;0,1)
        phi_np       = lambda u: 0.5 * (1.0 + scipy.special.erf(u / _sqrt2))
        phi_prime_np = _g
        phi_pp_np    = lambda u: -u * _g(u)                      # φ'' = -u·N(u)
        phi_ppp_np   = lambda u: (u ** 2 - 1.0) * _g(u)          # φ''' = (u²-1)·N(u)
        noise_compress = 1.0                                     # lif=Φ ⇒ exact c=1
    elif nl_str == "lif_sc":
        # Rescaled LIF: φ(x)=(1+erf(x√π))/2, range [0,1], φ'(0)=1 (matches tanh at origin)
        _sqrtpi = np.sqrt(np.pi)
        _gc     = lambda u: np.exp(-np.pi * np.clip(u, -20.0, 20.0) ** 2)             # φ' = exp(-π u²)
        phi_np       = lambda u: 0.5 * (1.0 + scipy.special.erf(np.clip(u, -20.0, 20.0) * _sqrtpi))
        phi_prime_np = _gc
        phi_pp_np    = lambda u: -2.0 * np.pi * u * _gc(u)                            # φ'' = -2πu·φ'
        phi_ppp_np   = lambda u: 2.0 * np.pi * (2.0 * np.pi * u ** 2 - 1.0) * _gc(u)  # φ''' = 2π(2πu²-1)φ'
        noise_compress = 2.0 * np.pi                                                  # lif_sc=Φ(x√2π) ⇒ c=2π
    elif nl_str == "tanh_asym":
        # φ = tanh(u) + γ·tanh²(u);  φ' = (1-tanh²)(1 + 2γ·tanh)
        g = float(getattr(model, "nl_gamma", 0.0))
        phi_np       = lambda u, g=g: np.tanh(u) + g * np.tanh(u) ** 2
        phi_prime_np = lambda u, g=g: (1.0 - np.tanh(u) ** 2) * (1.0 + 2.0 * g * np.tanh(u))
    else:  # tanh (default)
        phi_np       = np.tanh
        phi_prime_np = lambda u: 1.0 - np.tanh(u) ** 2
        phi_pp_np    = lambda u: -2.0 * np.tanh(u) * (1.0 - np.tanh(u) ** 2)
        phi_ppp_np   = lambda u: (6.0 * np.tanh(u) ** 2 - 2.0) * (1.0 - np.tanh(u) ** 2)

    N = model.m.shape[0]
    unit_bias = getattr(model, "unit_bias", None)
    if unit_bias is None:
        unit_bias_np = np.zeros(N, dtype=np.float64)
    else:
        unit_bias_np = unit_bias.detach().cpu().numpy().astype(np.float64)

    # Effective loading includes the per-mode recurrent scale: M_eff = rec_scale · m
    # (readout Nvec = n is unscaled). Ones when rec_scale is disabled → unchanged.
    rec_scale = getattr(model, "rec_scale", None)
    m_eff = model.m if rec_scale is None else (rec_scale * model.m)

    return {
        "M":         m_eff.detach().cpu().numpy().astype(np.float64),
        "Nvec":      n_eff.detach().cpu().numpy().astype(np.float64),
        "Wi":        model.wi.weight.detach().cpu().numpy().astype(np.float64),
        "bi":        model.wi.bias.detach().cpu().numpy().astype(np.float64),
        "Ai":        Ai,
        "gain":      gain,
        "beta":      1.0 - np.exp(-alpha),
        "unit_bias": unit_bias_np,
        "phi":       phi_np,
        "phi_prime": phi_prime_np,
        "phi_pp":    phi_pp_np,
        "phi_ppp":   phi_ppp_np,
        "noise_compress": noise_compress,
    }


# ---------------------------------------------------------------------------
# Low-rank vector field and Jacobians
# ---------------------------------------------------------------------------


def low_rank_field_np(params, kappa, ff_input=None, include_beta=False, noise_sigma=0.0):
    """
    Vector field in κ-coordinates:  F(κ; x) = Ψ(κ; x) - κ

    If include_beta=True, returns βF, matching the discrete update increment.

    noise_sigma > 0 adds the analytic INPUT-NOISE mean-field correction (Gaussian/Itô term):
    each neuron's drive aᵢ acquires variance sᵢ² = g²·Aᵢ²·σ²·‖wᵢ‖² from input noise ξ~N(0,σ²),
    so ⟨φ(aᵢ)⟩ = φ(āᵢ) + ½φ''(āᵢ)sᵢ² and
        Ψ_σ(κ) = Ψ_0(κ) + (1/2N) Σᵢ nᵢ φ''(āᵢ) sᵢ² .
    Leading (direct-input) term only; the recurrent/self-consistent variance (DMFT) is higher order.
    """
    M, Nvec = params["M"], params["Nvec"]
    Wi, bi  = params["Wi"], params["bi"]
    Ai, gain= params["Ai"], params["gain"]

    kappa      = np.asarray(kappa, dtype=np.float64)
    orig_shape = kappa.shape
    rank       = M.shape[1]
    kappa_flat = kappa.reshape(-1, rank)

    ff_input = (
        np.zeros((1, Wi.shape[1]), dtype=np.float64) if ff_input is None
        else np.atleast_2d(np.asarray(ff_input, dtype=np.float64))
    )   # (K, input_size); K>1 ⇒ NOISE-AVERAGED field E_x[Ψ(κ)] over K input draws

    phi = params.get("phi", np.tanh)
    ub  = params.get("unit_bias", 0.0)
    h   = kappa_flat @ M.T                                  # (B, N)

    phi_pp    = params.get("phi_pp")
    compress  = params.get("noise_compress")
    use_noise = bool(noise_sigma) and noise_sigma > 0.0 and (compress is not None or phi_pp is not None)
    s2    = ((gain ** 2) * (np.asarray(Ai, dtype=np.float64) ** 2)
             * (float(noise_sigma) ** 2) * np.sum(Wi ** 2, axis=1)) if use_noise else None   # (N,)
    denom = np.sqrt(1.0 + compress * s2) if (use_noise and compress is not None) else None    # (N,)

    psi = np.zeros((kappa_flat.shape[0], rank), dtype=np.float64)
    for xk in ff_input:                                     # average φ over the K input draws
        input_drive = Ai * (xk @ Wi.T + bi)                 # (N,)
        a   = gain * (input_drive[None, :] + h) + ub        # (B, N) full drive
        if not use_noise:
            psi += phi(a) @ Nvec / M.shape[0]
        elif denom is not None:                             # EXACT Gaussian resummation: φ(a/√(1+c·s²))
            psi += phi(a / denom[None, :]) @ Nvec / M.shape[0]
        else:                                               # leading Taylor: φ(a) + ½φ''(a)s²
            psi += (phi(a) + 0.5 * phi_pp(a) * s2[None, :]) @ Nvec / M.shape[0]
    psi /= ff_input.shape[0]

    field = psi - kappa_flat
    if include_beta:
        field = params["beta"] * field
    return field.reshape(orig_shape)


def low_rank_jacobian_flow_np(params, kappa, ff_input=None, noise_sigma=0.0):
    """Jacobian of F(κ; x) = Ψ(κ; x) - κ.

    noise_sigma > 0 matches the noise-corrected field: the effective per-neuron slope becomes
    φ'(aᵢ) + ½φ'''(aᵢ)sᵢ², sᵢ² = g²Aᵢ²σ²‖wᵢ‖². For lif φ'''(0)<0, so noise LOWERS the effective
    gain — which is what can turn a bistable well marginal/unstable (saddle-node)."""
    M, Nvec = params["M"], params["Nvec"]
    Wi, bi  = params["Wi"], params["bi"]
    Ai, gain= params["Ai"], params["gain"]

    kappa    = np.asarray(kappa, dtype=np.float64).reshape(-1)
    ff_input = (
        np.zeros((1, Wi.shape[1]), dtype=np.float64) if ff_input is None
        else np.atleast_2d(np.asarray(ff_input, dtype=np.float64))
    )   # (K, input_size); K>1 ⇒ noise-averaged Jacobian (matches the noise-averaged field)

    phi_prime_fn = params.get("phi_prime", lambda u: 1.0 - np.tanh(u) ** 2)
    ub          = params.get("unit_bias", 0.0)
    phi_ppp   = params.get("phi_ppp")
    phi_fn    = params.get("phi", np.tanh)
    compress  = params.get("noise_compress")
    use_noise = bool(noise_sigma) and noise_sigma > 0.0 and (compress is not None or phi_ppp is not None)
    s2    = ((gain ** 2) * (np.asarray(Ai, dtype=np.float64) ** 2)
             * (float(noise_sigma) ** 2) * np.sum(Wi ** 2, axis=1)) if use_noise else None
    denom = np.sqrt(1.0 + compress * s2) if (use_noise and compress is not None) else None
    J = np.zeros((M.shape[1], M.shape[1]), dtype=np.float64)
    for xk in ff_input:
        input_drive = Ai * (xk @ Wi.T + bi)
        u           = gain * (input_drive + M @ kappa) + ub
        if not use_noise:
            slope = phi_prime_fn(u)
        elif denom is not None:                              # d/dκ φ(a/√(1+cs²)) = φ'(a/·)·(gM)/√(1+cs²)
            slope = phi_prime_fn(u / denom) / denom
        else:                                                # φ' → φ' + ½φ'''·s²
            slope = phi_prime_fn(u) + 0.5 * phi_ppp(u) * s2
        J += Nvec.T @ (slope[:, None] * (gain * M)) / M.shape[0]
    J /= ff_input.shape[0]
    J -= np.eye(M.shape[1])
    return J


def low_rank_jacobian_map_np(params, kappa, ff_input=None):
    """Jacobian of the discrete map κ⁺ = κ + βF(κ)."""
    J_flow = low_rank_jacobian_flow_np(params, kappa, ff_input=ff_input)
    return np.eye(J_flow.shape[0]) + params["beta"] * J_flow


# ---------------------------------------------------------------------------
# SELF-CONSISTENT (DMFT) noise mean field — input noise + recurrent variance
#
# ⚠ EXPERIMENTAL / QUALITATIVE ONLY (as of the sweep_cue noise study). This adds the recurrent
# variance closure on top of the input-noise mean field. Validation against a noisy simulation
# (scratchpad/validate_sc.py) showed the RIGHT STRUCTURE (rule-mode variance matches; gain
# compression + criticality amplification present) but it OVER-predicts the stiff/slow modes
# (deep-memory κ0 by ~10–20×) because our TWO-TIMESCALE discrete dynamics temporally filter the
# injected noise (fluctuation–dissipation) — a factor this instantaneous-variance closure omits.
# PRODUCTION noise field = the input-only EXACT Gaussian resummation in low_rank_field_np(noise_sigma=…)
# (validated to ~2e-3 vs Monte-Carlo E_ξ[Ψ]). Use these SC functions for qualitative exploration only;
# to make them quantitative, fold the α/α_rec low-pass stationary-variance factor into s².
# ---------------------------------------------------------------------------


def _phi_avgs(params, a_bar, Delta):
    """Gaussian averages ⟨φ(ā+√Δ z)⟩ and ⟨φ'(ā+√Δ z)⟩ (z~N(0,1)), per element of a_bar/Delta.
    Gaussian-CDF φ (lif/erf/lif_sc, noise_compress=c): EXACT closed form φ̄=φ(ā/√(1+cΔ)),
    φ̄'=φ'(ā/√(1+cΔ))/√(1+cΔ). Otherwise (tanh, …): probabilists' Gauss-Hermite quadrature."""
    phi   = params.get("phi", np.tanh)
    phip  = params.get("phi_prime", lambda u: 1.0 - np.tanh(u) ** 2)
    c     = params.get("noise_compress")
    Delta = np.maximum(np.asarray(Delta, float), 0.0)
    if c is not None:
        comp = np.sqrt(1.0 + c * Delta)
        ae   = a_bar / comp
        return phi(ae), phip(ae) / comp
    nodes, wts = np.polynomial.hermite_e.hermegauss(15)     # ∫f(x)e^{-x²/2}dx = Σ w f(x)
    wts = wts / np.sqrt(2.0 * np.pi)
    sd  = np.sqrt(Delta)
    pb  = np.zeros_like(a_bar); ppb = np.zeros_like(a_bar)
    for zk, wk in zip(nodes, wts):
        arg = a_bar + sd * zk
        pb  = pb + wk * phi(arg); ppb = ppb + wk * phip(arg)
    return pb, ppb


def solve_sc_variance(params, a_bar, noise_sigma, n_iter=80, tol=1e-8, damping=0.5, crit_eps=0.02):
    """Self-consistent per-neuron current variance Δᵢ(κ) for input noise σ (see closure in the field
    docstring). a_bar: (B,N) mean drive per κ point. Returns (Delta (B,N), sigma_tilde (B,R,R),
    phip_bar (B,N)). Near/above criticality (I−σ̃ singular) the recurrent amplification is dropped
    (G→I) so the field stays finite in unstable regions; Δ is floored at the input-only value."""
    M, Nvec = params["M"], params["Nvec"]
    Wi, gain = params["Wi"], params["gain"]
    N, R = M.shape
    Ai = np.asarray(params["Ai"], dtype=np.float64)
    if Ai.ndim == 0:
        Ai = np.full(N, float(Ai))
    sig2   = float(noise_sigma) ** 2
    s2     = (gain ** 2) * (Ai ** 2) * sig2 * np.sum(Wi ** 2, axis=1)     # (N,) input-direct variance
    B      = a_bar.shape[0]
    Delta  = np.tile(s2, (B, 1))
    I_R    = np.eye(R)
    sigt = phip_bar = None
    for _ in range(n_iter):
        phi_bar, phip_bar = _phi_avgs(params, a_bar, Delta)               # (B,N)
        sigt = (gain / N) * np.einsum('jr,bj,js->brs', Nvec, phip_bar, M)          # (B,R,R)
        Ut   = (1.0 / N) * np.einsum('jr,bj,j,jk->brk', Nvec, phip_bar, Ai, Wi)    # (B,R,n_in)
        eig    = np.linalg.eigvals(sigt).real                             # (B,R)
        stable = eig.max(axis=1) < (1.0 - crit_eps)
        G = np.tile(I_R, (B, 1, 1))
        if stable.any():
            G[stable] = np.linalg.inv(I_R[None] - sigt[stable])
        Cov_b = (gain ** 2 * sig2) * np.einsum('brk,bsk->brs', Ut, Ut)    # g²σ² Ũ Ũᵀ
        C     = np.einsum('brs,bst,but->brt', G, Cov_b, G)                # G Cov_b Gᵀ (B,R,R)
        P     = np.einsum('brs,bsk->brk', G, Ut)                          # (B,R,n_in)
        cross = (gain ** 3 * sig2) * Ai[None, :] * np.einsum('ir,brk,ik->bi', M, P, Wi)
        rec   = (gain ** 2) * np.einsum('brs,ir,is->bi', C, M, M)
        Delta_new = np.maximum(s2[None, :] + 2.0 * cross + rec, s2[None, :])
        step  = float(np.max(np.abs(Delta_new - Delta)))
        Delta = (1.0 - damping) * Delta + damping * Delta_new
        if step < tol:
            break
    return Delta, sigt, phip_bar


def _sc_a_bar(params, kappa_flat, ff_input):
    """Mean drive ā = g(Ai(Wi x̄+b) + Mκ) + unit_bias, (B,N)."""
    M, Wi, bi = params["M"], params["Wi"], params["bi"]
    Ai, gain = params["Ai"], params["gain"]
    ub = params.get("unit_bias", 0.0)
    ff = (np.zeros(Wi.shape[1]) if ff_input is None
          else np.atleast_2d(np.asarray(ff_input, float)).mean(axis=0))   # mean input condition
    input_drive = Ai * (ff @ Wi.T + bi)                                   # (N,)
    return gain * (input_drive[None, :] + kappa_flat @ M.T) + ub          # (B,N)


def low_rank_field_sc_np(params, kappa, ff_input=None, noise_sigma=0.0, return_delta=False, **kw):
    """Reduced field F(κ)=Ψ_σ(κ)−κ with the SELF-CONSISTENT (DMFT) input-noise mean field:
    Ψ_σ,r(κ)=(1/N)Σ_j n_jr ⟨φ(ā_j+√Δ_j z)⟩ with Δ_j solved self-consistently (input-direct +
    recurrent variance amplified through (I−σ̃)⁻¹). noise_sigma=0 → deterministic field."""
    M = params["M"]; Nvec = params["Nvec"]
    kappa = np.asarray(kappa, float); orig = kappa.shape; R = M.shape[1]
    kf = kappa.reshape(-1, R)
    a_bar = _sc_a_bar(params, kf, ff_input)
    if noise_sigma and noise_sigma > 0.0:
        Delta, _, _ = solve_sc_variance(params, a_bar, noise_sigma, **kw)
        phi_bar, _ = _phi_avgs(params, a_bar, Delta)
    else:
        Delta = np.zeros_like(a_bar)
        phi_bar = params.get("phi", np.tanh)(a_bar)
    F = (phi_bar @ Nvec / M.shape[0] - kf).reshape(orig)
    return (F, Delta.reshape(orig[:-1] + (a_bar.shape[-1],))) if return_delta else F


def low_rank_jacobian_sc_np(params, kappa, ff_input=None, noise_sigma=0.0, **kw):
    """Jacobian of the self-consistent field: J = σ̃(κ) − I, σ̃ evaluated at the converged Δ.
    (Neglects ∂Δ/∂κ — the standard slowly-varying-variance approximation.)"""
    M = params["M"]; R = M.shape[1]
    kf = np.asarray(kappa, float).reshape(-1, R)[:1]
    a_bar = _sc_a_bar(params, kf, ff_input)
    if noise_sigma and noise_sigma > 0.0:
        _, sigt, _ = solve_sc_variance(params, a_bar, noise_sigma, **kw)
        return sigt[0] - np.eye(R)
    return low_rank_jacobian_flow_np(params, kf[0], ff_input=ff_input)


def trace_slow_manifold(model, ff_input, xlim, ylim, n_angles=180,
                        r_min=0.2, vel_thresh=0.12, center=(0.0, 0.0)):
    """Trace the low-velocity ridge (slow manifold / remnant ring) of the κ field.

    For each angle around `center`, find the radius minimising |F(κ)|; keep the
    point if |F| <= vel_thresh. Returns (pts [K,2], vmag [K], v_tang [K]) where
    v_tang is the tangential (along-ring) speed — the slow drift. Empty if the
    field has no slow ridge below the threshold within the axes.
    """
    from scipy.optimize import minimize_scalar
    params   = low_rank_numpy_params(model)
    ff_np    = torch.as_tensor(ff_input).detach().cpu().numpy().astype(np.float64)
    c        = np.asarray(center, dtype=np.float64)
    r_max    = float(max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1])))

    def Fmag(k):
        return float(np.linalg.norm(low_rank_field_np(params, k, ff_input=ff_np).reshape(-1)))

    pts, vmag, vtang = [], [], []
    for th in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        u   = np.array([np.cos(th), np.sin(th)])
        res = minimize_scalar(lambda r: Fmag(c + r * u), bounds=(r_min, r_max), method="bounded")
        p   = c + res.x * u
        if not (xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]):
            continue
        v = low_rank_field_np(params, p, ff_input=ff_np).reshape(-1)
        if np.linalg.norm(v) <= vel_thresh:
            tang = np.array([-np.sin(th), np.cos(th)])
            pts.append(p); vmag.append(float(np.linalg.norm(v))); vtang.append(abs(float(v @ tang)))
    if not pts:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,))
    return np.asarray(pts), np.asarray(vmag), np.asarray(vtang)


# ---------------------------------------------------------------------------
# Projection h -> κ
# ---------------------------------------------------------------------------


@torch.no_grad()
def project_rec_inputs_to_kappa(model, rec_inputs):
    """Least-squares coordinates for h ≈ Mκ.  rec_inputs: [..., N]"""
    device      = _model_device(model)
    dtype       = _model_dtype(model)
    h           = torch.as_tensor(rec_inputs, device=device, dtype=dtype)
    orig_shape  = h.shape
    hidden_size = model.m.shape[0]
    rank        = model.m.shape[1]

    if orig_shape[-1] != hidden_size:
        raise ValueError(f"Expected last dim {hidden_size}, got {orig_shape[-1]}.")

    h_flat = h.reshape(-1, hidden_size)
    sol    = torch.linalg.lstsq(model.m, h_flat.T).solution  # [rank, B]
    kappa  = sol.T.reshape(*orig_shape[:-1], rank)
    return kappa


# ---------------------------------------------------------------------------
# Run model while recording effective inputs
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_low_rank_with_effective_inputs(
    model,
    ff_inputs,
    targets=None,
    use_reward_feedback=None,
    reward_readout_index=None,
):
    """
    Run a LowRankModel while recording the actual input x_t used at each step.

    Returns: readouts, rates, rec_inputs, effective_x  (all [B, T, *])
    """
    device = _model_device(model)
    dtype  = _model_dtype(model)

    ff_inputs = ff_inputs.to(device=device, dtype=dtype)
    if targets is not None:
        targets = targets.to(device=device)

    batch_size, T, input_size = ff_inputs.shape
    hidden_size = model.m.shape[0]

    rec_inputs = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
    rates      = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
    noise      = getattr(model, "noise", 0.0)

    rwd_channel = getattr(model, "rwd_channel", None)
    if rwd_channel is None:
        rwd_channel = getattr(model, "reward_channel", None)

    if use_reward_feedback is None:
        use_reward_feedback = (
            targets is not None
            and rwd_channel is not None
            and (bool(getattr(model, "rwd", False)) or getattr(model, "reward_channel", None) is not None)
        )

    readout_list, rates_list, rec_list, eff_input_list = [], [], [], []
    rwd_next = torch.zeros(batch_size, device=device, dtype=dtype)

    for step in range(T):
        x_t = ff_inputs[:, step].clone()

        if use_reward_feedback:
            x_t[:, rwd_channel] = x_t[:, rwd_channel] + rwd_next

        sigma  = noise * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        rates, rec_inputs = model.update_dynamics(x_t, rec_inputs + sigma, rates)
        readout = model.get_readout(rates, rec_inputs)

        readout_list.append(readout)
        rates_list.append(rates)
        rec_list.append(rec_inputs)
        eff_input_list.append(x_t)

        if use_reward_feedback:
            idx = (
                rwd_channel if rwd_channel < readout.shape[-1] else -1
            ) if reward_readout_index is None else reward_readout_index
            rwd_mask = (targets[:, step, rwd_channel] == 1) & (readout[:, idx] > 0.5)
            rwd_next = rwd_mask.to(dtype=dtype)
        else:
            rwd_next.zero_()

    return (
        torch.stack(readout_list,   dim=1),
        torch.stack(rates_list,     dim=1),
        torch.stack(rec_list,       dim=1),
        torch.stack(eff_input_list, dim=1),
    )


# ---------------------------------------------------------------------------
# Plotting support
# ---------------------------------------------------------------------------


def auto_limits_from_trajs(kappa_traj, qlo=1, qhi=99, pad=1.25, symmetric=True, min_lim=1.5):
    z  = kappa_traj.reshape(-1, kappa_traj.shape[-1])
    lo = np.percentile(z[:, :2], qlo, axis=0)
    hi = np.percentile(z[:, :2], qhi, axis=0)

    if symmetric:
        lim = float(pad * np.max(np.abs(np.concatenate([lo, hi]))))
        lim = max(lim, float(min_lim))
        return (-lim, lim), (-lim, lim)

    center = 0.5 * (lo + hi)
    half   = 0.5 * (hi - lo) * pad
    half   = np.maximum(half, float(min_lim))
    return (
        (float(center[0] - half[0]), float(center[0] + half[0])),
        (float(center[1] - half[1]), float(center[1] + half[1])),
    )


def _canonical_flow_panels(cue_on_go_input=False, cue_scale=1.0):
    cue_dims = [4] if cue_on_go_input else [6]
    cue_val  = cue_scale if cue_on_go_input else 1.0
    return [
        dict(name="Autonomous", dims=None,     value=1.0),
        dict(name="A",          dims=[0],      value=1.0),
        dict(name="B",          dims=[1],      value=1.0),
        dict(name="Go",         dims=[4],      value=1.0),
        dict(name="NoGo",       dims=[5],      value=1.0),
        dict(name="Cue",        dims=cue_dims, value=cue_val),
        dict(name="C",          dims=[2],      value=1.0),
        dict(name="D",          dims=[3],      value=1.0),
    ]


@torch.no_grad()
def sim_kappa_field(model, ff_input, xlim, ylim, n_grid=61, n_warmup=0, device=None):
    """
    Simulation-based κ-plane flow field.

    Initializes G = n_grid² states from a (κ₀, κ₁) grid using the adiabatic
    approximation (rates = tanh(gain·(Wi·x + M·κ)), h = M·κ), runs n_warmup
    steps of the actual model (including W_fixed), then records one additional
    Δκ step.

    n_warmup=0 (default): field is measured at the exact grid κ values →
        K0, K1 form a regular grid, compatible with matplotlib streamplot.
    n_warmup>0: states are allowed to drift before measurement →
        K0, K1 are scattered; use quiver for visualization.

    Returns
    -------
    K0, K1 : ndarray (n_grid, n_grid)  κ₀ / κ₁ origin of each arrow.
    U, V   : ndarray (n_grid, n_grid)  Δκ₀ / Δκ₁.
    speed  : ndarray (n_grid, n_grid)  ‖Δκ‖.
    """
    if device is None:
        device = _model_device(model)
    dtype = _model_dtype(model)
    model.eval()

    ff       = torch.as_tensor(ff_input, device=device, dtype=dtype)
    N, m, n  = model.hidden_size, model.m.detach(), model.n.detach()

    k0v = torch.linspace(xlim[0], xlim[1], n_grid, device=device, dtype=dtype)
    k1v = torch.linspace(ylim[0], ylim[1], n_grid, device=device, dtype=dtype)
    K0g, K1g = torch.meshgrid(k0v, k1v, indexing="xy")           # (n_grid, n_grid)
    kappa_grid = torch.stack([K0g.reshape(-1), K1g.reshape(-1)], dim=-1)  # (G, 2)
    G          = kappa_grid.shape[0]

    ff_batch = ff.unsqueeze(0).expand(G, -1)                      # (G, input_size)

    # Adiabatic init: h = M·κ,  rates = tanh(gain·(Wi·x + h))
    h = kappa_grid @ m.T                                          # (G, N)
    if model.wi is not None:
        input_drive = model.Ai * model.wi(ff_batch)              # (G, N)
    else:
        input_drive = torch.zeros(G, N, device=device, dtype=dtype)
    rates = model.nonlinearity(model.gain * (input_drive + h))

    for _ in range(n_warmup):
        rates, h = model.update_dynamics(ff_batch, h, rates)

    kappa_base    = rates @ n / N                                  # (G, 2)
    rates_next, _ = model.update_dynamics(ff_batch, h, rates)
    kappa_next    = rates_next @ n / N                             # (G, 2)

    if n_warmup == 0:
        # Keep arrows anchored at exact grid points (streamplot-compatible).
        # Δκ = one simulation step from the analytically-initialized state,
        # reported relative to the grid origin.
        origin      = kappa_grid
        delta_kappa = kappa_next - origin
    else:
        # Warmup let the state drift; report Δκ from wherever it settled.
        origin      = kappa_base
        delta_kappa = kappa_next - kappa_base

    K0    = origin[:, 0].reshape(n_grid, n_grid).cpu().numpy()
    K1    = origin[:, 1].reshape(n_grid, n_grid).cpu().numpy()
    U     = delta_kappa[:, 0].reshape(n_grid, n_grid).cpu().numpy()
    V     = delta_kappa[:, 1].reshape(n_grid, n_grid).cpu().numpy()
    speed = np.sqrt(U**2 + V**2)
    return K0, K1, U, V, speed


@torch.no_grad()
def _sim_step_single(model, ff_input, kappa_np, n_warmup=0):
    """One-step sim map for a single κ point. Returns Δκ as a float64 array."""
    device  = _model_device(model)
    dtype   = _model_dtype(model)
    ff      = torch.as_tensor(ff_input, device=device, dtype=dtype).unsqueeze(0)
    N, m, n = model.hidden_size, model.m.detach(), model.n.detach()

    kappa = torch.as_tensor(kappa_np, device=device, dtype=dtype).unsqueeze(0)
    h     = kappa @ m.T
    if model.wi is not None:
        input_drive = model.Ai * model.wi(ff)
    else:
        input_drive = torch.zeros(1, N, device=device, dtype=dtype)
    rates = model.nonlinearity(model.gain * (input_drive + h))

    for _ in range(n_warmup):
        rates, h = model.update_dynamics(ff, h, rates)

    kappa_base    = rates @ n / N
    rates_next, _ = model.update_dynamics(ff, h, rates)
    return (rates_next @ n / N - kappa_base).cpu().numpy()[0].astype(np.float64)


@torch.no_grad()
def integrate_kappa_trajectories(model, ff_input, kappa0, n_steps=500, noise_sigma=0.0,
                                 noise_seed=0, record_every=1):
    """GENUINE simulated κ-trajectories (rank-general). Seed the full network on the low-rank manifold
    (h = κ₀ mᵀ, rates = φ(gain·(A_i·W_i·x + h))) and integrate the TRUE two-timescale dynamics for
    n_steps with the input CLAMPED at ff_input. noise_sigma>0 injects per-step input noise
    (x_t = ff + σ·N(0,1)) → genuine NOISY trajectories. Returns κ(t): (B, n_rec+1, rank) float64
    (n_rec = n_steps//record_every). This is the real flow — NOT the adiabatic one-step map."""
    device = _model_device(model); dtype = _model_dtype(model)
    N, m, n = model.hidden_size, model.m.detach(), model.n.detach()
    ff = torch.as_tensor(ff_input, device=device, dtype=dtype).reshape(-1)
    kap = torch.as_tensor(kappa0, device=device, dtype=dtype)
    if kap.ndim == 1:
        kap = kap.unsqueeze(0)
    B = kap.shape[0]
    ffb = ff[None, :].expand(B, -1).contiguous()
    h = kap @ m.T
    drive = model.Ai * model.wi(ffb) if model.wi is not None else torch.zeros(B, N, device=device, dtype=dtype)
    rates = model.nonlinearity(model.gain * (drive + h))
    gen = None
    if noise_sigma and noise_sigma > 0.0:
        gen = torch.Generator(device=device).manual_seed(int(noise_seed))
    traj = [(rates @ n / N).cpu().numpy()]
    for t in range(n_steps):
        ff_t = ffb
        if gen is not None:
            ff_t = ffb + noise_sigma * torch.randn(B, ff.shape[0], generator=gen, device=device, dtype=dtype)
        rates, h = model.update_dynamics(ff_t, h, rates)
        if (t + 1) % record_every == 0:
            traj.append((rates @ n / N).cpu().numpy())
    return np.stack(traj, axis=1).astype(np.float64)     # (B, n_rec+1, rank)

