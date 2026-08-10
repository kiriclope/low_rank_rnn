"""Fixed-point finding — rank-general, two backends (scipy root / brainpy SlowPointFinder)."""
from __future__ import annotations


import numpy as np
import torch
import scipy.special
from scipy.optimize import root

from .flow_field import (
    NOISE_COMPRESS, _model_device, _model_dtype, make_input, low_rank_numpy_params, low_rank_field_np, low_rank_jacobian_flow_np, low_rank_jacobian_map_np, _phi_avgs, solve_sc_variance, _sc_a_bar, low_rank_field_sc_np, low_rank_jacobian_sc_np, trace_slow_manifold, project_rec_inputs_to_kappa, run_low_rank_with_effective_inputs, auto_limits_from_trajs, _canonical_flow_panels, sim_kappa_field, _sim_step_single, integrate_kappa_trajectories,
)


def merge_roots(roots, residuals, merge_tol=5e-2):
    if len(roots) == 0:
        return np.empty((0, 2)), np.empty((0,))

    roots     = np.asarray(roots, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    order     = np.argsort(residuals)
    kept, kept_res = [], []

    for idx in order:
        r = roots[idx]
        if all(np.linalg.norm(r - q) > merge_tol for q in kept):
            kept.append(r)
            kept_res.append(residuals[idx])

    return np.stack(kept, axis=0), np.asarray(kept_res)


def find_all_fixed_points(model, xlim, ylim, ff_input, n_seeds=41,
                           residual_tol=1e-8, merge_tol=5e-2, noise_sigma=0.0):
    if model.m.shape[1] != 2:
        raise ValueError("Fixed-point finding assumes rank == 2.")

    params      = low_rank_numpy_params(model)
    ff_input_np = torch.as_tensor(ff_input).detach().cpu().numpy().astype(np.float64)

    def fun(k):
        return low_rank_field_np(params, k, ff_input=ff_input_np, noise_sigma=noise_sigma).reshape(-1)

    def jac(k):
        return low_rank_jacobian_flow_np(params, k, ff_input=ff_input_np, noise_sigma=noise_sigma)

    roots, residuals = [], []

    for k1_0 in np.linspace(xlim[0], xlim[1], n_seeds):
        for k2_0 in np.linspace(ylim[0], ylim[1], n_seeds):
            sol = root(fun, np.array([k1_0, k2_0], dtype=np.float64),
                       jac=jac, method="hybr", options={"xtol": 1e-12, "maxfev": 1000})
            if not sol.success:
                continue
            fp       = sol.x
            residual = np.linalg.norm(fun(fp))
            in_bounds = (xlim[0] <= fp[0] <= xlim[1] and ylim[0] <= fp[1] <= ylim[1])
            if residual <= residual_tol and in_bounds:
                roots.append(fp)
                residuals.append(residual)

    return merge_roots(roots, residuals, merge_tol=merge_tol)


def classify_fixed_points(model, fixed_points, ff_input, eig_tol=1e-5,
                          marginal_tol=2e-3, slow_tol=None, noise_sigma=0.0):
    # marginal_tol default 2e-3 (was 1e-2): the loose band mislabeled genuine but SLOW attractors
    # (map |λ|≈0.99 — shallow subcritical wells, non-saturating φ like relu/softplus) as "marginal",
    # so they were dropped/faint in the flow plots and excluded from well stats. 2e-3 still catches
    # true line/ring manifolds (|λ|=1.000) but keeps slow point attractors as attractor/slow. §18.
    """Classify roots of the κ-plane map by the discrete-map Jacobian spectrum.

    A near-unit eigenvalue (|λ|-1| <= marginal_tol) means the point lies on a
    *marginal* slow manifold — a line/ring attractor remnant — where the
    attractor/saddle distinction is numerically meaningless. Such points are
    labelled "marginal" (if no strictly unstable direction) rather than
    attractor/saddle, so figures don't sprinkle fragile isolated attractor dots
    along a continuous manifold.

    If `slow_tol` is set, *stable* attractors whose slowest eigenvalue lies within
    slow_tol of the unit circle (1 - max|λ| <= slow_tol) are re-labelled
    "slow_attractor" — a genuine but shallow attractor on a soft slow ring,
    distinct from a strictly marginal (|λ|≈1) degeneracy. Off by default.
    """
    params      = low_rank_numpy_params(model)
    ff_input_np = torch.as_tensor(ff_input).detach().cpu().numpy().astype(np.float64)

    labels, eigvals = [], []

    for fp in fixed_points:
        J  = low_rank_jacobian_map_np(params, fp, ff_input=ff_input_np, noise_sigma=noise_sigma)
        ev = np.linalg.eigvals(J)
        rad = np.abs(ev)

        n_marginal = np.sum(np.abs(rad - 1.0) <= marginal_tol)
        n_stable   = np.sum(rad < 1.0 - marginal_tol)
        n_unstable = np.sum(rad > 1.0 + marginal_tol)

        if n_marginal > 0 and n_unstable == 0:
            label = "marginal"            # on a (stable) line/ring attractor
        elif n_stable == len(rad):         label = "attractor"
        elif n_unstable == len(rad):       label = "repeller"
        elif n_stable > 0 or n_unstable > 0: label = "saddle"
        else:                              label = "nonhyperbolic"

        if label == "attractor" and slow_tol is not None and (1.0 - rad.max()) <= slow_tol:
            label = "slow_attractor"

        labels.append(label)
        eigvals.append(ev)

    if len(eigvals) == 0:
        return np.asarray([], dtype=object), np.empty((0, 2), dtype=np.complex128)

    return np.asarray(labels, dtype=object), np.stack(eigvals, axis=0)


# ---------------------------------------------------------------------------
# Effective-input trajectory masks
# ---------------------------------------------------------------------------


def find_sim_fixed_points(model, ff_input, xlim, ylim, n_seeds=21, n_warmup=0,
                           residual_tol=1e-5, merge_tol=5e-2):
    """
    Find fixed points of the simulation-based one-step map via scipy.optimize.root.

    Identical API to find_all_fixed_points but uses the actual model (including
    W_fixed) instead of the analytical κ-plane equation. Numerical Jacobian is
    used (no analytic Jacobian available for the sim path).
    """
    def f(kappa_np):
        return _sim_step_single(model, ff_input, kappa_np, n_warmup=n_warmup)

    roots, residuals = [], []
    for k0 in np.linspace(xlim[0], xlim[1], n_seeds):
        for k1 in np.linspace(ylim[0], ylim[1], n_seeds):
            sol = root(f, np.array([k0, k1], dtype=np.float64),
                       method="hybr", options={"xtol": 1e-12, "maxfev": 2000})
            if not sol.success:
                continue
            fp    = sol.x
            resid = float(np.linalg.norm(f(fp)))
            in_bounds = (xlim[0] <= fp[0] <= xlim[1] and ylim[0] <= fp[1] <= ylim[1])
            if resid <= residual_tol and in_bounds:
                roots.append(fp)
                residuals.append(resid)

    return merge_roots(roots, np.asarray(residuals), merge_tol)


def classify_sim_fixed_points(model, fixed_points, ff_input, n_warmup=0,
                               eig_tol=1e-5, eps=1e-4, marginal_tol=2e-3, slow_tol=None):  # §18: was 1e-2
    """
    Classify fixed points using a numerical Jacobian of the simulation-based map.

    Same return convention as classify_fixed_points: (labels, eigvals).
    """
    labels, eigvals = [], []
    for fp in fixed_points:
        J = np.zeros((2, 2), dtype=np.float64)
        for j in range(2):
            step       = np.zeros(2, dtype=np.float64)
            step[j]    = eps
            fp_plus    = _sim_step_single(model, ff_input, fp + step, n_warmup)
            fp_minus   = _sim_step_single(model, ff_input, fp - step, n_warmup)
            J[:, j]    = (fp_plus - fp_minus) / (2.0 * eps)

        J_map = np.eye(2) + J
        ev    = np.linalg.eigvals(J_map)
        rad   = np.abs(ev)

        n_marginal = np.sum(np.abs(rad - 1.0) <= marginal_tol)
        n_stable   = np.sum(rad < 1.0 - marginal_tol)
        n_unstable = np.sum(rad > 1.0 + marginal_tol)
        if n_marginal > 0 and n_unstable == 0:
            label = "marginal"
        elif n_stable == len(rad):    label = "attractor"
        elif n_unstable == len(rad):  label = "repeller"
        elif n_stable > 0 or n_unstable > 0: label = "saddle"
        else:                         label = "nonhyperbolic"

        if label == "attractor" and slow_tol is not None and (1.0 - rad.max()) <= slow_tol:
            label = "slow_attractor"

        labels.append(label)
        eigvals.append(ev)

    if not eigvals:
        return np.asarray([], dtype=object), np.empty((0, 2), dtype=np.complex128)
    return np.asarray(labels, dtype=object), np.stack(eigvals)


# ---------------------------------------------------------------------------
# Rank-2 torch-native flow field
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared RANK-GENERAL fixed-point finder — two backends: scipy (numpy field) / brainpy (jax field)
# ---------------------------------------------------------------------------

# noise-compression c: imported from flow_field so the jax and numpy fields can never diverge.
_JAX_PHI_NC = NOISE_COMPRESS


def build_jax_field(params, phi_name, ff, noise_sigma=0.0):
    """jax reduced field F(κ)=Ψ(κ)−κ for a single input condition, RANK-GENERAL (k @ Mᵀ). Used by the
    brainpy FP backend and the rank-3 adiabatic slice. noise_sigma>0 applies the exact Gaussian-CDF gain
    compression (φ(a/√(1+c·s²)), c per _JAX_PHI_NC). Non-Gaussian φ: noise ignored (matches numpy field)."""
    import jax, jax.numpy as jnp
    PHI = {"tanh": jnp.tanh, "relu": lambda x: jnp.maximum(x, 0.0),
           "erf": jax.scipy.special.erf, "softplus": jax.nn.softplus, "elu": jax.nn.elu,
           "lif":    lambda x: 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0))),
           "lif_sc": lambda x: 0.5 * (1.0 + jax.scipy.special.erf(x * jnp.sqrt(jnp.pi)))}
    M = jnp.asarray(params["M"]); Nv = jnp.asarray(params["Nvec"]); g = float(params["gain"]); N = params["M"].shape[0]
    drive = jnp.asarray(params["Ai"] * (np.asarray(ff, float) @ params["Wi"].T + params["bi"]))
    phi = PHI.get(phi_name, jnp.tanh); c = _JAX_PHI_NC.get(phi_name)
    denom = None
    if noise_sigma and noise_sigma > 0.0 and c is not None:
        s2 = (g ** 2) * (np.asarray(params["Ai"], float) ** 2) * (float(noise_sigma) ** 2) \
             * np.sum(np.asarray(params["Wi"], float) ** 2, axis=1)
        denom = jnp.asarray(np.sqrt(1.0 + c * s2))

    def field(k):
        a = g * (drive[None, :] + k @ M.T)
        if denom is not None:
            a = a / denom[None, :]
        return phi(a) @ Nv / N - k
    return field


def classify_lowrank_fps(params, fps, ff_input, noise_sigma=0.0, marg=0.04, slow_tol=None):
    """Classify each κ* by the reduced-flow Jacobian eigenvalues (rank-general): attractor / saddle /
    repeller / marginal (+ slow_attractor if slow_tol given and the attractor is near-marginal)."""
    ff = np.asarray(ff_input, dtype=np.float64)
    labs = []
    for f in np.atleast_2d(np.asarray(fps, dtype=np.float64)):
        ev = np.sort(np.linalg.eigvals(
            low_rank_jacobian_flow_np(params, f, ff_input=ff[None, :], noise_sigma=noise_sigma)).real)
        lo, hi = ev[0], ev[-1]
        lab = ("attractor" if hi < -marg else "repeller" if lo > marg else
               "saddle" if (hi > marg and lo < -marg) else "marginal")
        if lab == "attractor" and slow_tol is not None and (-hi) <= slow_tol:
            lab = "slow_attractor"
        labs.append(lab)
    return np.array(labs)


def find_fixed_points(params, ff_input, phi_name="tanh", rank=None, backend="scipy",
                      box=2.5, n_seeds=7, noise_sigma=0.0, slow_tol=1e-7, marg=0.04, merge=8e-2,
                      residual_tol=1e-6, classify_slow_tol=None):
    """RANK-GENERAL fixed points of the reduced field F(κ)=Ψ(κ)−κ, with two interchangeable backends:
      backend='scipy'   — scipy.root on the numpy field `low_rank_field_np` from an n_seedsᵏ κ-grid.
      backend='brainpy' — brainpy SlowPointFinder (jax GD on ½‖F‖²) on `build_jax_field`, same grid.
    Both classify via the analytic reduced-flow Jacobian. `noise_sigma`>0 uses the input-noise mean field.
    Returns (fps (n,rank), labels (n,))."""
    rank = int(rank or params["M"].shape[1])
    ff = np.asarray(ff_input, dtype=np.float64)
    if backend == "brainpy":
        import brainpy as bp, brainpy.math as bm
        field = build_jax_field(params, phi_name, ff, noise_sigma)
        gx = np.linspace(-box, box, n_seeds)
        cand = np.array(np.meshgrid(*([gx] * rank))).reshape(rank, -1).T.astype(np.float32)
        fdr = bp.analysis.SlowPointFinder(f_cell=field, f_type="continuous")
        fdr.find_fps_with_gd_method(candidates=bm.asarray(cand), tolerance=1e-9, num_batch=400,
            num_opt=1500, optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.05, 1, 0.9999)))
        fdr.filter_loss(slow_tol); fdr.keep_unique(tolerance=merge)
        fps = np.asarray(fdr.fixed_points).reshape(-1, rank)
    else:
        fun = lambda k: low_rank_field_np(params, k.reshape(1, -1), ff_input=ff[None, :],
                                          noise_sigma=noise_sigma).reshape(-1)
        jac = lambda k: low_rank_jacobian_flow_np(params, k.reshape(1, -1), ff_input=ff[None, :],
                                                  noise_sigma=noise_sigma)
        axes = [np.linspace(-box, box, n_seeds)] * rank
        seeds = np.stack([g.ravel() for g in np.meshgrid(*axes)], axis=1)
        fps = []
        for s in seeds:
            sol = root(fun, s, jac=jac, tol=1e-11)
            if sol.success and np.max(np.abs(fun(sol.x))) < residual_tol:
                if not any(np.linalg.norm(sol.x - q) < merge for q in fps):
                    fps.append(sol.x)
        fps = np.array(fps) if fps else np.zeros((0, rank))
    labels = (classify_lowrank_fps(params, fps, ff, noise_sigma=noise_sigma, marg=marg,
                                   slow_tol=classify_slow_tol) if len(fps) else np.array([]))
    return fps, labels
