from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.special
from scipy.optimize import root

from .tasks import TaskTiming


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

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

    nl_str = getattr(model, "nonlinearity_str", "tanh")
    if nl_str == "relu":
        phi_np       = lambda u: np.maximum(u, 0.0)
        phi_prime_np = lambda u: (u > 0).astype(np.float64)
    elif nl_str == "softplus":
        # numerically stable: softplus(u) = u for u>>0, log1p(exp(u)) otherwise
        phi_np       = lambda u: np.where(u > 20.0, u, np.log1p(np.exp(np.minimum(u, 20.0))))
        phi_prime_np = lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -20.0, 20.0)))
    elif nl_str == "erf":
        _2_sqrt_pi   = 2.0 / np.sqrt(np.pi)
        phi_np       = scipy.special.erf
        phi_prime_np = lambda u: _2_sqrt_pi * np.exp(-np.clip(u, -20.0, 20.0) ** 2)
    elif nl_str == "elu":
        # ELU: x for x>0, exp(x)-1 for x<=0 (alpha=1); φ'= 1 for x>0, exp(x) for x<=0
        phi_np       = lambda u: np.where(u > 0, u, np.exp(np.minimum(u, 0.0)) - 1.0)
        phi_prime_np = lambda u: np.where(u > 0, 1.0, np.exp(np.minimum(u, 0.0)))
    elif nl_str == "lif":
        # Brunel erfc approximation: Gaussian CDF, φ ∈ [0,1], φ'(0) = 1/sqrt(2π) ≈ 0.399
        _sqrt2    = np.sqrt(2.0)
        _sqrt2pi  = np.sqrt(2.0 * np.pi)
        phi_np       = lambda u: 0.5 * (1.0 + scipy.special.erf(u / _sqrt2))
        phi_prime_np = lambda u: np.exp(-np.clip(u, -20.0, 20.0) ** 2 / 2.0) / _sqrt2pi
    else:  # tanh (default)
        phi_np       = np.tanh
        phi_prime_np = lambda u: 1.0 - np.tanh(u) ** 2

    return {
        "M":         model.m.detach().cpu().numpy().astype(np.float64),
        "Nvec":      n_eff.detach().cpu().numpy().astype(np.float64),
        "Wi":        model.wi.weight.detach().cpu().numpy().astype(np.float64),
        "bi":        model.wi.bias.detach().cpu().numpy().astype(np.float64),
        "Ai":        Ai,
        "gain":      gain,
        "beta":      1.0 - np.exp(-alpha),
        "phi":       phi_np,
        "phi_prime": phi_prime_np,
    }


# ---------------------------------------------------------------------------
# Low-rank vector field and Jacobians
# ---------------------------------------------------------------------------

def low_rank_field_np(params, kappa, ff_input=None, include_beta=False):
    """
    Vector field in κ-coordinates:  F(κ; x) = Ψ(κ; x) - κ

    If include_beta=True, returns βF, matching the discrete update increment.
    """
    M, Nvec = params["M"], params["Nvec"]
    Wi, bi  = params["Wi"], params["bi"]
    Ai, gain= params["Ai"], params["gain"]

    kappa      = np.asarray(kappa, dtype=np.float64)
    orig_shape = kappa.shape
    rank       = M.shape[1]
    kappa_flat = kappa.reshape(-1, rank)

    ff_input = (
        np.zeros(Wi.shape[1], dtype=np.float64) if ff_input is None
        else np.asarray(ff_input, dtype=np.float64)
    )

    phi = params.get("phi", np.tanh)
    input_drive = Ai * (ff_input @ Wi.T + bi)
    h   = kappa_flat @ M.T
    r   = phi(gain * (input_drive[None, :] + h))
    psi = r @ Nvec / M.shape[0]

    field = psi - kappa_flat
    if include_beta:
        field = params["beta"] * field
    return field.reshape(orig_shape)


def low_rank_jacobian_flow_np(params, kappa, ff_input=None):
    """Jacobian of F(κ; x) = Ψ(κ; x) - κ."""
    M, Nvec = params["M"], params["Nvec"]
    Wi, bi  = params["Wi"], params["bi"]
    Ai, gain= params["Ai"], params["gain"]

    kappa    = np.asarray(kappa, dtype=np.float64).reshape(-1)
    ff_input = (
        np.zeros(Wi.shape[1], dtype=np.float64) if ff_input is None
        else np.asarray(ff_input, dtype=np.float64)
    )

    phi_prime_fn = params.get("phi_prime", lambda u: 1.0 - np.tanh(u) ** 2)
    input_drive = Ai * (ff_input @ Wi.T + bi)
    u           = gain * (input_drive + M @ kappa)
    phi_prime   = phi_prime_fn(u)

    J  = Nvec.T @ (phi_prime[:, None] * (gain * M)) / M.shape[0]
    J -= np.eye(M.shape[1])
    return J


def low_rank_jacobian_map_np(params, kappa, ff_input=None):
    """Jacobian of the discrete map κ⁺ = κ + βF(κ)."""
    J_flow = low_rank_jacobian_flow_np(params, kappa, ff_input=ff_input)
    return np.eye(J_flow.shape[0]) + params["beta"] * J_flow


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

def auto_limits_from_trajs(kappa_traj, qlo=1, qhi=99, pad=1.25, symmetric=True):
    z  = kappa_traj.reshape(-1, kappa_traj.shape[-1])
    lo = np.percentile(z[:, :2], qlo, axis=0)
    hi = np.percentile(z[:, :2], qhi, axis=0)

    if symmetric:
        lim = float(pad * np.max(np.abs(np.concatenate([lo, hi]))))
        lim = max(lim, 1e-6)
        return (-lim, lim), (-lim, lim)

    center = 0.5 * (lo + hi)
    half   = 0.5 * (hi - lo) * pad
    return (
        (float(center[0] - half[0]), float(center[0] + half[0])),
        (float(center[1] - half[1]), float(center[1] + half[1])),
    )


def make_vector_field_grid(model, ff_input, xlim, ylim, n_grid=151, include_beta=False):
    params       = low_rank_numpy_params(model)
    ff_input_np  = torch.as_tensor(ff_input).detach().cpu().numpy().astype(np.float64)

    k1 = np.linspace(xlim[0], xlim[1], n_grid)
    k2 = np.linspace(ylim[0], ylim[1], n_grid)
    K1, K2 = np.meshgrid(k1, k2)
    K      = np.stack([K1, K2], axis=-1)

    field  = low_rank_field_np(params, K, ff_input=ff_input_np, include_beta=include_beta)
    U, V   = field[..., 0], field[..., 1]
    speed  = np.sqrt(U**2 + V**2)

    return K1, K2, U, V, speed, params, ff_input_np


# ---------------------------------------------------------------------------
# Fixed points
# ---------------------------------------------------------------------------

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
                           residual_tol=1e-8, merge_tol=5e-2):
    if model.m.shape[1] != 2:
        raise ValueError("Fixed-point finding assumes rank == 2.")

    params      = low_rank_numpy_params(model)
    ff_input_np = torch.as_tensor(ff_input).detach().cpu().numpy().astype(np.float64)

    def fun(k):
        return low_rank_field_np(params, k, ff_input=ff_input_np).reshape(-1)

    def jac(k):
        return low_rank_jacobian_flow_np(params, k, ff_input=ff_input_np)

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


def classify_fixed_points(model, fixed_points, ff_input, eig_tol=1e-5):
    params      = low_rank_numpy_params(model)
    ff_input_np = torch.as_tensor(ff_input).detach().cpu().numpy().astype(np.float64)

    labels, eigvals = [], []

    for fp in fixed_points:
        J  = low_rank_jacobian_map_np(params, fp, ff_input=ff_input_np)
        ev = np.linalg.eigvals(J)
        rad = np.abs(ev)

        n_stable   = np.sum(rad < 1.0 - eig_tol)
        n_unstable = np.sum(rad > 1.0 + eig_tol)

        if n_stable == len(rad):       label = "attractor"
        elif n_unstable == len(rad):   label = "repeller"
        elif n_stable > 0:             label = "saddle"
        else:                          label = "nonhyperbolic"

        labels.append(label)
        eigvals.append(ev)

    if len(eigvals) == 0:
        return np.asarray([], dtype=object), np.empty((0, 2), dtype=np.complex128)

    return np.asarray(labels, dtype=object), np.stack(eigvals, axis=0)


# ---------------------------------------------------------------------------
# Effective-input trajectory masks
# ---------------------------------------------------------------------------

def _canonical_input_mask(effective_x, dims, threshold=0.35, atol_inactive=0.35):
    x = effective_x.detach().cpu()

    if dims is None:
        return x.abs().amax(dim=-1) < atol_inactive

    dims = list(dims)
    active = torch.ones(x.shape[:2], dtype=torch.bool)
    for d in dims:
        active &= x[..., d] > threshold

    inactive_dims = [d for d in range(x.shape[-1]) if d not in dims]
    inactive = x[..., inactive_dims].abs().amax(dim=-1) < atol_inactive
    return active & inactive


def _segments_from_mask(mask_1d, min_len=2):
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    chunks = np.split(idx, breaks)
    return [slice(int(c[0]), int(c[-1]) + 1) for c in chunks if len(c) >= min_len]


# ---------------------------------------------------------------------------
# Time arrows
# ---------------------------------------------------------------------------

def default_arrow_times(T, n_arrows=4, arrow_step=3):
    if T <= arrow_step + 1:
        return np.array([], dtype=int)
    lo, hi = 1, T - arrow_step - 1
    n_arrows = min(n_arrows, max(1, hi - lo + 1))
    return np.unique(np.linspace(lo, hi, n_arrows).astype(int))


def add_time_arrows(ax, xy, times=None, color="k", arrow_step=3,
                    mutation_scale=13, lw=1.5, alpha=0.95, zorder=8, min_length=1e-4):
    xy = np.asarray(xy, dtype=np.float64)
    T  = xy.shape[0]

    if times is None:
        times = default_arrow_times(T, n_arrows=4, arrow_step=arrow_step)

    for t in times:
        t  = int(t)
        if t < 0: t += T
        if t < 0 or t >= T - 1: continue

        j0 = t
        j1 = min(t + arrow_step, T - 1)
        while j1 < T and np.linalg.norm(xy[j1] - xy[j0]) < min_length:
            j0 += 1
            j1  = min(j0 + arrow_step, T - 1)
            if j0 >= T - 1: break

        if j0 >= T - 1: continue
        p0, p1 = xy[j0], xy[j1]
        if np.linalg.norm(p1 - p0) < min_length: continue

        ax.annotate(
            "",
            xy=p1, xytext=p0,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                shrinkA=0, shrinkB=0, mutation_scale=mutation_scale,
            ),
            zorder=zorder,
        )


# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------

def _parse_dual_name(name):
    parts = str(name).split("_")
    if len(parts) == 2:
        sample, test = parts
        gng = "none"
    elif len(parts) == 3:
        sample, gng, test = parts
    else:
        raise ValueError(f"Unexpected dual condition name: {name}")
    return {"sample": sample, "gng": gng, "test": test, "pair": f"{sample}{test}"}


def condition_indices_for_task(inputs, timing, task, condition_names=None, dual_mode="conditions"):
    task = task.lower()
    X    = inputs.detach().cpu()
    n_on = timing.n_stim_on
    n_off= timing.n_stim_off

    if task == "dpa":
        sample_epoch = slice(int(n_on[0]),  int(n_off[0]))
        test_epoch   = slice(int(n_on[1]),  int(n_off[1]))
        sample_drive = X[:, sample_epoch, 0:2].mean(dim=1)
        test_drive   = X[:, test_epoch,   2:4].mean(dim=1)
        is_A = sample_drive[:, 0] > sample_drive[:, 1]
        is_C = test_drive[:,   0] > test_drive[:,   1]
        cond_idx = {
            "AC": torch.where( is_A &  is_C)[0], "AD": torch.where( is_A & ~is_C)[0],
            "BC": torch.where(~is_A &  is_C)[0], "BD": torch.where(~is_A & ~is_C)[0],
        }
        meta = {
            "AC": dict(sample="A", test="C", pair="AC"), "AD": dict(sample="A", test="D", pair="AD"),
            "BC": dict(sample="B", test="C", pair="BC"), "BD": dict(sample="B", test="D", pair="BD"),
        }
        return cond_idx, meta

    if task == "gng":
        stim_epoch  = slice(int(n_on[0]), int(n_off[0]))
        stim_drive  = X[:, stim_epoch, 4:6].mean(dim=1)
        is_go       = stim_drive[:, 0] > stim_drive[:, 1]
        cond_idx    = {"Go": torch.where(is_go)[0], "NoGo": torch.where(~is_go)[0]}
        meta        = {"Go": dict(gng="go"), "NoGo": dict(gng="nogo")}
        return cond_idx, meta

    if task == "dual":
        if condition_names is None:
            raise ValueError("For task='dual', pass condition_names from generate_dual_trials.")
        names = np.asarray(condition_names).astype(str)

        if dual_mode == "conditions":
            cond_idx = {}
            meta     = {}
            for name in sorted(np.unique(names)):
                idx = np.where(names == name)[0]
                cond_idx[name] = torch.as_tensor(idx, dtype=torch.long)
                meta[name]     = _parse_dual_name(name)
            return cond_idx, meta

        if dual_mode == "groups":
            groups = {"AC": [], "AD": [], "BC": [], "BD": [], "Go": [], "NoGo": [], "DPAOnly": []}
            for i, name in enumerate(names):
                info = _parse_dual_name(name)
                if info["pair"] in groups:
                    groups[info["pair"]].append(i)
                if info["gng"] == "go":      groups["Go"].append(i)
                elif info["gng"] == "nogo":  groups["NoGo"].append(i)
                elif info["gng"] == "none":  groups["DPAOnly"].append(i)
            cond_idx = {k: torch.as_tensor(v, dtype=torch.long) for k, v in groups.items()}
            meta = {
                "AC": dict(pair="AC"), "AD": dict(pair="AD"),
                "BC": dict(pair="BC"), "BD": dict(pair="BD"),
                "Go": dict(gng="go"), "NoGo": dict(gng="nogo"), "DPAOnly": dict(gng="none"),
            }
            return cond_idx, meta

    raise ValueError(f"Unknown task: {task}")


def flow_specs_for_task(timing, task, input_size, cond_idx, meta, dual_mode="conditions",
                        cue_on_go_input=False):
    task = task.lower()

    if task == "dpa":
        return [
            dict(name="Autonomous", dims=None, conds=["AC", "AD", "BC", "BD"]),
            dict(name="A", dims=[0], conds=["AC", "AD"]),
            dict(name="B", dims=[1], conds=["BC", "BD"]),
            dict(name="C", dims=[2], conds=["AC", "BC"]),
            dict(name="D", dims=[3], conds=["AD", "BD"]),
        ]

    if task == "gng":
        specs = [
            dict(name="Autonomous", dims=None, conds=["Go", "NoGo"]),
            dict(name="Go",   dims=[4], conds=["Go"]),
            dict(name="NoGo", dims=[5], conds=["NoGo"]),
        ]
        if not cue_on_go_input:
            specs.append(dict(name="Cue", dims=[6], conds=["Go", "NoGo"]))
        return specs

    if task == "dual":
        labels = list(cond_idx.keys())
        if dual_mode == "conditions":
            A_conds    = [c for c in labels if meta[c]["sample"] == "A"]
            B_conds    = [c for c in labels if meta[c]["sample"] == "B"]
            C_conds    = [c for c in labels if meta[c]["test"]   == "C"]
            D_conds    = [c for c in labels if meta[c]["test"]   == "D"]
            Go_conds   = [c for c in labels if meta[c]["gng"]    == "go"]
            NoGo_conds = [c for c in labels if meta[c]["gng"]    == "nogo"]
            auto_conds = labels
        else:
            auto_conds = ["AC", "AD", "BC", "BD", "DPAOnly"]
            A_conds, B_conds = ["AC", "AD"], ["BC", "BD"]
            C_conds, D_conds = ["AC", "BC"], ["AD", "BD"]
            Go_conds, NoGo_conds = ["Go"], ["NoGo"]

        specs = [
            dict(name="Autonomous", dims=None,  conds=auto_conds),
            dict(name="A",    dims=[0], conds=A_conds),
            dict(name="B",    dims=[1], conds=B_conds),
            dict(name="Go",   dims=[4], conds=Go_conds),
            dict(name="NoGo", dims=[5], conds=NoGo_conds),
        ]
        if not cue_on_go_input:
            specs.append(dict(name="Cue", dims=[6], conds=Go_conds + NoGo_conds))
        specs += [
            dict(name="C", dims=[2], conds=C_conds),
            dict(name="D", dims=[3], conds=D_conds),
        ]
        return specs

    raise ValueError(f"Unknown task: {task}")


def _condition_colors(cond_labels):
    base = {
        "AC": "tab:blue", "AD": "tab:orange", "BC": "tab:green", "BD": "tab:purple",
        "Go": "tab:red",  "NoGo": "tab:brown", "DPAOnly": "tab:gray",
    }
    cmap   = plt.get_cmap("tab20")
    colors = {}
    for i, c in enumerate(cond_labels):
        colors[c] = base.get(c, cmap(i % 20))
    return colors


# ---------------------------------------------------------------------------
# Main phase-portrait plotter
# ---------------------------------------------------------------------------

def plot_task_flow_fields(
    model, inputs, timing, task,
    targets=None, condition_names=None, dual_mode="conditions",
    xlim=None, ylim=None, n_grid=151, n_fp_seeds=41,
    include_beta_in_field=False, qlo=1, qhi=99, pad=1.25,
    symmetric_auto=True, speed_percentile=98,
    figsize_per_panel=4.2, input_threshold=0.35, inactive_atol=0.35,
    show_single_trials=False, max_single_trials=12, max_autonomous_conditions=None,
    cue_on_go_input=False,
    use_sim_field=False, sim_n_warmup=0,
):
    """Generic low-rank phase portrait for DPA, GNG, and Dual tasks."""
    task = task.lower()
    if model.m.shape[1] != 2:
        raise ValueError("This plotter assumes rank == 2.")

    model.eval()
    device     = _model_device(model)
    dtype      = _model_dtype(model)
    input_size = inputs.shape[-1]

    with torch.no_grad():
        readouts, rates, rec_inputs, effective_x = run_low_rank_with_effective_inputs(
            model, inputs, targets=targets
        )

    kappa_traj = project_rec_inputs_to_kappa(model, rec_inputs).detach().cpu().numpy()

    if xlim is None or ylim is None:
        auto_xlim, auto_ylim = auto_limits_from_trajs(
            kappa_traj[..., :2], qlo=qlo, qhi=qhi, pad=pad, symmetric=symmetric_auto
        )
        xlim = auto_xlim if xlim is None else xlim
        ylim = auto_ylim if ylim is None else ylim

    cond_idx, meta = condition_indices_for_task(
        inputs.detach().cpu(), timing=timing, task=task,
        condition_names=condition_names, dual_mode=dual_mode,
    )
    specs = flow_specs_for_task(
        timing=timing, task=task, input_size=input_size,
        cond_idx=cond_idx, meta=meta, dual_mode=dual_mode,
        cue_on_go_input=cue_on_go_input,
    )

    max_dim = max(
        [d for spec in specs if spec["dims"] is not None for d in spec["dims"]], default=-1
    )
    if input_size <= max_dim:
        raise ValueError(f"Task {task} requires input_size > {max_dim}, got {input_size}.")

    colors = _condition_colors(cond_idx.keys())

    sim_scattered = use_sim_field and sim_n_warmup > 0   # quiver vs streamplot

    caches, all_speeds = [], []
    for spec in specs:
        ff_input = make_input(input_size, active_dims=spec["dims"], value=1.0, device=device, dtype=dtype)

        if use_sim_field:
            K1, K2, U, V, speed = sim_kappa_field(
                model, ff_input, xlim=xlim, ylim=ylim, n_grid=n_grid,
                n_warmup=sim_n_warmup,
            )
            fixed_points, fixed_residuals = find_sim_fixed_points(
                model, ff_input, xlim=xlim, ylim=ylim,
                n_seeds=n_fp_seeds, n_warmup=sim_n_warmup,
                residual_tol=1e-5, merge_tol=5e-2,
            )
            fp_labels, fp_eigvals = classify_sim_fixed_points(
                model, fixed_points, ff_input, n_warmup=sim_n_warmup,
            )
        else:
            K1, K2, U, V, speed, params, ff_input_np = make_vector_field_grid(
                model, ff_input=ff_input, xlim=xlim, ylim=ylim, n_grid=n_grid,
                include_beta=include_beta_in_field,
            )
            fixed_points, fixed_residuals = find_all_fixed_points(
                model, xlim=xlim, ylim=ylim, ff_input=ff_input,
                n_seeds=n_fp_seeds, residual_tol=1e-8, merge_tol=5e-2,
            )
            fp_labels, fp_eigvals = classify_fixed_points(model, fixed_points, ff_input=ff_input)

        panel_mask = _canonical_input_mask(
            effective_x, dims=spec["dims"], threshold=input_threshold, atol_inactive=inactive_atol
        )
        caches.append(dict(
            spec=spec, ff_input=ff_input,
            K1=K1, K2=K2, U=U, V=V, speed=speed,
            fixed_points=fixed_points, fixed_residuals=fixed_residuals,
            fp_labels=fp_labels, fp_eigvals=fp_eigvals,
            panel_mask=panel_mask.detach().cpu().numpy(),
        ))
        all_speeds.append(speed[np.isfinite(speed)].reshape(-1))

    speed_vmax = np.percentile(np.concatenate(all_speeds), speed_percentile)

    n_panels = len(specs)
    fig = plt.figure(figsize=(n_panels * figsize_per_panel + 0.45, figsize_per_panel),
                     constrained_layout=False)
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(
        1, n_panels + 1,
        width_ratios=[1.0] * n_panels + [0.045],
        left=0.045, right=0.965, bottom=0.16, top=0.84, wspace=0.16,
    )
    axes = []
    for i in range(n_panels):
        ax_i = fig.add_subplot(gs[0, i]) if i == 0 else fig.add_subplot(gs[0, i], sharex=axes[0], sharey=axes[0])
        axes.append(ax_i)
    axes = np.asarray(axes, dtype=object)
    cax  = fig.add_subplot(gs[0, -1])

    last_hm = None

    for ax, cache in zip(axes, caches):
        spec = cache["spec"]
        K1, K2, U, V, speed = cache["K1"], cache["K2"], cache["U"], cache["V"], cache["speed"]

        last_hm = ax.pcolormesh(K1, K2, speed, shading="auto", cmap="magma",
                                vmax=speed_vmax, rasterized=True, zorder=0)
        if sim_scattered:
            # Warmed-up sim field: K1/K2 are scattered → quiver
            ax.quiver(K1, K2, U, V, color="white", alpha=0.7,
                      scale=None, scale_units="xy", angles="xy", zorder=2)
        else:
            ax.streamplot(K1[0, :], K2[:, 0], U, V, color="white", density=1.05,
                          linewidth=0.70, arrowsize=0.80, zorder=2)

        for label, marker in [("attractor","o"),("saddle","x"),("repeller","^"),("nonhyperbolic","D")]:
            if len(cache["fp_labels"]) == 0: continue
            mask = cache["fp_labels"] == label
            if not np.any(mask): continue
            pts = cache["fixed_points"][mask]
            if label == "saddle":
                ax.scatter(pts[:, 0], pts[:, 1], s=60, marker="x", color="cyan", linewidths=1.5, zorder=10, label=label)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=60, marker=marker,
                           facecolors="cyan" if label == "attractor" else "none",
                           edgecolors="cyan", linewidths=1.1, zorder=10, label=label)

        conds       = [] if spec["dims"] is None else list(spec["conds"])
        panel_mask  = cache["panel_mask"]

        if spec["name"] == "Autonomous" and max_autonomous_conditions is not None:
            conds = conds[:max_autonomous_conditions]

        for cond in conds:
            if cond not in cond_idx: continue
            idx = cond_idx[cond].detach().cpu().numpy()
            if len(idx) == 0: continue

            cond_mask = panel_mask[idx].mean(axis=0) > 0.5
            segments  = _segments_from_mask(cond_mask, min_len=2)
            if not segments: continue

            traj_mean = kappa_traj[idx].mean(axis=0)
            color     = colors.get(cond, "tab:gray")

            for seg_slice in segments:
                seg = traj_mean[seg_slice, :2]
                if seg.shape[0] < 2: continue

                if show_single_trials:
                    for i in idx[:max_single_trials]:
                        seg_i = kappa_traj[i, seg_slice, :2]
                        ax.plot(seg_i[:, 0], seg_i[:, 1], color=color, lw=0.7, alpha=0.12, zorder=4)

                ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2.7, alpha=0.95, zorder=6, label=cond)
                add_time_arrows(ax, seg, times=default_arrow_times(seg.shape[0], 4, 3),
                                color=color, arrow_step=3, mutation_scale=13, lw=1.5, alpha=0.95, zorder=8)
                ax.scatter(seg[0, 0],  seg[0, 1],  marker="o", s=38, color=color, edgecolors="white", linewidths=0.6, zorder=9)
                ax.scatter(seg[-1, 0], seg[-1, 1], marker="X", s=46, color=color, edgecolors="white", linewidths=0.6, zorder=9)

        ax.set_title(spec["name"])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\kappa_1$")

    axes[0].set_ylabel(r"$\kappa_2$")

    handles, labels_list = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels_list.extend(l)
    unique = dict(zip(labels_list, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=7, loc="best", handlelength=1.1)

    cbar = fig.colorbar(last_hm, cax=cax)
    if use_sim_field:
        cbar.set_label(r"$\|\Delta\kappa\|$ (simulation)")
    elif include_beta_in_field:
        cbar.set_label(r"$\beta\|\Psi(\kappa;x)-\kappa\|$")
    else:
        cbar.set_label(r"$\|\Psi(\kappa;x)-\kappa\|$")
    fig.suptitle(f"{task.upper()} low-rank dynamics: autonomous and frozen-input fields", y=0.965)
    fig.set_layout_engine("none")

    return fig, axes, {
        "task": task, "kappa": kappa_traj,
        "readouts": readouts.detach().cpu(), "rates": rates.detach().cpu(),
        "rec_inputs": rec_inputs.detach().cpu(), "effective_x": effective_x.detach().cpu(),
        "condition_indices": cond_idx, "condition_meta": meta,
        "panel_specs": specs, "caches": caches,
    }


# ---------------------------------------------------------------------------
# Simulation-based κ-plane flow field  (exact: includes W_fixed, full dynamics)
# ---------------------------------------------------------------------------

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
                               eig_tol=1e-5, eps=1e-4):
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

        n_stable   = np.sum(rad < 1.0 - eig_tol)
        n_unstable = np.sum(rad > 1.0 + eig_tol)
        if n_stable == len(rad):      label = "attractor"
        elif n_unstable == len(rad):  label = "repeller"
        elif n_stable > 0:            label = "saddle"
        else:                         label = "nonhyperbolic"

        labels.append(label)
        eigvals.append(ev)

    if not eigvals:
        return np.asarray([], dtype=object), np.empty((0, 2), dtype=np.complex128)
    return np.asarray(labels, dtype=object), np.stack(eigvals)


# ---------------------------------------------------------------------------
# Rank-2 torch-native flow field
# ---------------------------------------------------------------------------

@contextmanager
def temporary_attrs(obj, **kwargs):
    old = {k: getattr(obj, k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@torch.no_grad()
def rank2_kappa_delta_torch(model, x_t, kappa, q_slice="same"):
    """
    Compute Δκ for the rank-2 projected dynamics.

    q_slice: "same" (q = κ, slaved closure) or "zero" (q = 0).
    """
    assert model.rank == 2
    device = model.device
    kappa  = kappa.to(device)
    if x_t.ndim == 1: x_t = x_t[None, :]
    x_t = x_t.to(device)

    G, N = kappa.shape[0], model.hidden_size
    m, n = model.m.detach(), model.n.detach()

    if q_slice == "same":  q = kappa
    elif q_slice == "zero": q = torch.zeros_like(kappa)
    else: raise ValueError("q_slice must be 'same' or 'zero'.")

    beta_h = model.exp_alpha_rec.to(device)
    beta_r = model.exp_alpha.to(device)

    q_next = beta_h * q + (1.0 - beta_h) * kappa

    if model.wi is not None:
        input_drive = model.Ai * model.wi(x_t.expand(G, -1))
    else:
        input_drive = 0.0

    phi        = model.nonlinearity(model.gain * (input_drive + q_next @ m.T))
    F          = phi @ n / N
    kappa_next = beta_r * kappa + (1.0 - beta_r) * F
    return kappa_next - kappa


@torch.no_grad()
def rank2_kappa_flow(model, x_t, kappa1_lim=(-3.0, 3.0), kappa2_lim=(-3.0, 3.0),
                     grid_size=61, q_slice="same"):
    assert model.rank == 2
    device = model.device
    k1 = torch.linspace(kappa1_lim[0], kappa1_lim[1], grid_size, device=device)
    k2 = torch.linspace(kappa2_lim[0], kappa2_lim[1], grid_size, device=device)
    K1, K2 = torch.meshgrid(k1, k2, indexing="ij")
    kappa   = torch.stack([K1.reshape(-1), K2.reshape(-1)], dim=-1)

    d_kappa = rank2_kappa_delta_torch(model, x_t=x_t, kappa=kappa, q_slice=q_slice)
    U = d_kappa[:, 0].reshape(grid_size, grid_size)
    V = d_kappa[:, 1].reshape(grid_size, grid_size)
    speed = torch.sqrt(U**2 + V**2)

    return K1.cpu().numpy(), K2.cpu().numpy(), U.cpu().numpy(), V.cpu().numpy(), speed.cpu().numpy()


# ---------------------------------------------------------------------------
# Fixed points (torch-native)
# ---------------------------------------------------------------------------

def classify_fixed_point(model, x_t, kappa_star, q_slice="same", eps=1e-4):
    kappa_star = np.asarray(kappa_star, dtype=np.float64)
    J_delta    = np.zeros((2, 2), dtype=np.float64)

    for j in range(2):
        step = np.zeros(2)
        step[j] = eps
        kp = torch.tensor(kappa_star + step, dtype=torch.float32, device=model.device)[None, :]
        km = torch.tensor(kappa_star - step, dtype=torch.float32, device=model.device)[None, :]
        fp = rank2_kappa_delta_torch(model, x_t, kp, q_slice=q_slice).cpu().numpy()[0]
        fm = rank2_kappa_delta_torch(model, x_t, km, q_slice=q_slice).cpu().numpy()[0]
        J_delta[:, j] = (fp - fm) / (2.0 * eps)

    J_map   = np.eye(2) + J_delta
    eigvals = np.linalg.eigvals(J_map)
    abs_eigs= np.abs(eigvals)

    if   np.all(abs_eigs < 1.0): kind = "stable"
    elif np.all(abs_eigs > 1.0): kind = "unstable"
    else:                         kind = "saddle"

    return eigvals, kind


def find_rank2_fixed_points(model, x_t, kappa1_lim=(-3, 3), kappa2_lim=(-3, 3),
                             n_init=21, q_slice="same", root_tol=1e-8,
                             residual_tol=1e-5, duplicate_tol=1e-3, pad=0.25):
    assert model.rank == 2

    def f_np(z):
        z_t = torch.tensor(z, dtype=torch.float32, device=model.device)[None, :]
        return rank2_kappa_delta_torch(model, x_t=x_t, kappa=z_t, q_slice=q_slice).cpu().numpy()[0].astype(np.float64)

    fixed_points = []

    for k1 in np.linspace(kappa1_lim[0], kappa1_lim[1], n_init):
        for k2 in np.linspace(kappa2_lim[0], kappa2_lim[1], n_init):
            sol = root(f_np, np.array([k1, k2], dtype=np.float64), method="hybr", tol=root_tol)
            if not sol.success: continue
            z     = sol.x.astype(np.float64)
            resid = float(np.linalg.norm(f_np(z)))
            in_bounds = (
                kappa1_lim[0] - pad <= z[0] <= kappa1_lim[1] + pad and
                kappa2_lim[0] - pad <= z[1] <= kappa2_lim[1] + pad
            )
            if resid > residual_tol or not in_bounds: continue

            is_new = True
            for fp in fixed_points:
                if np.linalg.norm(z - fp["kappa"]) < duplicate_tol:
                    is_new = False
                    if resid < fp["residual"]:
                        fp["kappa"], fp["residual"] = z, resid
                    break

            if is_new:
                eigvals, kind = classify_fixed_point(model, x_t, z, q_slice=q_slice)
                fixed_points.append({"kappa": z, "residual": resid, "eigvals": eigvals, "kind": kind})

    return sorted(fixed_points, key=lambda d: (d["kappa"][0], d["kappa"][1]))


def overlay_fixed_points(ax, fixed_points, label=False, stable_kwargs=None,
                         saddle_kwargs=None, unstable_kwargs=None):
    stable_kwargs   = stable_kwargs   or dict(s=200, marker="o", facecolor="white", edgecolor="black", linewidth=1.0, zorder=40)
    saddle_kwargs   = saddle_kwargs   or dict(s=200, marker="x", color="white", linewidth=2.0, zorder=40)
    unstable_kwargs = unstable_kwargs or dict(s=200, marker="o", facecolor="none", edgecolor="red", linewidth=1.7, zorder=40)
    used_labels = set()

    for fp in fixed_points:
        k, kind = fp["kappa"], fp["kind"]
        if   kind == "stable":   kwargs, lab = stable_kwargs.copy(),   "stable FP"
        elif kind == "saddle":   kwargs, lab = saddle_kwargs.copy(),   "saddle FP"
        else:                    kwargs, lab = unstable_kwargs.copy(), "unstable FP"
        if label and lab not in used_labels:
            kwargs["label"] = lab
            used_labels.add(lab)
        ax.scatter(k[0], k[1], **kwargs)


# ---------------------------------------------------------------------------
# Heatmap + streamplot panel
# ---------------------------------------------------------------------------

def plot_kappa_heatmap_flow_panel(ax, K1, K2, U, V, heatmap=None, title=None,
                                   cmap="magma", flow_color="white", density=1.2,
                                   linewidth=0.8, arrowsize=1.0, normalize_flow=False,
                                   vmin=None, vmax=None):
    if heatmap is None:
        heatmap = np.sqrt(U**2 + V**2)

    if normalize_flow:
        speed   = np.sqrt(U**2 + V**2)
        U_plot  = U / (speed + 1e-12)
        V_plot  = V / (speed + 1e-12)
    else:
        U_plot, V_plot = U, V

    hm = ax.pcolormesh(K1.T, K2.T, heatmap.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.streamplot(K1.T, K2.T, U_plot.T, V_plot.T, color=flow_color,
                  density=density, linewidth=linewidth, arrowsize=arrowsize)
    ax.set_xlabel(r"$\kappa_1$")
    ax.set_ylabel(r"$\kappa_2$")
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)
    return hm


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_kappa_trajectories(model, ff_inputs, targets=None, disable_rwd=False):
    """Run the model and return κ_t = r_t @ n / N."""
    ff_inputs = ff_inputs.to(model.device)
    if targets is not None:
        targets = targets.to(model.device)
    attrs = {"noise": 0.0}
    if disable_rwd:
        attrs["rwd"] = False
    with temporary_attrs(model, **attrs):
        readout, rates, rec_inputs = model(ff_inputs, targets=targets, ret_rates=True)
    kappa = rates @ model.n.detach() / model.hidden_size
    return kappa.cpu().numpy(), rates, rec_inputs, readout


def infer_dpa_labels_from_inputs(ff_inputs, timing=None, sample_channels=(0, 1), test_channels=(2, 3)):
    x = ff_inputs.detach().cpu() if torch.is_tensor(ff_inputs) else torch.as_tensor(ff_inputs)

    if timing is not None:
        n_on, n_off = timing.n_stim_on, timing.n_stim_off
        s_sample = slice(int(n_on[0]), int(n_off[0]))
        s_test   = slice(int(n_on[3]), int(n_off[3])) if len(n_on) >= 4 else slice(int(n_on[1]), int(n_off[1]))
        sample_score = x[:, s_sample, list(sample_channels)].mean(dim=1)
        test_score   = x[:, s_test,   list(test_channels)].mean(dim=1)
    else:
        sample_score = x[:, :, list(sample_channels)].mean(dim=1)
        test_score   = x[:, :, list(test_channels)].mean(dim=1)

    sample_char = np.where(sample_score.argmax(dim=1).numpy() == 0, "A", "B")
    test_char   = np.where(test_score.argmax(dim=1).numpy()   == 0, "C", "D")
    return np.char.add(sample_char, test_char)


def dpa_labels_from_condition_names(condition_names):
    labels = []
    for name in condition_names:
        s = str(name)
        if   s.startswith("A"): sample = "A"
        elif s.startswith("B"): sample = "B"
        else: raise ValueError(f"Cannot parse sample from condition name: {s}")
        if   s.endswith("C"): test = "C"
        elif s.endswith("D"): test = "D"
        else: raise ValueError(f"Cannot parse test from condition name: {s}")
        labels.append(sample + test)
    return np.asarray(labels)


def overlay_kappa_trajectories(ax, kappa, trial_inds=None, color="cyan", alpha=0.25,
                                linewidth=0.9, start_marker=False, end_marker=False):
    B = kappa.shape[0]
    if trial_inds is None:
        trial_inds = range(B)
    for b in trial_inds:
        traj = kappa[b]
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=linewidth, zorder=7)
        if start_marker:
            ax.scatter(traj[0, 0], traj[0, 1], color=color, s=20, marker="o", alpha=alpha, zorder=8)
        if end_marker:
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=30, marker="x", alpha=alpha, zorder=8)


def overlay_average_kappa_trajectories(ax, kappa, labels, trial_types=("AC", "AD", "BC", "BD"),
                                        colors=None, linewidth=3.0, alpha=0.95,
                                        start_marker=True, end_marker=True, label_prefix=""):
    if colors is None:
        colors = {"AC": "tab:blue", "AD": "tab:orange", "BC": "tab:green", "BD": "tab:red"}
    labels = np.asarray(labels)
    for tt in trial_types:
        mask = labels == tt
        if not np.any(mask): continue
        mean_traj = kappa[mask].mean(axis=0)
        color     = colors.get(tt, "black")
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=color, linewidth=linewidth,
                alpha=alpha, label=f"{label_prefix}{tt} mean", zorder=12)
        if start_marker:
            ax.scatter(mean_traj[0, 0],  mean_traj[0, 1],  color=color, s=45, marker="o",
                       edgecolor="black", linewidth=0.7, zorder=13)
        if end_marker:
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], color=color, s=65, marker="X",
                       edgecolor="black", linewidth=0.7, zorder=13)


# ---------------------------------------------------------------------------
# Two-panel autonomous vs input-driven figure
# ---------------------------------------------------------------------------

def plot_autonomous_vs_input_heatmap_flow(
    model, x_autonomous, x_input,
    ff_inputs=None, targets=None, timing=None, condition_names=None,
    kappa1_lim=(-3, 3), kappa2_lim=(-3, 3), grid_size=61, q_slice="same",
    heatmap_kind="speed", cmap="magma", flow_color="white", normalize_flow=False,
    shared_color_scale=True, show_fixed_points=True, fp_n_init=21,
    overlay_average=True, overlay_individual=False, trial_inds=None,
    avg_trial_types=("AC", "AD", "BC", "BD"), avg_colors=None,
):
    K1a, K2a, Ua, Va, Sa = rank2_kappa_flow(model, x_t=x_autonomous, kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, grid_size=grid_size, q_slice=q_slice)
    K1i, K2i, Ui, Vi, Si = rank2_kappa_flow(model, x_t=x_input,      kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, grid_size=grid_size, q_slice=q_slice)

    if   heatmap_kind == "speed": Ha, Hi, cbar_label = Sa, Si, r"$\|\Delta \kappa\|$"
    elif heatmap_kind == "dK1":   Ha, Hi, cbar_label = Ua, Ui, r"$\Delta \kappa_1$"
    elif heatmap_kind == "dK2":   Ha, Hi, cbar_label = Va, Vi, r"$\Delta \kappa_2$"
    else: raise ValueError("heatmap_kind must be 'speed', 'dK1', or 'dK2'.")

    vmin = min(np.nanmin(Ha), np.nanmin(Hi)) if shared_color_scale else None
    vmax = max(np.nanmax(Ha), np.nanmax(Hi)) if shared_color_scale else None

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True, sharex=True, sharey=True)

    hm0 = plot_kappa_heatmap_flow_panel(axs[0], K1a, K2a, Ua, Va, heatmap=Ha, title="Autonomous / baseline",
                                         cmap=cmap, flow_color=flow_color, normalize_flow=normalize_flow, vmin=vmin, vmax=vmax)
    hm1 = plot_kappa_heatmap_flow_panel(axs[1], K1i, K2i, Ui, Vi, heatmap=Hi, title="Input-driven",
                                         cmap=cmap, flow_color=flow_color, normalize_flow=normalize_flow, vmin=vmin, vmax=vmax)
    for ax in axs:
        ax.set_xlim(kappa1_lim); ax.set_ylim(kappa2_lim)

    fps_aut, fps_inp = [], []
    if show_fixed_points:
        fps_aut = find_rank2_fixed_points(model, x_t=x_autonomous, kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, n_init=fp_n_init, q_slice=q_slice)
        fps_inp = find_rank2_fixed_points(model, x_t=x_input,      kappa1_lim=kappa1_lim, kappa2_lim=kappa2_lim, n_init=fp_n_init, q_slice=q_slice)
        overlay_fixed_points(axs[0], fps_aut)
        overlay_fixed_points(axs[1], fps_inp)

    if ff_inputs is not None:
        kappa, rates, rec_inputs, readout = get_kappa_trajectories(model, ff_inputs=ff_inputs, targets=targets, disable_rwd=True)
        dpa_labels = (
            dpa_labels_from_condition_names(condition_names) if condition_names is not None
            else infer_dpa_labels_from_inputs(ff_inputs, timing=timing)
        )
        if overlay_individual:
            for ax in axs:
                overlay_kappa_trajectories(ax, kappa, trial_inds=trial_inds, color="cyan", alpha=0.25)
        if overlay_average:
            for ax in axs:
                overlay_average_kappa_trajectories(ax, kappa, dpa_labels, trial_types=avg_trial_types, colors=avg_colors)

    cbar = fig.colorbar(hm1, ax=axs, shrink=0.85)
    cbar.set_label(cbar_label)
    handles, labels_list = axs[1].get_legend_handles_labels()
    if handles:
        axs[1].legend(handles, labels_list, loc="upper right", frameon=True, fontsize=8)

    return fig, axs, fps_aut, fps_inp
