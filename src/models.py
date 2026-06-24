from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn


class LowRankModel(nn.Module):
    """
    Low-rank RNN with optional reward-feedback input channel.
    Reward feedback is teacher-forced.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        rank=1,
        gain=1.0,
        alpha=0.01,
        alpha_rec=0.01,
        train_inputs=True,
        train_outputs=True,
        rwd_channel=-1,
        rwd_thresh=0.5,
        rwd=True,
        rwd_scale=1.0,
        noise=0.1,
        use_fixed_weights=False,
        fixed_weight_scale=0.8,
        fixed_weight_orthogonalize=True,
        fixed_weight_sparsity=1.0,
        nonlinearity="tanh",
        nl_gamma=0.0,
        use_unit_bias=False,
        unit_bias_trainable=True,
        unit_bias_scale=0.2,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.rank        = rank
        self.register_buffer('gain', torch.tensor(float(gain), device=self.device))

        self.rwd         = rwd
        self.rwd_channel = rwd_channel
        self.rwd_thresh  = rwd_thresh
        self.rwd_scale   = rwd_scale
        self.noise       = noise

        self.wi = None
        if input_size > 0:
            self.Ai = 1.0
            self.wi = nn.Linear(input_size, hidden_size, device=self.device)
            with torch.no_grad():
                nn.init.normal_(self.wi.weight, mean=0.0, std=1.0)
                nn.init.zeros_(self.wi.bias)
            if not train_inputs:
                self.Ai = nn.Parameter(torch.randn(1, device=self.device))
                self.wi.weight.requires_grad_(False)
                self.wi.bias.requires_grad_(False)

        self.m = nn.Parameter(torch.randn(hidden_size, rank, device=self.device))
        self.n = nn.Parameter(torch.randn(hidden_size, rank, device=self.device))
        nn.init.normal_(self.m, mean=0.0, std=1.0)
        nn.init.normal_(self.n, mean=0.0, std=1.0)

        self.wo = None
        if output_size > 0:
            self.Ao = 1.0
            self.wo = nn.Linear(hidden_size, output_size, device=self.device)
            with torch.no_grad():
                nn.init.normal_(self.wo.weight, mean=0.0, std=1.0)
                nn.init.zeros_(self.wo.bias)
            if not train_outputs:
                self.Ao = nn.Parameter(torch.randn(1, device=self.device))
                self.wo.weight.requires_grad_(False)
                self.wo.bias.requires_grad_(False)

        import math as _math
        _sqrt2   = _math.sqrt(2.0)
        _sqrtpi  = _math.sqrt(_math.pi)
        _nl = {"tanh": torch.tanh, "relu": torch.relu,
               "softplus": torch.nn.functional.softplus,
               "erf":      torch.erf,
               "elu":      torch.nn.functional.elu,
               # Brunel LIF approx: Gaussian CDF, range [0,1], φ'(0)=1/√(2π)≈0.4
               "lif":      lambda x: (1.0 + torch.erf(x / _sqrt2)) / 2.0,
               # Rescaled LIF: φ'(0)=1 (matches tanh/relu at origin), range [0,1]
               "lif_sc":   lambda x: (1.0 + torch.erf(x * _sqrtpi)) / 2.0,
               # Asymmetric saturating tanh: saturates to 1+γ / -1+γ; the γ·tanh² even
               # component breaks the odd κ-symmetry intrinsically (non-removable by bias).
               "tanh_asym": (lambda g: (lambda x: torch.tanh(x) + g * torch.tanh(x) ** 2))(float(nl_gamma))}
        self.nl_gamma         = float(nl_gamma)
        self.nonlinearity     = _nl[nonlinearity]
        self.nonlinearity_str = nonlinearity

        # Per-unit bias inside the nonlinearity: φ(gain·(Ai·Wi·x + h) + b).
        # Random init breaks the odd symmetry of the κ-field generically (its readout
        # projection is random-signed, NOT pre-aligned to push attractors down), so the
        # ring is *free* to settle off-origin rather than being forced there. Free in all
        # stages (independent of the input-freeze).
        self.use_unit_bias = use_unit_bias
        if use_unit_bias:
            b0 = unit_bias_scale * torch.randn(hidden_size, device=self.device)
            if unit_bias_trainable:
                self.unit_bias = nn.Parameter(b0)
            else:
                self.register_buffer("unit_bias", b0.detach())
        else:
            self.register_buffer("unit_bias", torch.zeros(hidden_size, device=self.device),
                                 persistent=False)
        self.register_buffer("alpha_rec",     torch.tensor(float(alpha_rec)))
        self.register_buffer("exp_alpha_rec", torch.exp(-torch.tensor(float(alpha_rec))))
        self.register_buffer("alpha",         torch.tensor(float(alpha)))
        self.register_buffer("exp_alpha",     torch.exp(-torch.tensor(float(alpha))))

        # Fixed random recurrent backbone. **Persistent** (saved in state_dict) so the
        # exact trained backbone is reloaded for analysis/flow — non-orthogonalized
        # backbones shape the κ-plane and cannot be regenerated reproducibly otherwise.
        # Optionally projected ⊥ m,n so n^T W_fixed = 0 and W_fixed^T m = 0 (keeps the
        # κ-plane reduction valid). Old checkpoints without this key load fine under
        # load_state_dict(strict=False).
        self.use_fixed_weights = use_fixed_weights
        if use_fixed_weights:
            w_fixed = torch.randn(hidden_size, hidden_size, device=self.device)
            # Sparse backbone: keep each entry with prob p (Bernoulli mask), rescale
            # nonzeros by 1/sqrt(p) so the effective recurrent gain (row-sum variance)
            # matches the dense case. p=1.0 → fully dense.
            p = float(fixed_weight_sparsity)
            if p < 1.0:
                mask = (torch.rand(hidden_size, hidden_size, device=self.device) < p).float()
                w_fixed = w_fixed * mask / (p ** 0.5)
            if fixed_weight_orthogonalize:
                n_hat = self.n / self.n.norm(dim=0, keepdim=True).clamp_min(1e-12)
                m_hat = self.m / self.m.norm(dim=0, keepdim=True).clamp_min(1e-12)
                w_fixed = w_fixed - n_hat @ (n_hat.T @ w_fixed)
                w_fixed = w_fixed - (w_fixed @ m_hat) @ m_hat.T
            w_fixed *= fixed_weight_scale / (hidden_size ** 0.5)
            self.register_buffer("w_fixed", w_fixed.detach(), persistent=True)
        else:
            self.w_fixed = None

    def get_readout(self, rates, rec_inputs):
        rec = rates @ self.n / self.hidden_size
        if self.wo is not None:
            ext = self.Ao * self.wo(rates)
            return torch.cat((rec, ext), dim=-1)
        return rec

    def update_dynamics(self, ff_inputs, rec_inputs, rates):
        input_drive = self.Ai * self.wi(ff_inputs) if self.wi is not None else 0.0
        hidden      = (rates @ self.n) @ self.m.T / self.hidden_size
        if self.w_fixed is not None:
            hidden = hidden + rates @ self.w_fixed.T
        rec_inputs  = self.exp_alpha_rec * rec_inputs + (1.0 - self.exp_alpha_rec) * hidden
        phi         = self.nonlinearity(self.gain * (input_drive + rec_inputs) + self.unit_bias)
        rates       = self.exp_alpha * rates + (1.0 - self.exp_alpha) * phi
        return rates, rec_inputs

    def forward(self, ff_inputs, targets=None, ret_rates=False):
        "ff_inputs: (B, T, input_size)"
        B, T = ff_inputs.shape[0], ff_inputs.shape[1]
        N    = self.hidden_size

        rec_inputs = torch.zeros(B, N, device=self.device)
        rates      = torch.zeros(B, N, device=self.device)

        readout_list = []
        rates_list   = [] if ret_rates else None
        inputs_list  = [] if ret_rates else None

        rwd_next = torch.zeros(B, device=self.device, dtype=ff_inputs.dtype)

        for step in range(T):
            x_t = ff_inputs[:, step].clone()

            if torch.any(rwd_next) and self.rwd:
                x_t[:, self.rwd_channel] = x_t[:, self.rwd_channel] + self.rwd_scale * rwd_next

            noise  = self.noise * torch.randn(B, N, device=self.device)
            rates, rec_inputs = self.update_dynamics(x_t, rec_inputs + noise, rates)

            readout = self.get_readout(rates, rec_inputs)
            readout_list.append(readout)

            if ret_rates:
                rates_list.append(rates)
                inputs_list.append(rec_inputs)

            if self.rwd:
                if targets is not None:
                    rwd_mask = (targets[:, step, self.rwd_channel] == 1) & (readout[:, self.rwd_channel] > 0.5)
                    rwd_next = rwd_mask.to(dtype=ff_inputs.dtype)
                else:
                    rwd_next.zero_()

        readout_list = torch.stack(readout_list, dim=1)
        if ret_rates:
            return (
                readout_list,
                torch.stack(rates_list, dim=1),
                torch.stack(inputs_list, dim=1),
            )
        return readout_list


# ---------------------------------------------------------------------------
# Excitatory–Inhibitory low-rank model
# ---------------------------------------------------------------------------

class EILowRankModel(nn.Module):
    """Dale-respecting EI RNN with a strong frozen static backbone + weak trained
    low-rank E→E perturbation, relu (non-negative) rates, inputs to E only.

    Units 0..n_exc-1 are excitatory, n_exc..N-1 inhibitory (N = n_exc + n_inh).
    - `W_static` (N×N, frozen buffer): Dale-signed (E source columns ≥0, I source
      columns ≤0), E/I balanced (mean drive ≈0), scaled to spectral radius
      `static_radius` ("strongly recurrent").
    - Trained low-rank on the **E→E block only**: `m,n` (n_exc×rank), free/signed.
      The *effective* E→E weight is `relu(W_static_EE + m@nᵀ/n_exc)` — the total
      weight is rectified to ≥0 each step (Dale via clipping; the strong positive
      backbone keeps violations rare), so signed κ survives.
    - Inputs reach only E. Readout κ = rates_E @ n / n_exc (signed).

    Interface matches LowRankModel (forward / get_readout / update_dynamics / .m /
    .n / .gain / .noise) so the training pipeline and sim-based κ tools work
    unchanged. The *analytic* low-rank fixed-point reduction does NOT apply (the
    static backbone adds N-dim dynamics) — use simulation-based flow tools.
    """

    def __init__(self, input_size, hidden_size=None, output_size=0, rank=2,
                 n_exc=512, n_inh=128, gain=1.0, alpha=0.01, alpha_rec=0.01,
                 static_radius=1.5, low_rank_scale=0.3, noise=0.0,
                 rwd=False, rwd_scale=1.0, rwd_channel=-1,
                 nonlinearity="relu", low_rank_full=False,
                 use_stp=False, stp_U=0.2, stp_tau_f=1.5, stp_tau_d=0.3, stp_dt=0.0225,
                 stp_rate_scale=1.0, device="cpu", seed=None):
        super().__init__()
        self.device = torch.device(device)
        self.n_exc, self.n_inh = n_exc, n_inh
        N = n_exc + n_inh
        self.hidden_size = N          # full state dim
        self.rank        = rank
        self.rwd         = rwd
        self.rwd_scale   = rwd_scale
        self.rwd_channel = rwd_channel
        self.noise       = noise
        self.register_buffer("gain", torch.tensor(float(gain), device=self.device))
        self.register_buffer("alpha",         torch.tensor(float(alpha)))
        self.register_buffer("exp_alpha",     torch.exp(-torch.tensor(float(alpha))))
        self.register_buffer("alpha_rec",     torch.tensor(float(alpha_rec)))
        self.register_buffer("exp_alpha_rec", torch.exp(-torch.tensor(float(alpha_rec))))

        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        # --- Dale-signed, balanced static backbone, scaled to spectral radius ---
        mag = torch.randn(N, N, generator=g).abs()      # positive magnitudes
        W = torch.empty(N, N)
        W[:, :n_exc] =  mag[:, :n_exc]                    # E sources: excitatory
        W[:, n_exc:] = -mag[:, n_exc:] * (n_exc / max(n_inh, 1))  # I sources: scaled for balance
        W = W - W.mean(dim=1, keepdim=True)              # zero row-mean → balanced net drive
        # re-apply Dale signs after balancing (mean-subtraction can flip a few entries)
        W[:, :n_exc] = W[:, :n_exc].clamp_min(0.0)
        W[:, n_exc:] = W[:, n_exc:].clamp_max(0.0)
        radius = torch.linalg.eigvals(W).abs().max().real.clamp_min(1e-6)
        W = W * (static_radius / radius)
        self.register_buffer("W_static", W.to(self.device))   # persistent → saved/reloaded

        # --- trained weak low-rank: E→E block only (default) or the WHOLE N×N graph ---
        # low_rank_full=True lets the rank-r perturbation recruit E↔I loops (and reads κ
        # over the full population) — more leverage to carve a persistent memory attractor.
        self.low_rank_full = low_rank_full
        lr_dim = N if low_rank_full else n_exc
        self.m = nn.Parameter(low_rank_scale * torch.randn(lr_dim, rank, generator=g).to(self.device))
        self.n = nn.Parameter(low_rank_scale * torch.randn(lr_dim, rank, generator=g).to(self.device))

        # --- inputs reach E only ---
        self.wi = nn.Linear(input_size, n_exc, device=self.device)
        with torch.no_grad():
            nn.init.normal_(self.wi.weight, mean=0.0, std=1.0)
            nn.init.zeros_(self.wi.bias)
        self.Ai = 1.0
        self.input_size = input_size

        _nl = {"relu": torch.relu, "tanh": torch.tanh,
               "softplus": torch.nn.functional.softplus}
        self.nonlinearity     = _nl[nonlinearity]
        self.nonlinearity_str = nonlinearity
        self.wo = None

        # --- short-term plasticity (Tsodyks–Markram) on E presynaptic terminals ---
        # u: facilitation (utilisation), x: depression (available resources). Gates ALL
        # outputs of each E neuron. Transmitted E rate = rate·(u·x)/U → gate=1 at rest,
        # facilitation lifts it above 1 so the active assembly self-sustains over the delay.
        self.use_stp   = use_stp
        self.stp_U     = float(stp_U)
        self.stp_tau_f = float(stp_tau_f)
        self.stp_tau_d = float(stp_tau_d)
        self.stp_dt    = float(stp_dt)
        self.stp_rate_scale = float(stp_rate_scale)   # normalises rE driving STP (large relu rates saturate it)

    def init_stp(self, B):
        u = torch.full((B, self.n_exc), self.stp_U, device=self.device)
        x = torch.ones((B, self.n_exc), device=self.device)
        return u, x

    def stp_update(self, rE, u, x):
        """One TM step from presynaptic E rates rE; returns (u_new, x_new), both in [0,1].

        STP is a forward dynamical process, not a learned mechanism — we DETACH it from
        the graph. The gate still modulates the forward dynamics (and thus the loss via the
        current rates), but gradients don't backprop through the long multiplicative u·x
        history, which otherwise stalls optimization of the low-rank weights.
        """
        dt, U = self.stp_dt, self.stp_U
        rE = rE.detach() * self.stp_rate_scale
        u_new = (u + dt * ((U - u) / self.stp_tau_f + U * (1.0 - u) * rE)).clamp(0.0, 1.0)
        x_new = (x + dt * ((1.0 - x) / self.stp_tau_d - u_new * x * rE)).clamp(0.0, 1.0)
        return u_new, x_new

    def _W_rec_eff(self):
        """Full N×N effective recurrent matrix. low_rank_full=False → low-rank on the
        E→E block (relu-rectified). True → low-rank on the whole graph, then Dale-clip
        the total per column-block (E cols ≥0, I cols ≤0)."""
        ne = self.n_exc
        if self.low_rank_full:
            W = self.W_static + self.m @ self.n.T / self.hidden_size
            # Dale-clip per column-block, rebuilt by concat (no in-place → autograd-safe)
            W_e = W[:, :ne].clamp_min(0.0)           # E sources stay excitatory
            W_i = W[:, ne:].clamp_max(0.0)           # I sources stay inhibitory
            return torch.cat([W_e, W_i], dim=1)
        W = self.W_static.clone()
        W[:ne, :ne] = torch.relu(self.W_static[:ne, :ne] + self.m @ self.n.T / ne)
        return W

    def update_dynamics(self, ff_inputs, rec_inputs, rates, W_rec=None, u=None, x=None):
        ne = self.n_exc
        if W_rec is None:
            W_rec = self._W_rec_eff()
        if self.use_stp and u is not None:
            gate   = (u * x) / self.stp_U                 # rest=1; facilitation lifts >1
            r_eff  = torch.cat([rates[:, :ne] * gate, rates[:, ne:]], dim=1)  # E presyn gated
            hidden = r_eff @ W_rec.T
            u, x   = self.stp_update(rates[:, :ne], u, x)
        else:
            hidden = rates @ W_rec.T                      # (B, N)
        drive  = torch.zeros_like(rec_inputs)
        drive[:, :ne] = self.Ai * self.wi(ff_inputs)      # inputs to E only
        rec_inputs = self.exp_alpha_rec * rec_inputs + (1.0 - self.exp_alpha_rec) * hidden
        phi        = self.nonlinearity(self.gain * (drive + rec_inputs))
        rates      = self.exp_alpha * rates + (1.0 - self.exp_alpha) * phi
        return rates, rec_inputs, u, x

    def get_readout(self, rates, rec_inputs):
        if self.low_rank_full:
            return rates @ self.n / self.hidden_size         # κ over full population
        return rates[:, :self.n_exc] @ self.n / self.n_exc   # κ over E population

    def forward(self, ff_inputs, targets=None, ret_rates=False):
        B, T = ff_inputs.shape[0], ff_inputs.shape[1]
        N = self.hidden_size
        rec_inputs = torch.zeros(B, N, device=self.device)
        rates      = torch.zeros(B, N, device=self.device)
        u, x = self.init_stp(B) if self.use_stp else (None, None)
        W_rec = self._W_rec_eff()                          # constant across the trial
        readout_list, rates_list, inputs_list = [], [], []
        for step in range(T):
            x_t = ff_inputs[:, step]
            if self.noise:
                rec_inputs = rec_inputs + self.noise * torch.randn(B, N, device=self.device)
            rates, rec_inputs, u, x = self.update_dynamics(x_t, rec_inputs, rates, W_rec=W_rec, u=u, x=x)
            readout_list.append(self.get_readout(rates, rec_inputs))
            if ret_rates:
                rates_list.append(rates); inputs_list.append(rec_inputs)
        readout = torch.stack(readout_list, dim=1)
        if ret_rates:
            return readout, torch.stack(rates_list, dim=1), torch.stack(inputs_list, dim=1)
        return readout


# ---------------------------------------------------------------------------
# Spectral radius utility
# ---------------------------------------------------------------------------

def set_lowrank_spectral_radius(model, target_radius=0.99):
    """
    Rescale m and n so that the spectral radius of the linearised map equals
    target_radius.  Uses the fact that non-zero eigenvalues of m n^T / N are
    those of C = n^T m / N.
    """
    with torch.no_grad():
        N = model.hidden_size
        C = (model.n.T @ model.m) / N
        eigenvalues = torch.linalg.eigvals(C)
        current_max = eigenvalues.abs().max().item()
        if current_max == 0:
            return
        alpha_     = model.alpha.item()
        desired_max = (target_radius - (1.0 - alpha_)) / alpha_
        scale = (desired_max / current_max) ** 0.5
        model.m.mul_(scale)
        model.n.mul_(scale)


# ---------------------------------------------------------------------------
# Multivariate Gaussian initializer
# ---------------------------------------------------------------------------

class LowRankInitializer:
    """
    Initialise a LowRankModel's loadings (m, n, wi, wo) as i.i.d. samples
    from N(0, Sigma) across neurons.

    Loading vector per neuron:
        [ m_1..m_R, n_1..n_R, I_1..I_{D_in}, w_1..w_{D_out} ]
    """

    BLOCKS = ("m", "n", "in", "out")

    def __init__(
        self,
        sigma: Optional[torch.Tensor] = None,
        var_m: float | torch.Tensor = 1.0,
        var_n: float | torch.Tensor = 1.0,
        var_in: float | torch.Tensor = 1.0,
        var_out: float | torch.Tensor = 1.0,
        overlaps: Optional[dict] = None,
        jitter: float = 1e-8,
        zero_biases: bool = True,
        auto_psd: bool = True,
        psd_tol: float = 1e-8,
        psd_warn_threshold: float = 1e-3,
        preserve_diagonal: bool = True,
    ):
        self.sigma             = sigma
        self.var_m             = var_m
        self.var_n             = var_n
        self.var_in            = var_in
        self.var_out           = var_out
        self.overlaps          = overlaps or {}
        self.jitter            = jitter
        self.zero_biases       = zero_biases
        self.auto_psd          = auto_psd
        self.psd_tol           = psd_tol
        self.psd_warn_threshold= psd_warn_threshold
        self.preserve_diagonal = preserve_diagonal

    def _block_sizes(self, model) -> dict:
        R    = model.rank
        D_in = model.input_size
        D_out= model.wo.weight.shape[0] if model.wo is not None else 0
        return {"m": R, "n": R, "in": D_in, "out": D_out}

    def _block_offsets(self, sizes: dict) -> dict:
        offsets, cur = {}, 0
        for b in self.BLOCKS:
            offsets[b] = cur
            cur += sizes[b]
        return offsets

    def _expand_var(self, v, size: int) -> torch.Tensor:
        if size == 0:
            return torch.zeros(0)
        if torch.is_tensor(v):
            v = v.flatten().float()
            assert v.numel() == size
            return v
        return torch.full((size,), float(v))

    def _project_psd(self, Sigma: torch.Tensor):
        S = 0.5 * (Sigma + Sigma.T)
        d_orig = torch.diag(S).clone()
        evals, evecs = torch.linalg.eigh(S)
        evals_clipped = evals.clamp_min(self.psd_tol)
        S_psd = (evecs * evals_clipped) @ evecs.T
        if self.preserve_diagonal:
            d_new  = torch.diag(S_psd).clamp_min(self.psd_tol)
            scale  = torch.sqrt(d_orig.clamp_min(0.0) / d_new)
            S_psd  = S_psd * scale.unsqueeze(0) * scale.unsqueeze(1)
        S_psd = 0.5 * (S_psd + S_psd.T)
        return S_psd, evals.min().item()

    def _maybe_project(self, Sigma: torch.Tensor) -> torch.Tensor:
        evals  = torch.linalg.eigvalsh(Sigma)
        min_ev = float(evals.min())
        if min_ev >= -self.psd_tol:
            return Sigma
        if not self.auto_psd:
            raise ValueError(
                f"Sigma is not PSD (min eigenvalue = {min_ev:.3e}). "
                "Pass auto_psd=True to project onto the PSD cone."
            )
        Sigma_psd, _ = self._project_psd(Sigma)
        delta      = Sigma_psd - Sigma
        rel_change = (
            torch.linalg.matrix_norm(delta, ord="fro")
            / torch.linalg.matrix_norm(Sigma, ord="fro").clamp_min(1e-12)
        ).item()
        msg = (
            f"Sigma was not PSD (min eigval = {min_ev:.3e}); "
            f"projected to nearest PSD (Frobenius rel. change = {rel_change:.3e})."
        )
        if rel_change > self.psd_warn_threshold:
            msg += (
                " Correction is large — consider reducing overlap magnitudes "
                "or increasing the relevant variances."
            )
        warnings.warn(msg, stacklevel=3)
        return Sigma_psd

    def build_sigma(self, model) -> torch.Tensor:
        sizes   = self._block_sizes(model)
        offsets = self._block_offsets(sizes)
        P       = sum(sizes.values())

        if self.sigma is not None:
            assert self.sigma.shape == (P, P)
            return self._maybe_project(self.sigma.float().clone())

        diag = torch.cat([
            self._expand_var(self.var_m,   sizes["m"]),
            self._expand_var(self.var_n,   sizes["n"]),
            self._expand_var(self.var_in,  sizes["in"]),
            self._expand_var(self.var_out, sizes["out"]),
        ])
        Sigma = torch.diag(diag)

        for ((ba, ia), (bb, ib)), cov in self.overlaps.items():
            assert ba in self.BLOCKS and bb in self.BLOCKS
            assert 0 <= ia < sizes[ba]
            assert 0 <= ib < sizes[bb]
            a = offsets[ba] + ia
            b = offsets[bb] + ib
            Sigma[a, b] = cov
            Sigma[b, a] = cov

        return self._maybe_project(Sigma)

    def _sample_loadings(self, Sigma, N, device, generator=None):
        P       = Sigma.shape[0]
        Sigma   = Sigma.to(device)
        Sigma_j = Sigma + self.jitter * torch.eye(P, device=device, dtype=Sigma.dtype)
        try:
            L = torch.linalg.cholesky(Sigma_j)
        except RuntimeError as e:
            warnings.warn(f"Cholesky failed ({e}); falling back to eigendecomposition.")
            evals, evecs = torch.linalg.eigh(Sigma_j)
            evals = evals.clamp_min(0.0)
            L = evecs * evals.sqrt().unsqueeze(0)
        z = torch.randn(N, P, device=device, dtype=Sigma.dtype, generator=generator)
        return z @ L.T

    @torch.no_grad()
    def apply(self, model: nn.Module, seed: Optional[int] = None) -> torch.Tensor:
        Sigma   = self.build_sigma(model)
        sizes   = self._block_sizes(model)
        offsets = self._block_offsets(sizes)
        N       = model.hidden_size
        device  = model.device

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

        X = self._sample_loadings(Sigma, N, device, generator=gen)

        def slc(b):
            return slice(offsets[b], offsets[b] + sizes[b])

        model.m.copy_(X[:, slc("m")])
        model.n.copy_(X[:, slc("n")])

        if sizes["in"] > 0 and model.wi is not None:
            model.wi.weight.copy_(X[:, slc("in")])
            if self.zero_biases:
                model.wi.bias.zero_()

        if sizes["out"] > 0 and model.wo is not None:
            model.wo.weight.copy_(X[:, slc("out")].T)
            if self.zero_biases:
                model.wo.bias.zero_()

        return Sigma

    def validate(self, model, raise_on_fail: bool = True) -> tuple[bool, float]:
        old = self.auto_psd
        self.auto_psd = False
        try:
            try:
                Sigma  = self.build_sigma(model)
                min_ev = float(torch.linalg.eigvalsh(Sigma).min())
                return True, min_ev
            except ValueError:
                if raise_on_fail:
                    raise
                self.auto_psd = True
                Sigma  = self.build_sigma(model)
                min_ev = float(torch.linalg.eigvalsh(Sigma).min())
                return False, min_ev
        finally:
            self.auto_psd = old


# ---------------------------------------------------------------------------
# Mixture-of-Gaussians population initializer
# ---------------------------------------------------------------------------

BLOCKS = ("m", "n", "in", "out")


class FullMixturePopulationInitializer:
    """
    Mixture-of-Gaussians initializer for low-rank RNN loadings.

    Each neuron is assigned to population k and sampled from N(mu_k, Sigma_k).
    """

    def __init__(
        self,
        pop_means,
        pop_covs,
        pop_probs=None,
        jitter: float = 1e-6,
        zero_biases: bool = True,
        auto_psd: bool = True,
        psd_tol: float = 1e-8,
    ):
        assert len(pop_means) == len(pop_covs)
        self.K         = len(pop_means)
        self.pop_means = [m.flatten().float() for m in pop_means]
        self.pop_covs  = [c.float() for c in pop_covs]
        self.pop_probs = (
            pop_probs.float() if pop_probs is not None
            else torch.ones(self.K) / self.K
        )
        self.pop_probs = self.pop_probs / self.pop_probs.sum()
        self.jitter    = jitter
        self.zero_biases = zero_biases
        self.auto_psd  = auto_psd
        self.psd_tol   = psd_tol
        P = self.pop_means[0].numel()
        for k, (mu, S) in enumerate(zip(self.pop_means, self.pop_covs)):
            assert mu.numel() == P
            assert S.shape == (P, P)
        self.P = P

    @staticmethod
    def block_sizes(model) -> dict:
        return {
            "m":   model.rank,
            "n":   model.rank,
            "in":  model.input_size,
            "out": model.wo.weight.shape[0] if model.wo is not None else 0,
        }

    @staticmethod
    def block_offsets(sizes: dict) -> dict:
        offsets, cur = {}, 0
        for b in BLOCKS:
            offsets[b] = cur
            cur += sizes[b]
        return offsets

    @staticmethod
    def loading_dim(sizes: dict) -> int:
        return sum(sizes.values())

    @staticmethod
    def block_slice(block: str, sizes: dict) -> slice:
        offsets = FullMixturePopulationInitializer.block_offsets(sizes)
        return slice(offsets[block], offsets[block] + sizes[block])

    def expected_dim(self, model) -> int:
        return self.loading_dim(self.block_sizes(model))

    def _project_psd(self, S: torch.Tensor) -> torch.Tensor:
        S = 0.5 * (S + S.T)
        evals, evecs = torch.linalg.eigh(S)
        evals = evals.clamp_min(self.psd_tol)
        return (evecs * evals) @ evecs.T

    def _maybe_psd(self, S: torch.Tensor, k: int) -> torch.Tensor:
        S = 0.5 * (S + S.T)
        evals = torch.linalg.eigvalsh(S)
        if float(evals.min()) >= -self.psd_tol:
            return S
        if not self.auto_psd:
            raise ValueError(
                f"Sigma[{k}] is not PSD. Min eigenvalue = {float(evals.min()):.3e}"
            )
        return self._project_psd(S)

    @torch.no_grad()
    def apply(self, model, seed: Optional[int] = None):
        device    = model.device
        sizes     = self.block_sizes(model)
        N         = model.hidden_size
        expected_P = self.expected_dim(model)
        assert self.P == expected_P, f"Initializer loading dim {self.P} != model loading dim {expected_P}"

        gen = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None else None
        )

        pop_ids = torch.multinomial(
            self.pop_probs.to(device), N, replacement=True, generator=gen
        )

        V   = torch.zeros(N, self.P, device=device)
        eye = torch.eye(self.P, device=device)

        for k in range(self.K):
            idx = pop_ids == k
            n_k = int(idx.sum())
            if n_k == 0:
                continue
            mu = self.pop_means[k].to(device)
            S  = self._maybe_psd(self.pop_covs[k].to(device), k)
            L  = torch.linalg.cholesky(S + self.jitter * eye)
            z  = torch.randn(n_k, self.P, device=device, generator=gen)
            V[idx] = mu + z @ L.T

        sl_m   = self.block_slice("m",   sizes)
        sl_n   = self.block_slice("n",   sizes)
        sl_in  = self.block_slice("in",  sizes)
        sl_out = self.block_slice("out", sizes)

        model.m.copy_(V[:, sl_m])
        model.n.copy_(V[:, sl_n])

        if sizes["in"] > 0 and model.wi is not None:
            model.wi.weight.copy_(V[:, sl_in])
            if self.zero_biases and model.wi.bias is not None:
                model.wi.bias.zero_()

        if sizes["out"] > 0 and model.wo is not None:
            model.wo.weight.copy_(V[:, sl_out].T)
            if self.zero_biases and model.wo.bias is not None:
                model.wo.bias.zero_()

        return pop_ids


# ---------------------------------------------------------------------------
# Correlation-spec covariance builder
# ---------------------------------------------------------------------------

def population_cov_from_correlations(
    spec: dict,
    sizes: dict,
    var_m: float = 1.0,
    var_n: float = 1.0,
    var_in: float = 1.0,
    var_out: float = 1.0,
):
    """
    Build one population covariance matrix from a correlation spec.

    spec maps (block_a, idx_a, block_b, idx_b) -> rho, where block is one of
    {"m", "n", "in", "out"}.

    Returns (mean, Sigma).
    """
    R    = sizes["m"]
    D_in = sizes["in"]
    D_out= sizes["out"]

    offsets = {"m": 0, "n": R, "in": 2 * R, "out": 2 * R + D_in}
    block_size = {"m": R, "n": R, "in": D_in, "out": D_out}
    block_var  = {"m": var_m, "n": var_n, "in": var_in, "out": var_out}

    def global_idx(block, idx):
        assert 0 <= idx < block_size[block]
        return offsets[block] + idx

    diag = torch.cat([
        torch.full((R,),    float(var_m)),
        torch.full((R,),    float(var_n)),
        torch.full((D_in,), float(var_in)),
        torch.full((D_out,),float(var_out)),
    ])
    P     = diag.numel()
    Sigma = torch.diag(diag)

    for (ba, ia, bb, ib), rho in spec.items():
        assert -1.0 <= rho <= 1.0
        ga  = global_idx(ba, ia)
        gb  = global_idx(bb, ib)
        cov = rho * block_var[ba]**0.5 * block_var[bb]**0.5
        Sigma[ga, gb] = cov
        Sigma[gb, ga] = cov

    return torch.zeros(P), Sigma


# ---------------------------------------------------------------------------
# NeuroFlame dual-EI network — portable minimal re-implementation
# ---------------------------------------------------------------------------

class EISTPModel(nn.Module):
    """Minimal, self-contained port of the NeuroFlame dual EI network
    (`conf/train_dual_EI.yml` + `src/network.py`) that consistently pushes the
    decision wells into the lower κ-plane. Independent of the NeuroFlame repo.

    Essential mechanism reproduced (everything else in NeuroFlame is dropped):
      - 2 populations E,I with Na = round(N·frac); sparse *binary* connectivity
        C[i,j] (prob K/N_pre); Dale block strengths `Jab` balanced by 1/√K.
      - relu rates, **two timescales**: a synaptic filter (`tau_syn`) on the
        recurrent current, then a rate filter (`tau`) on the activation.
      - **Markram STP on E→E** (`u,x` per presynaptic-E unit; output u·x·r).
      - trained rank-R low-rank `m,n` on E (LowRankModel notation) that **multiplicatively**
        modulates the STP E→E weight:  W_EE = J_STP·gain·(C_EE/√K)·(1 + n mᵀ / N_E), clamped
        ≥0 (Dale). `n` = output/readout direction, `m` = presynaptic selection. Only trained set.
      - readout κ = rates_E @ n / N_E  (along `n`, same as LowRankModel.get_readout).

    Inputs are **linear** (NeuroFlame dual task `dualStim` = strength·pattern, NOT
    cosine): a fixed random E-pattern per channel (`wi`, the "odors"), injected to E
    only, with balanced-network scaling external_current = gain·√K·M0·(Ja0 + Wi·code),
    Ja0 baseline to E and I. `forward` takes a low-dim code (B, T, input_size) like
    `LowRankModel` (or pre-built neuron-space currents with ff_is_current=True).

    Notation aligned with LowRankModel: `m`, `n` (low-rank), `wi` (inputs), `gain`,
    `noise`, `nonlinearity`, `hidden_size`, `n_exc`, `exp_alpha` (rate τ),
    `exp_alpha_rec` (synaptic τ_syn); plus `forward`, `get_readout`, `update_dynamics`.
    """

    def __init__(self, n_neuron=2000, frac=(0.75, 0.25), K=250.0, rank=2,
                 Jab=(1.0, -1.5, 1.0, -1.0), gain=1.0,
                 tau=(0.4, 0.2), tau_syn=(0.2, 0.1), dt=0.02,
                 stp_use=0.05, stp_tau_fac=0.5, stp_tau_rec=0.2, j_stp=1.0,
                 lr_ini=1.0, lr_ueqv=True, lr_scale="N", clamp=True, noise=0.0, init_noise=1.0,
                 r_max=None, input_size=8, Ja0=(2.0, 1.0), M0=1.0, var_ff=(0.0, 0.0),
                 train_inputs=False, nonlinearity="relu", device="cpu", seed=None):
        super().__init__()
        self.device = torch.device(device)
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        self.n_pop = 2
        ne = int(round(n_neuron * frac[0]))
        ni = int(n_neuron - ne)
        self.n_exc, self.n_inh = ne, ni
        self.hidden_size = ne + ni
        self.rank = int(rank)
        self.K = float(K)
        self.dt = float(dt)
        self.clamp = bool(clamp)
        self.noise = float(noise)
        self.init_noise = float(init_noise)
        self.r_max = None if r_max is None else float(r_max)   # rate cap (anti-runaway)
        self.j_stp = float(j_stp)
        self.input_size = int(input_size)
        self.register_buffer("gain", torch.tensor(float(gain)))

        sl = [slice(0, ne), slice(ne, ne + ni)]
        self.slices = sl
        Jab = torch.tensor(Jab, dtype=torch.float32).reshape(2, 2)

        # --- sparse binary connectivity blocks C[post, pre] (prob K/N_pre) ---
        Na = [ne, ni]
        def conn(n_post, n_pre):
            return (torch.rand(n_post, n_pre, generator=g) <= (self.K / n_pre)).float()
        C = [[conn(Na[i], Na[j]) for j in range(2)] for i in range(2)]

        # --- static recurrent weights W[post, pre] (Dale, balanced 1/√K) ---
        # E→E carried by the STP/low-rank path → left out of the static matrix.
        W = torch.zeros(self.hidden_size, self.hidden_size)
        for i in range(2):
            for j in range(2):
                if i == 0 and j == 0:
                    continue
                W[sl[i], sl[j]] = float(gain) * Jab[i, j] / (self.K ** 0.5) * C[i][j]
        self.register_buffer("W_static", W.to(self.device))
        # base (unsigned) E→E connectivity for the STP path, balanced 1/√K
        self.register_buffer("C_EE", (C[0][0] / (self.K ** 0.5)).to(self.device))

        # --- trained rank-R low-rank on E (aligned with LowRankModel m,n) ---
        # n = recurrent OUTPUT + readout direction; m = presynaptic selection direction
        # (note: roles of m/n are swapped vs the vanilla class, since NeuroFlame reads out
        # along the output direction; `n` stays the readout in both, matching the tools).
        self.n = nn.Parameter(lr_ini * torch.randn(ne, self.rank, generator=g).to(self.device))
        m0 = self.n.detach().clone() if lr_ueqv else \
            lr_ini * torch.randn(ne, self.rank, generator=g).to(self.device)
        self.m = nn.Parameter(m0)
        # low-rank normalisation divisor for (n@mᵀ)/lr_scale (NeuroFlame train_scale):
        #   'N'/'all' → N_E (weak modulation);  'sqrtK'/'sparse' → √K (≈68× stronger);
        #   'sqrtN'/'dense' → √N_E.  A numeric value is used verbatim.
        if isinstance(lr_scale, str):
            key = lr_scale.lower()
            self.lr_scale = (float(self.n_exc)   if key in ("n", "all", "ne")     else
                             (self.K ** 0.5)     if key in ("sqrtk", "sparse")    else
                             (self.n_exc ** 0.5) if key in ("sqrtn", "dense")     else
                             float(self.n_exc))
        else:
            self.lr_scale = float(lr_scale)
        self.lr_scale_kind = lr_scale

        # --- two-timescale integration constants (per-neuron) ---
        # exp_alpha = rate filter (τ); exp_alpha_rec = synaptic/recurrent filter (τ_syn)
        self.tau = tuple(float(t) for t in tau)
        self.tau_syn = tuple(float(t) for t in tau_syn)
        self.stp_tau_fac = float(stp_tau_fac)
        self.stp_tau_rec = float(stp_tau_rec)
        exp_alpha = torch.ones(self.hidden_size); exp_alpha_rec = torch.ones(self.hidden_size)
        for i in range(2):
            exp_alpha[sl[i]]     = float(torch.tensor(-self.dt / tau[i]).exp())
            exp_alpha_rec[sl[i]] = float(torch.tensor(-self.dt / tau_syn[i]).exp())
        self.register_buffer("exp_alpha", exp_alpha.to(self.device))
        self.register_buffer("exp_alpha_rec", exp_alpha_rec.to(self.device))

        # --- Markram STP constants (E presynaptic) ---
        self.stp_use = float(stp_use)
        self.stp_dt_fac = self.dt / stp_tau_fac if stp_tau_fac > 0 else 0.0
        self.stp_dt_rec = self.dt / stp_tau_rec if stp_tau_rec > 0 else 0.0

        _nl = {"relu": torch.relu, "tanh": torch.tanh,
               "softplus": torch.nn.functional.softplus}
        self.nonlinearity = _nl[nonlinearity]
        self.nonlinearity_str = nonlinearity

        # --- LINEAR feedforward inputs (NeuroFlame dual-task style) ----------
        # Stimuli are linear: a fixed random E-pattern per channel (the "odors"),
        # injected to E only, amplitude carried by the input code. Balanced-network
        # scaling: external current = gain·√K·M0·(Ja0 + Wi·code), Ja0 to E and I.
        self.M0 = float(M0)
        self.bscale = float(gain) * (self.K ** 0.5) * self.M0   # balanced input scale
        # wi: input_size → E patterns (std-1 randn = "odors"); fixed unless train_inputs
        self.wi = nn.Linear(self.input_size, ne, bias=False, device=self.device)
        with torch.no_grad():
            self.wi.weight.copy_(torch.randn(ne, self.input_size, generator=g).to(self.device))
        if not train_inputs:
            self.wi.weight.requires_grad_(False)
        # constant balanced baseline current Ja0 (E and I)
        base = torch.empty(self.hidden_size)
        base[sl[0]] = self.bscale * float(Ja0[0])
        base[sl[1]] = self.bscale * float(Ja0[1])
        self.register_buffer("ext_base", base.to(self.device))
        self.register_buffer("var_ff", torch.tensor([float(var_ff[0]), float(var_ff[1])]))

    def W_EE_eff(self):
        """STP E→E weight modulated by the trained low-rank, Dale-clamped (≥0).

        In [post, pre] convention the weight is C·(1 + n[post]·m[pre]/N_E) =
        C·(1 + (n@mᵀ)/N_E): n is the output (=readout) direction, m the presynaptic
        selection direction.
        """
        W = self.gain * self.j_stp * self.C_EE * (1.0 + (self.n @ self.m.T) / self.lr_scale)
        if self.clamp:
            W = W.clamp_min(0.0)
        return W

    def init_stp(self, B):
        u = self.stp_use * torch.ones(B, self.n_exc, device=self.device)
        x = torch.ones(B, self.n_exc, device=self.device)
        return u, x

    def stp_step(self, rE, u, x):
        """Markram STP, exactly src/plasticity.markram_stp: depress x with old u,x;
        facilitate u with old u; return (u_new·x_new)·rE. Differentiable through u,x
        (NeuroFlame does NOT detach the STP state during training)."""
        x_new = x + (1.0 - x) * self.stp_dt_rec - self.dt * u * x * rE
        u_new = u + self.stp_dt_fac * (self.stp_use - u) + self.dt * self.stp_use * (1.0 - u) * rE
        return (u_new * x_new) * rE, u_new, x_new

    def get_readout(self, rates, rec_inputs=None):
        return rates[:, :self.n_exc] @ self.n / self.n_exc      # κ = rates_E·n/N_E

    def external_current(self, ff_code):
        """Low-dim input code (B, T, input_size) → per-neuron external current (B, T, N).
        Balanced scaling: gain·√K·M0·(Ja0 + Wi·code); stimuli to E only, Ja0 to E and I;
        optional per-population ff noise (var_ff)."""
        B, T = ff_code.shape[0], ff_code.shape[1]
        cur = self.ext_base.view(1, 1, -1).expand(B, T, self.hidden_size).clone()
        cur[..., :self.n_exc] = cur[..., :self.n_exc] + self.bscale * self.wi(ff_code)
        if torch.any(self.var_ff > 0):
            noise = torch.randn(B, T, self.hidden_size, device=self.device)
            noise[..., :self.n_exc] *= self.var_ff[0]
            noise[..., self.n_exc:] *= self.var_ff[1]
            cur = cur + self.bscale * noise
        return cur

    def update_dynamics(self, ff_t, rates, syn, u, x, W_ee):
        ne = self.n_exc
        # recurrent current: static (E→I,I→E,I→I) + STP-modulated E→E
        hidden = rates @ self.W_static.T
        gated, u, x = self.stp_step(rates[:, :ne], u, x)        # u·x·rE
        hidden = hidden.clone()
        hidden[:, :ne] = hidden[:, :ne] + gated @ W_ee.T        # E←E
        # synaptic filter, then rate filter through relu
        syn  = syn * self.exp_alpha_rec + hidden * (1.0 - self.exp_alpha_rec)
        nl   = self.nonlinearity(ff_t + syn)
        rates = rates * self.exp_alpha + nl * (1.0 - self.exp_alpha)
        if self.r_max is not None:                  # cap rates → forward can't run away to inf
            rates = rates.clamp(max=self.r_max)
        return rates, syn, u, x

    def forward(self, ff_inputs, targets=None, ret_rates=False, ff_is_current=False):
        """ff_inputs: (B, T, input_size) low-dim input code (projected to E via Wi with
        balanced scaling) — vanilla syntax (`targets` accepted & ignored, matching
        LowRankModel). Pass ff_is_current=True to feed pre-built neuron-space currents
        (B, T, N) directly (e.g. for flow/grid injection). Returns κ (B, T, rank)."""
        ext = ff_inputs if ff_is_current else self.external_current(ff_inputs)
        B, T = ext.shape[0], ext.shape[1]
        N = self.hidden_size
        # NeuroFlame initRates: random recurrent kick, rates₀ = relu(ff₀ + kick)
        syn   = self.init_noise * torch.randn(B, N, device=self.device)
        rates = self.nonlinearity(ext[:, 0] + syn)
        u, x  = self.init_stp(B)
        W_ee  = self.W_EE_eff()                                  # constant across trial
        kappa, rate_hist = [], []
        for t in range(T):
            if self.noise:
                rates = rates + self.noise * torch.randn(B, N, device=self.device)
            rates, syn, u, x = self.update_dynamics(ext[:, t], rates, syn, u, x, W_ee)
            kappa.append(self.get_readout(rates))
            if ret_rates:
                rate_hist.append(rates)
        kappa = torch.stack(kappa, dim=1)
        if ret_rates:
            return kappa, torch.stack(rate_hist, dim=1), None   # (κ, rates, _) like LowRankModel
        return kappa
