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
        nonlinearity="tanh",
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.rank        = rank
        self.gain        = torch.tensor(float(gain), device=self.device)

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

        _nl = {"tanh": torch.tanh, "relu": torch.relu,
               "softplus": torch.nn.functional.softplus,
               "erf":      torch.erf,
               "elu":      torch.nn.functional.elu,
               "lif":      lambda x: (1.0 + torch.erf(x / 1.4142135623730951)) / 2.0}
        self.nonlinearity     = _nl[nonlinearity]
        self.nonlinearity_str = nonlinearity
        self.register_buffer("alpha_rec",     torch.tensor(float(alpha_rec)))
        self.register_buffer("exp_alpha_rec", torch.exp(-torch.tensor(float(alpha_rec))))
        self.register_buffer("alpha",         torch.tensor(float(alpha)))
        self.register_buffer("exp_alpha",     torch.exp(-torch.tensor(float(alpha))))

        # Fixed random weights — not saved in state_dict (persistent=False) so old
        # checkpoints load without issue; re-created from scratch at init time.
        # Projected to be orthogonal to the initial m and n so that n^T W_fixed = 0
        # and W_fixed^T m = 0, keeping the κ-plane analysis approximately valid.
        self.use_fixed_weights = use_fixed_weights
        if use_fixed_weights:
            w_fixed = torch.randn(hidden_size, hidden_size, device=self.device)
            # Remove components along n (rows) and m (columns)
            n_hat = self.n / self.n.norm(dim=0, keepdim=True).clamp_min(1e-12)  # (N, rank)
            m_hat = self.m / self.m.norm(dim=0, keepdim=True).clamp_min(1e-12)  # (N, rank)
            w_fixed = w_fixed - n_hat @ (n_hat.T @ w_fixed)        # n^T W_fixed = 0
            w_fixed = w_fixed - (w_fixed @ m_hat) @ m_hat.T        # W_fixed^T m = 0
            w_fixed *= fixed_weight_scale / (hidden_size ** 0.5)
            self.register_buffer("w_fixed", w_fixed.detach(), persistent=False)
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
        phi         = self.nonlinearity(self.gain * (input_drive + rec_inputs))
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
