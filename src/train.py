from __future__ import annotations

import copy
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from .tasks import TaskTiming


def train_val_split(X, y, batch_size=32, frac=0.8):
    dataset = TensorDataset(X.float(), y.float())
    n_total = len(dataset)
    n_train = int(frac * n_total)
    n_val   = n_total - n_train
    generator = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False)
    return train_loader, val_loader


class Optimization:
    """Small PyTorch train/validation loop with optional parameter freezing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any | None = None,
        grad_clip_norm: float | None = None,
        num_epochs: int = 100,
        warmup_epochs: int = 25,
        stop_loss: float = 0.005,
        max_val_loss: float = 100.0,
        # Freeze specific columns of m, n.
        zero_low_rank_grad: Any | None = None,   # backward-compat alias
        freeze_low_rank_cols: Any | None = None,
        # Freeze specific input dimensions in wi.weight.
        freeze_input_dims: Any | None = None,
        device: str | torch.device | None = None,
        keep_best: bool = False,
        verbose: bool = True,
        regularizer: Any | None = None,
        hebb_lr: float = 0.0,
        kappa1_clamp: float | None = None,
        kappa_gain_target: float | None = None,
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.regularizer   = regularizer   # callable(model) -> scalar tensor, added to train loss
        self.hebb_lr       = hebb_lr       # three-factor Hebbian lr for reward input (0 = disabled)
        self.kappa1_clamp  = kappa1_clamp  # hard cap on decision self-gain g·λ₁ after each step (None=off)
        self.kappa_gain_target = kappa_gain_target  # pin ALL modes' g·λ to this value each step (None=off)
        self.num_epochs    = num_epochs
        self.warmup_epochs = warmup_epochs
        self.stop_loss     = stop_loss
        self.max_val_loss  = max_val_loss
        self.scheduler     = scheduler
        self.grad_clip_norm= grad_clip_norm
        self.keep_best     = keep_best
        self.verbose       = verbose

        self.device = torch.device(device) if device is not None else self._infer_device()
        self.model.to(self.device)
        self.model.device = self.device

        self.train_losses:  list[float] = []
        self.val_losses:    list[float] = []
        self.learning_rates:list[float] = []
        self.learning_rate  = self.optimizer.param_groups[0]["lr"]

        self.best_val_loss   = float("inf")
        self.best_state_dict: dict | None = None

        if freeze_low_rank_cols is None:
            freeze_low_rank_cols = zero_low_rank_grad
        self.freeze_low_rank_cols = self._normalize_low_rank_cols(freeze_low_rank_cols)
        self.freeze_input_dims    = freeze_input_dims

        self._frozen_m  = None
        self._frozen_n  = None
        self._frozen_wi = None

        if self.freeze_low_rank_cols is not None:
            if not (hasattr(model, "m") and hasattr(model, "n")):
                raise AttributeError("freeze_low_rank_cols requires model.m and model.n.")
            self._frozen_m = model.m.detach().clone()
            self._frozen_n = model.n.detach().clone()

        if self.freeze_input_dims is not None:
            if not (hasattr(model, "wi") and model.wi is not None):
                raise AttributeError("freeze_input_dims requires model.wi.")
            self._frozen_wi = model.wi.weight.detach().clone()

    # ------------------------------------------------------------------

    def _infer_device(self) -> torch.device:
        if hasattr(self.model, "device") and isinstance(self.model.device, torch.device):
            return self.model.device
        return next(self.model.parameters()).device

    @staticmethod
    def _normalize_low_rank_cols(cols):
        if cols is None:    return None
        if cols == "all":   return "all"
        if isinstance(cols, int): return [cols]
        if torch.is_tensor(cols): return cols.detach().cpu().long().tolist()
        return list(cols)

    # ------------------------------------------------------------------

    def _zero_low_rank_grads(self):
        if self.freeze_low_rank_cols is None:
            return
        for param in (self.model.m, self.model.n):
            if param.grad is None:
                continue
            if self.freeze_low_rank_cols == "all":
                param.grad.zero_()
            else:
                param.grad[:, self.freeze_low_rank_cols] = 0.0

    def _zero_input_grads(self):
        if self.freeze_input_dims is None:
            return
        if self.model.wi.weight.grad is not None:
            self.model.wi.weight.grad[:, self.freeze_input_dims] = 0.0

    def _restore_frozen_weights(self):
        if self.freeze_low_rank_cols is not None:
            with torch.no_grad():
                if self.freeze_low_rank_cols == "all":
                    self.model.m.copy_(self._frozen_m)
                    self.model.n.copy_(self._frozen_n)
                else:
                    cols = self.freeze_low_rank_cols
                    self.model.m[:, cols] = self._frozen_m[:, cols]
                    self.model.n[:, cols] = self._frozen_n[:, cols]
        if self.freeze_input_dims is not None:
            with torch.no_grad():
                self.model.wi.weight[:, self.freeze_input_dims] = (
                    self._frozen_wi[:, self.freeze_input_dims]
                )

    def _clamp_kappa1_gain(self):
        """Hard constraint (vs. the soft kappa1 penalty): after each step, if the
        decision self-gain g·λ₁ = gain·n₁ᵀm₁/N exceeds `kappa1_clamp`, rescale the
        rank-1 columns m₁, n₁ so g·λ₁ == kappa1_clamp exactly. Magnitude is pinned;
        the *direction* of m₁/n₁ is free to keep training (so go/nogo stays learnable),
        but the decision mode can never grow an autonomous bistable well (no ring).
        One-sided: subcritical (g·λ₁ ≤ clamp, incl. ≤0) is left untouched."""
        if self.kappa1_clamp is None:
            return
        m, n = self.model.m, self.model.n
        if m.shape[1] < 2:
            return
        gain = float(self.model.gain) if torch.is_tensor(self.model.gain) else float(self.model.gain)
        N = m.shape[0]
        with torch.no_grad():
            c = gain * torch.dot(n[:, 1], m[:, 1]) / N          # current g·λ₁
            if c.item() > self.kappa1_clamp:
                s = (self.kappa1_clamp / c).sqrt()               # scale both cols → g·λ₁ = clamp
                m[:, 1].mul_(s)
                n[:, 1].mul_(s)

    def _pin_kappa_gains(self):
        """Criticality constraint: pin EVERY mode's self-gain g·λ_a = gain·n_aᵀm_a/N to a
        single target (two-sided rescale of m_a,n_a, up or down). With target=1 both the
        memory and decision modes sit at the pitchfork/marginal point → line-attractor
        (integrator) dynamics, consistent across modes. Off-diagonals stay free."""
        if self.kappa_gain_target is None:
            return
        m, n = self.model.m, self.model.n
        gain = float(self.model.gain) if torch.is_tensor(self.model.gain) else float(self.model.gain)
        N = m.shape[0]
        t = self.kappa_gain_target
        with torch.no_grad():
            for a in range(m.shape[1]):
                c = gain * torch.dot(n[:, a], m[:, a]) / N
                if c.item() > 1e-6:                              # same-sign as (positive) target
                    s = (t / c).sqrt()
                    m[:, a].mul_(s)
                    n[:, a].mul_(s)

    # ------------------------------------------------------------------

    def _hebb_update(self, y_pred: torch.Tensor, y: torch.Tensor, rates: torch.Tensor):
        """Three-factor Hebbian update for the reward input column of wi.

        Δwi[:, -1] += hebb_lr * mean( rates[t+1] | reward fires at t+1 )

        Reward fires at t+1 when target[t,-1]==1 AND readout[t,-1]>0.5
        (matching the teacher-forced reward logic in LowRankModel.forward).
        """
        # reward_mask[b, t] = True when reward fires at step t+1
        reward_mask = (y[:, :-1, -1] == 1) & (y_pred[:, :-1, -1].detach() > 0.5)  # (B, T-1)
        n_events = reward_mask.sum().clamp_min(1)
        rates_at_rwd = rates[:, 1:, :].detach()  # (B, T-1, N)
        hebb_delta = (rates_at_rwd * reward_mask.unsqueeze(-1).float()).sum(dim=(0, 1)) / n_events
        with torch.no_grad():
            self.model.wi.weight[:, -1] += self.hebb_lr * hebb_delta

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss, total_n, n_seen = 0.0, 0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        use_hebb = self.hebb_lr > 0 and train

        with context:
            for X, y in loader:
                n_seen += 1
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                if use_hebb:
                    y_pred, rates, _ = self.model(X, y, ret_rates=True)
                else:
                    y_pred = self.model(X, y)
                    rates  = None
                loss   = self.criterion(y_pred, y)

                # Skip a non-finite batch (don't corrupt weights / abort the run) rather than
                # returning NaN — lets training survive a transient instability.
                if not torch.isfinite(loss):
                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                    continue

                if train:
                    if self.regularizer is not None:
                        loss = loss + self.regularizer(self.model)
                    loss.backward()
                    # Block BPTT from touching the reward input (Hebbian-only)
                    if use_hebb and self.model.wi.weight.grad is not None:
                        self.model.wi.weight.grad[:, -1] = 0.0
                    self._zero_low_rank_grads()
                    self._zero_input_grads()
                    # skip the step if any gradient went non-finite (anti-runaway)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in self.model.parameters()):
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    if self.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self._restore_frozen_weights()
                    self._clamp_kappa1_gain()
                    self._pin_kappa_gains()
                    if use_hebb:
                        self._hebb_update(y_pred, y, rates)

                total_loss += loss.item() * X.size(0)
                total_n    += X.size(0)

        if n_seen == 0:
            raise ValueError("Dataloader produced zero examples.")   # genuinely empty loader
        if total_n == 0:
            return float("nan")   # every batch was non-finite (diverged) → fit() stops gracefully
        return total_loss / total_n

    def _step_scheduler(self, val_loss: float):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    # ------------------------------------------------------------------

    def fit(self) -> tuple[list[float], list[float], list[float]]:
        self.train_losses.clear()
        self.val_losses.clear()
        self.learning_rates.clear()
        self.best_val_loss   = float("inf")
        self.best_state_dict = None

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss   = self._run_epoch(self.val_loader,   train=False)

            if epoch >= self.warmup_epochs:
                self._step_scheduler(val_loss)

            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(self.learning_rate)

            if self.keep_best and math.isfinite(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss   = val_loss
                self.best_state_dict = copy.deepcopy(self.model.state_dict())

            if self.verbose and epoch % 5 == 0:
                print(
                    f"Epoch {epoch:03d}/{self.num_epochs} | "
                    f"lr: {self.learning_rate:.5f} | "
                    f"train: {train_loss:.4f} | val: {val_loss:.4f}",
                    flush=True,
                )

            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                print("Stopping: non-finite loss.")
                break
            if train_loss < self.stop_loss and val_loss < self.stop_loss:
                print(f"Stopping: losses below {self.stop_loss}.")
                break
            if val_loss > self.max_val_loss:
                print(f"Stopping: validation loss above {self.max_val_loss}.")
                break

        if self.keep_best and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            self._restore_frozen_weights()

        return self.train_losses, self.val_losses, self.learning_rates


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class MaskedMultiTargetLoss(nn.Module):
    def __init__(
        self,
        criterion: nn.Module | None = None,
        target_weight: float = 1.0,
        zero_weight: float = 1.0,
    ):
        super().__init__()
        self.criterion    = criterion or nn.MSELoss(reduction="none")
        self.target_weight= target_weight
        self.zero_weight  = zero_weight

    @staticmethod
    def masked_mean(loss, mask):
        mask = mask.to(dtype=loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward_channel(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        finite    = torch.isfinite(pred) & torch.isfinite(target)
        safe_tgt  = torch.where(finite, target, torch.zeros_like(target))
        safe_pred = torch.where(finite, pred,   torch.zeros_like(pred))
        zero_mask = finite & (safe_tgt == 0)
        tgt_mask  = finite & (safe_tgt.abs() == 1)
        zero_loss = self.masked_mean(self.criterion(safe_pred, torch.zeros_like(safe_pred)), zero_mask)
        tgt_loss  = self.masked_mean(self.criterion(safe_pred, safe_tgt), tgt_mask)
        return self.zero_weight * zero_loss + self.target_weight * tgt_loss

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for ch in range(y.shape[-1]):
            loss = loss + self.forward_channel(y_pred[..., ch].clone(), y[..., ch].clone())
        return loss


class MaskedGNGLoss(nn.Module):
    """
    GNG-stage loss. On the decision channel, hinge_gng selects the shape:
      - hinge_gng=True  → ASYMMETRIC one-sided hinges: go/lick → κ₁ ≥ go_hinge_thresh;
        nogo/no-lick → κ₁ ≤ nogo_hinge_thresh (−1 during the memory delay, 0 after the cue).
      - hinge_gng=False → PURE MSE toward the task targets (go→+1, nogo→−1 delay / 0 after cue),
        weighted by zero/target. No hinges.
    The one-sided nolick penalty (relu(κ₁)² over free windows, sample excluded) is a separate
    directional term, applied in both modes via nolick_weight.
    """

    def __init__(
        self,
        timing: TaskTiming,
        readout_index: int = -1,
        criterion: nn.Module | None = None,
        target_weight: float = 1.0,
        zero_weight: float = 1.0,
        go_hinge_thresh: float | None = None,
        nolick_weight: float = 0.0,
        hinge_gng: bool = False,
        nogo_hinge_thresh: float = -1.0,
        pin_decay_zeros: bool = False,
    ):
        super().__init__()
        self.timing            = timing
        self.readout_index     = readout_index
        self.criterion         = criterion or nn.MSELoss(reduction="none")
        self.target_weight     = target_weight
        self.zero_weight       = zero_weight
        self.go_hinge_thresh   = go_hinge_thresh
        self.nolick_weight     = nolick_weight
        self.hinge_gng         = hinge_gng
        self.nogo_hinge_thresh = nogo_hinge_thresh   # memory-window nogo hinge; 0 after cue
        self.pin_decay_zeros   = pin_decay_zeros     # windowed decay: zero targets → MSE-to-0 (pin), not one-sided

    @staticmethod
    def masked_mean(loss, mask):
        mask = mask.to(dtype=loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, T, C = y.shape
        device  = y.device
        dec     = self.readout_index % C
        t       = torch.arange(T, device=device)
        resp_mask = (t >= self.timing.n_stim_on[1].to(device))[None, :]

        loss = y_pred.sum() * 0.0
        for ch in range(C):
            pred   = y_pred[..., ch]
            target = y[..., ch]
            finite    = torch.isfinite(pred) & torch.isfinite(target)
            safe_tgt  = torch.where(finite, target, torch.zeros_like(target))
            safe_pred = torch.where(finite, pred,   torch.zeros_like(pred))

            if ch == dec:
                if self.hinge_gng:
                    # ASYMMETRIC one-sided decision hinge (fixes the go/nogo collapse of the
                    # symmetric-at-0 version): lick/go(+) → κ₁ ≥ go_hinge_thresh; no-lick/nogo(≤0)
                    # → κ₁ ≤ nogo_th, where nogo_th = nogo_hinge_thresh (−1) during the MEMORY
                    # delay (pre-cue) and 0 after the cue (resp_mask). A real ±gap separates go
                    # from nogo (kills the trivial "always ≤0" solution), while each side is still
                    # one-sided (a transient more extreme than its threshold is unpenalised).
                    th_go   = self.go_hinge_thresh if self.go_hinge_thresh is not None else 1.0
                    nogo_th = torch.where(resp_mask, torch.zeros_like(safe_pred),
                                          torch.full_like(safe_pred, self.nogo_hinge_thresh))
                    hinge_raw = torch.where(safe_tgt > 0,
                                            torch.relu(th_go - safe_pred) ** 2,
                                            torch.relu(safe_pred - nogo_th) ** 2)
                    # TWO separate terms, each averaged over its OWN window so neither dilutes the
                    # other: (1) go/nogo one-sided hinge over the post-sample decision windows;
                    # (2) pre-sample BASELINE pinned to 0 (MSE-to-0, symmetric), like the
                    # non-decision channels — NOT folded into the nogo hinge (which would drive κ₁≤−1).
                    bl_mask    = (t < self.timing.n_stim_on[0].to(device))[None, :]
                    if self.pin_decay_zeros:
                        # Windowed decay: post-baseline ZERO targets (the decay-to-0 window) are
                        # PINNED (MSE-to-0, symmetric), not the one-sided nogo hinge — "decay" means
                        # return to 0, not "anywhere ≤0". Its OWN masked_mean; the pre-sample
                        # baseline keeps its separate term below (no interference).
                        pin_mask   = finite & ~bl_mask & (safe_tgt == 0)
                        hinge_loss = (self.masked_mean(hinge_raw, finite & ~bl_mask & (safe_tgt != 0))
                                      + self.masked_mean(self.criterion(safe_pred, torch.zeros_like(safe_pred)), pin_mask))
                    else:
                        hinge_loss = self.masked_mean(hinge_raw, finite & ~bl_mask)
                    bl_loss    = self.masked_mean(safe_pred ** 2, finite & bl_mask)
                    loss = loss + self.target_weight * hinge_loss + self.zero_weight * bl_loss
                else:
                    # PURE MSE toward the decision targets (go→+1, nogo→−1 in the delay / 0 after
                    # cue, all set by the task), weighted by zero/target. hinge_gng=False ⇒ no
                    # hinges anywhere on the decision channel (go_hinge_thresh is ignored here).
                    zero_mask = finite & (safe_tgt == 0)
                    tgt_mask  = finite & (safe_tgt != 0)
                    zero_loss = self.masked_mean(self.criterion(safe_pred, torch.zeros_like(safe_pred)), zero_mask)
                    tgt_loss  = self.masked_mean(self.criterion(safe_pred, safe_tgt), tgt_mask)
                    loss = loss + self.zero_weight * zero_loss + self.target_weight * tgt_loss

                # One-sided "no-lick" penalty over the free (NaN-target) decision windows:
                # penalise lick (κ₁>0) only, leave κ₁<0 free. Opt-in via nolick_weight.
                if self.nolick_weight:
                    sample_mask = ((t >= self.timing.n_stim_on[0].to(device)) &
                                   (t <  self.timing.n_stim_off[0].to(device)))[None, :]  # no loss during sample
                    nolick_mask = torch.isfinite(pred) & ~torch.isfinite(target) & ~sample_mask
                    nolick = self.masked_mean(torch.relu(pred) ** 2, nolick_mask)
                    loss = loss + self.nolick_weight * nolick
            else:
                zero_mask = finite & (safe_tgt == 0)
                tgt_mask  = finite & (safe_tgt.abs() == 1)
                zero_loss = self.masked_mean(
                    self.criterion(safe_pred, torch.zeros_like(safe_pred)), zero_mask)
                tgt_loss  = self.masked_mean(self.criterion(safe_pred, safe_tgt), tgt_mask)
                loss = loss + self.zero_weight * zero_loss + self.target_weight * tgt_loss

        return loss


class MaskedMultiTargetDualLoss(nn.Module):
    """
    Per-channel masked MSE (like `MaskedMultiTargetLoss`) whose **decision channel**
    is additionally split by time window into separate DPA and GNG components, so the
    two task demands can be weighted and/or logged independently.

    Intended for the dual stage (needs 4-epoch dual timing). Channel layout assumed:
        - decision channel = `readout_index` (default -1): time-multiplexed GNG
          response (cue window) then DPA decision (post-test window).
        - every other channel (e.g. the memory channel 0) = auxiliary; trained with
          the standard zero/±1 masked loss at weight `aux_weight`.

    Time windows (match `_dual_accuracy` / `WeightedDualTaskLoss`):
        baseline : t < n_on[0]                 (decision target ≈ 0)
        gng      : n_on[2] <= t < n_on[3]       (cue response)
        dpa      : t >= n_off[3]                (test decision)
    NaN target entries outside these windows are masked out as usual.

    After each `forward`, the unweighted per-component values are stored in
    `self.last_components = {"dpa","gng","baseline","aux"}` for logging.
    """

    def __init__(
        self,
        timing: TaskTiming,
        readout_index: int = -1,
        criterion: nn.Module | None = None,
        dpa_weight:      float = 1.0,
        gng_weight:      float = 1.0,
        gng_go_weight:   float = 1.0,
        gng_nogo_weight: float = 1.0,
        aux_weight:      float = 1.0,
        bl_weight:       float = 1.0,
        target_weight:   float = 1.0,
        zero_weight:     float = 1.0,
        go_hinge_thresh: float | None = None,
        dpa_hinge_thresh: float | None = None,
        nolick_weight: float = 0.0,
        hinge_gng: bool = False,
        nogo_hinge_thresh: float = -1.0,
        nogo_push_memory: bool = False,
        pin_decay_zeros: bool = False,
    ):
        super().__init__()
        self.timing          = timing
        self.readout_index   = readout_index
        self.criterion       = criterion or nn.MSELoss(reduction="none")
        self.dpa_hinge_thresh = dpa_hinge_thresh
        self.nolick_weight   = nolick_weight
        self.hinge_gng       = hinge_gng
        self.nogo_hinge_thresh = nogo_hinge_thresh
        self.nogo_push_memory  = nogo_push_memory
        self.pin_decay_zeros = pin_decay_zeros   # windowed decay: zero targets → MSE-to-0 (pin), not one-sided/nonmatch
        self.dpa_weight      = dpa_weight
        self.gng_weight      = gng_weight
        self.gng_go_weight   = gng_go_weight
        self.gng_nogo_weight = gng_nogo_weight
        self.aux_weight      = aux_weight
        self.bl_weight       = bl_weight
        self.target_weight   = target_weight
        self.zero_weight     = zero_weight
        self.go_hinge_thresh = go_hinge_thresh
        self.last_components: dict[str, float] = {}

    @staticmethod
    def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _aux_channel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        finite    = torch.isfinite(pred) & torch.isfinite(target)
        safe_tgt  = torch.where(finite, target, torch.zeros_like(target))
        safe_pred = torch.where(finite, pred,   torch.zeros_like(pred))
        zero_mask = finite & (safe_tgt == 0)
        tgt_mask  = finite & (safe_tgt.abs() == 1)
        zero_loss = self.masked_mean(self.criterion(safe_pred, torch.zeros_like(safe_pred)), zero_mask)
        tgt_loss  = self.masked_mean(self.criterion(safe_pred, safe_tgt), tgt_mask)
        return self.zero_weight * zero_loss + self.target_weight * tgt_loss

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, T, C = y.shape
        device  = y.device
        dec     = self.readout_index % y_pred.shape[-1]   # normalise negative index

        pred_dec = y_pred[..., dec]
        tgt_dec  = y[..., dec]
        finite   = torch.isfinite(pred_dec) & torch.isfinite(tgt_dec)
        safe_t   = torch.where(finite, tgt_dec,  torch.zeros_like(tgt_dec))
        safe_p   = torch.where(finite, pred_dec, torch.zeros_like(pred_dec))

        t     = torch.arange(T, device=device)
        n_on  = self.timing.n_stim_on.to(device)
        n_off = self.timing.n_stim_off.to(device)

        bl_mask      = finite & (t < n_on[0])[None, :]
        # Start the gng window at the memory-delay onset (n_off[1]) so the ±1 go/nogo MEMORY
        # (held at [n_off[1], n_on[2]) while the sample is held at κ₀=±1, no cue input) is scored —
        # this is where the nogo push-down (κ₁≤−1) lands to lower the memory wells.
        gng_win_mask = finite & ((t >= n_off[1]) & (t < n_on[3]))[None, :]
        dpa_mask     = finite & (t >= n_off[3])[None, :]

        go_mask   = gng_win_mask & (safe_t > 0)   # target = +1 → go
        nogo_mask = gng_win_mask & (safe_t < 0)  # target ≤  0 → nogo
        nogo_resp_mask = gng_win_mask & (safe_t == 0)  # target ==  0 → nogo

        bl_loss = self.masked_mean(self.criterion(safe_p, torch.zeros_like(safe_p)), bl_mask)
        
        if self.hinge_gng:
            th_go  = self.go_hinge_thresh  if self.go_hinge_thresh  is not None else 1.0
            th_dpa = self.dpa_hinge_thresh if self.dpa_hinge_thresh is not None else th_go
            # go/nogo: ASYMMETRIC one-sided (fixes the go/nogo collapse) — go → κ₁≥go_hinge_thresh.
            go_loss   = self.masked_mean(torch.relu(th_go - safe_p) ** 2, go_mask)
            # nogo MEMORY (target −1 delay). Default (nogo_push_memory=False) = gentle one-sided
            # κ₁≤0: the memory is only forbidden from licking, so its no-lick location is left free to
            # EMERGE from the asymmetric-cue nogo-error pressure. nogo_push_memory=True FORCES κ₁≤−1
            # (relu(κ₁+th_go)²) — an explicit push-down that reads high nogo but drags the autonomous
            # memory wells back UP into the lick plane (it supplies an alternative nogo-error route),
            # so it defeats the emergent lowering. See ring_lowerplane_log §16.
            if self.nogo_push_memory:
                nogo_loss = self.masked_mean(torch.relu(safe_p + th_go) ** 2, nogo_mask)
            else:
                nogo_loss = self.masked_mean(torch.relu(safe_p) ** 2, nogo_mask)
            if self.pin_decay_zeros:
                # Windowed decay: the gng decay-to-0 window is PINNED (MSE-to-0), not one-sided —
                # "decay" means return to 0. (Pre-sample baseline has its own bl term; gng_win_mask
                # starts at n_off[1] so there is no overlap/interference.)
                nogo_resp_loss = self.masked_mean(self.criterion(safe_p, torch.zeros_like(safe_p)), nogo_resp_mask)
            else:
                nogo_resp_loss = self.masked_mean(torch.relu(safe_p) ** 2, nogo_resp_mask)

            # match/nonmatch: SYMMETRIC ±th margin — identical to the DPA-stage ThresholdLoss, so
            # this decision has the same shape (nonmatch ≤ −th) across DPA and Dual. ZERO targets
            # (the pairing decay-to-0 tail) are split OUT of the binary hinge — they used to fall
            # into the nonmatch branch (relu(p+th)² ⇒ trained as κ₁ ≤ −1, a hard shove down on
            # every trial-end). Pinned (MSE-to-0) under pin_decay_zeros, else gentle one-sided ≤0.
            dpa_tgt_mask  = dpa_mask & (safe_t != 0)
            dpa_zero_mask = dpa_mask & (safe_t == 0)
            dpa_raw   = torch.where(safe_t > 0, torch.relu(th_dpa - safe_p) ** 2,
                                    torch.relu(safe_p + th_dpa) ** 2)
            if self.pin_decay_zeros:
                dpa_zero_loss = self.masked_mean(self.criterion(safe_p, torch.zeros_like(safe_p)), dpa_zero_mask)
            else:
                dpa_zero_loss = self.masked_mean(torch.relu(safe_p) ** 2, dpa_zero_mask)
            dpa_loss  = self.masked_mean(dpa_raw, dpa_tgt_mask) + dpa_zero_loss
        else:
            # PURE MSE toward the decision targets (go→+1, nogo→0, match/nonmatch→±1).
            # hinge_gng=False ⇒ no hinges anywhere (go_hinge_thresh / dpa_hinge_thresh ignored).
            go_loss   = self.masked_mean(self.criterion(safe_p, safe_t), go_mask)
            nogo_loss = self.masked_mean(self.criterion(safe_p, safe_t), nogo_mask)

            nogo_resp_loss = self.masked_mean(self.criterion(safe_p, safe_t), nogo_resp_mask)

            dpa_loss  = self.masked_mean(self.criterion(safe_p, safe_t), dpa_mask)

        gng_loss  = self.gng_go_weight * go_loss + self.gng_nogo_weight * (nogo_loss + nogo_resp_loss)

        aux_loss = pred_dec.sum() * 0.0   # zero scalar carrying grad/device/dtype
        for ch in range(C):
            if ch == dec:
                continue
            aux_loss = aux_loss + self._aux_channel_loss(
                y_pred[..., ch].clone(), y[..., ch].clone()
            )

        # One-sided "no-lick" penalty over the free (NaN-target) decision windows
        # (sample, memory delay, gng-stim, cue, and the gng-response→DPA gap): penalise
        # lick (κ₁>0) only, leave κ₁<0 free. Directional pressure with no painted κ₁ value.
        nolick_loss = pred_dec.sum() * 0.0
        if self.nolick_weight:
            sample_mask = ((t >= n_on[0]) & (t < n_off[0]))[None, :]   # no loss during sample
            nolick_mask = torch.isfinite(pred_dec) & ~torch.isfinite(tgt_dec) & ~sample_mask & ~bl_mask
            nolick_loss = self.masked_mean(torch.relu(pred_dec) ** 2, nolick_mask)

        self.last_components = {
            "dpa":      float(dpa_loss.detach()),
            "gng":      float(gng_loss.detach()),
            "gng_go":   float(go_loss.detach()),
            "gng_nogo": float(nogo_loss.detach()),
            "baseline": float(bl_loss.detach()),
            "aux":      float(aux_loss.detach()),
            "nolick":   float(nolick_loss.detach()),
        }

        return (
            self.dpa_weight * dpa_loss
            + self.gng_weight * gng_loss
            + self.bl_weight  * bl_loss
            + self.aux_weight * aux_loss
            + self.nolick_weight * nolick_loss
        )


class ThresholdLoss(nn.Module):
    """
    Squared hinge loss: penalises predictions on the wrong side of a threshold,
    with zero gradient once the prediction is correctly above/below it.

    For target = +1 :  loss = relu(thresh - pred)²        (penalise if pred < thresh)
    For target = -1 :  loss = relu(thresh + pred)²        (penalise if pred > -thresh)
    For target =  0 :  loss = relu(|pred| - zero_thresh)² (penalise if |pred| > zero_thresh)
    NaN targets are masked out (no gradient).

    The ±1 targets keep a wide margin (`thresh`) so the decision saturates, but the ZERO
    targets (baselines, nogo "rest") get their OWN `zero_thresh` — default 0 ⇒ relu(|pred|)² =
    pred², i.e. MSE-to-0. This pins baselines to zero instead of leaving a free ±thresh dead-zone
    (with `zero_thresh = thresh` the deep-memory dynamics drift the baseline into a ±well for free).

    Works as a drop-in for MaskedMultiTargetLoss.
    Supports any number of target values as long as they are +1, -1, 0, or nan.
    """

    def __init__(
        self,
        thresh: float = 0.5,
        target_weight: float = 1.0,
        zero_weight: float = 1.0,
        squared: bool = True,
        zero_thresh: float = 0.0,
    ):
        super().__init__()
        self.thresh        = thresh
        self.target_weight = target_weight
        self.zero_weight   = zero_weight
        self.squared       = squared   # True: relu(...)² (default); False: linear margin relu(...)
        self.zero_thresh   = zero_thresh   # dead-zone half-width for ZERO targets (0 ⇒ MSE-to-0)

    @staticmethod
    def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward_channel(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        finite    = torch.isfinite(pred) & torch.isfinite(target)
        safe_tgt  = torch.where(finite, target, torch.zeros_like(target))
        safe_pred = torch.where(finite, pred,   torch.zeros_like(pred))

        pos_mask  = finite & (safe_tgt > 0)   # target = +1
        neg_mask  = finite & (safe_tgt < 0)   # target = -1
        zero_mask = finite & (safe_tgt == 0)  # target =  0

        p    = 2 if self.squared else 1
        pos_loss  = self.masked_mean(
            torch.relu(self.thresh - safe_pred) ** p, pos_mask)
        neg_loss  = self.masked_mean(
            torch.relu(self.thresh + safe_pred) ** p, neg_mask)
        zero_loss = self.masked_mean(
            torch.relu(safe_pred.abs() - self.zero_thresh) ** p, zero_mask)

        return self.target_weight * (pos_loss + neg_loss) + self.zero_weight * zero_loss

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for ch in range(y.shape[-1]):
            loss = loss + self.forward_channel(
                y_pred[..., ch].clone(), y[..., ch].clone())
        return loss


class UnifiedLoss(nn.Module):
    """ONE loss for all three stages (DPA / GNG / Dual). The task semantics live entirely in
    the TARGETS (src/tasks.py windows); the loss just enforces the value classes:

        +1  → one-sided hinge  relu(thresh − p)²   (beyond +1 free)
        −1  → one-sided hinge  relu(p + thresh)²   (beyond −1 free)
         0  → pin              p²  (MSE-to-0)
        NaN → free             (optional nolick: relu(p)² on free decision windows)

    Every term is averaged over its OWN mask (short expression windows are never diluted by
    long tails). Timing knowledge is deliberately minimal — exactly two splits:
      • pre-sample BASELINE (t < n_on[0], target 0): its own pinned term `bl`, SEPARATE from
        the decay zeros;
      • the DECISION channel splits at `pair_start` into a GNG group (go/nogo: pre-cue hold +
        decay) and a PAIR group (match/nonmatch expression + decay), each with separate
        per-class terms — separate gradient normalisation and separate diagnostics.
        pair_start=None → single GNG group (the GNG stage). For DPA use the test onset
        (everything decision-side is pairing); for Dual use n_on[3].

    Component values are exposed in .last_components after each forward:
        bl · gng_pos/gng_neg/gng_decay · rwd_go/rwd_nogo · pair_pos/pair_neg/pair_decay ·
        mem_pos/mem_neg/mem_decay (non-decision channels, summed) · nolick

    Weights: gng_weight (pre-cue holds), gng_decay_weight, rwd_go_weight/rwd_nogo_weight
    (optional response window, see below), pair_weight (match/nonmatch expression),
    pair_decay_weight, mem_weight, bl_weight, nolick_weight — the decay terms are fully
    independent of their group's expression terms.

    rwd group (optional): pass `rwd_window=(start, stop)` — the response window right after
    cue-off. Decision-channel targets inside it are pulled OUT of the gng group into two
    separately-weighted terms: rwd_go = the +1 targets (hinge ≥ thresh) and rwd_nogo = the
    0/−1 targets (0 → pin MSE-to-0, −1 → hinge ≤ −thresh). This allows a go/nogo imbalance
    after the cue (e.g. up-weight nogo-to-zero against false licks). With no targets in the
    window (gng_response=False task flag) both terms are 0 and the group is inert.
    """

    def __init__(self, timing: TaskTiming, thresh: float = 1.0, readout_index: int = -1,
                 pair_start: int | None = None,
                 rwd_window: tuple[int, int] | None = None,
                 gng_weight: float = 1.0, pair_weight: float = 1.0,
                 gng_decay_weight: float = 1.0, pair_decay_weight: float = 1.0,
                 rwd_go_weight: float = 1.0, rwd_nogo_weight: float = 1.0,
                 rwd_nogo_onesided: bool = False,
                 mem_weight: float = 1.0, bl_weight: float = 1.0,
                 nolick_weight: float = 0.0):
        super().__init__()
        self.timing        = timing
        self.thresh        = thresh
        self.readout_index = readout_index
        self.pair_start    = pair_start
        self.rwd_window    = rwd_window
        self.gng_weight    = gng_weight
        self.pair_weight   = pair_weight
        self.gng_decay_weight  = gng_decay_weight   # decay terms fully independent of their
        self.pair_decay_weight = pair_decay_weight  # group's expression terms
        self.rwd_go_weight   = rwd_go_weight        # response window: go (+1) hinge
        self.rwd_nogo_weight = rwd_nogo_weight      # response window: nogo (0→pin / −1→hinge)
        self.rwd_nogo_onesided = rwd_nogo_onesided  # nogo response: True → one-sided relu(κ₁)²
        # (penalise lick only, no-lick value free); False → pin κ₁² (both sides)
        self.mem_weight    = mem_weight
        self.bl_weight     = bl_weight
        self.nolick_weight = nolick_weight
        self.last_components: dict[str, float] = {}

    @staticmethod
    def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _class_terms(self, p, tgt, mask):
        """(pos, neg, decay) — each its own masked_mean over `mask` ∩ its value class."""
        pos = self.masked_mean(torch.relu(self.thresh - p) ** 2, mask & (tgt > 0))
        neg = self.masked_mean(torch.relu(p + self.thresh) ** 2, mask & (tgt < 0))
        dec = self.masked_mean(p ** 2,                           mask & (tgt == 0))
        return pos, neg, dec

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, T, C = y.shape
        device  = y.device
        dec_ch  = self.readout_index % C
        t       = torch.arange(T, device=device)
        pre     = (t < int(self.timing.n_stim_on[0]))[None, :]

        zero_f  = y_pred.sum() * 0.0            # 0-scalar carrying grad/device/dtype
        comp: dict[str, torch.Tensor] = {
            k: zero_f for k in ("gng_pos", "gng_neg", "gng_decay",
                                "rwd_go", "rwd_nogo",
                                "pair_pos", "pair_neg", "pair_decay",
                                "mem_pos", "mem_neg", "mem_decay", "nolick")}
        bl_loss = zero_f

        for ch in range(C):
            pred   = y_pred[..., ch]
            target = y[..., ch]
            finite = torch.isfinite(pred) & torch.isfinite(target)
            tgt    = torch.where(finite, target, torch.zeros_like(target))
            p      = torch.where(finite, pred,   torch.zeros_like(pred))

            # pre-sample baseline: pinned term, SEPARATE from decay — per-channel mean (its own
            # mask), summed across channels like every other term
            blm     = finite & pre & (tgt == 0)
            bl_loss = bl_loss + self.masked_mean(p ** 2, blm)

            post = finite & ~pre
            if ch == dec_ch:
                if self.pair_start is None:
                    g_mask = post
                    p_mask = torch.zeros_like(post)
                else:
                    after  = (t >= int(self.pair_start))[None, :]
                    g_mask = post & ~after
                    p_mask = post & after
                if self.rwd_window is not None:
                    # response window: its OWN go/nogo terms, carved OUT of the gng group
                    r0, r1 = self.rwd_window
                    rwd_m  = post & ((t >= int(r0)) & (t < int(r1)))[None, :]
                    g_mask = g_mask & ~rwd_m
                    p_mask = p_mask & ~rwd_m
                    if self.rwd_nogo_onesided:
                        # ONLY penalise nogo licking (κ₁>0) in the response window — NO go +1 hinge,
                        # NO nogo −1 hinge. go response and the nogo no-lick value are both free.
                        comp["rwd_go"]   = zero_f
                        comp["rwd_nogo"] = self.masked_mean(torch.relu(p) ** 2, rwd_m & (tgt == 0))
                    else:
                        rgo = self.masked_mean(torch.relu(self.thresh - p) ** 2, rwd_m & (tgt > 0))  # go +1 hinge
                        rn  = self.masked_mean(torch.relu(p + self.thresh) ** 2, rwd_m & (tgt < 0))  # nogo −1 hinge
                        rz  = self.masked_mean(p ** 2, rwd_m & (tgt == 0))                            # nogo pin to 0
                        comp["rwd_go"]   = rgo
                        comp["rwd_nogo"] = rn + rz
                gp, gn, gd = self._class_terms(p, tgt, g_mask)
                pp, pn, pd = self._class_terms(p, tgt, p_mask)
                comp["gng_pos"], comp["gng_neg"], comp["gng_decay"]    = gp, gn, gd
                comp["pair_pos"], comp["pair_neg"], comp["pair_decay"] = pp, pn, pd
                if self.nolick_weight:
                    freem = torch.isfinite(pred) & ~torch.isfinite(target) & ~pre
                    pfree = torch.where(freem, pred, torch.zeros_like(pred))
                    comp["nolick"] = self.masked_mean(torch.relu(pfree) ** 2, freem)
            else:
                mp, mn, md = self._class_terms(p, tgt, post)
                comp["mem_pos"]   = comp["mem_pos"]   + mp
                comp["mem_neg"]   = comp["mem_neg"]   + mn
                comp["mem_decay"] = comp["mem_decay"] + md

        total = (self.bl_weight   * bl_loss
                 + self.gng_weight        * (comp["gng_pos"] + comp["gng_neg"])
                 + self.gng_decay_weight  * comp["gng_decay"]
                 + self.rwd_go_weight     * comp["rwd_go"]
                 + self.rwd_nogo_weight   * comp["rwd_nogo"]
                 + self.pair_weight       * (comp["pair_pos"] + comp["pair_neg"])
                 + self.pair_decay_weight * comp["pair_decay"]
                 + self.mem_weight  * (comp["mem_pos"] + comp["mem_neg"] + comp["mem_decay"])
                 + self.nolick_weight * comp["nolick"])

        self.last_components = {"bl": float(bl_loss.detach()),
                                **{k: float(v.detach()) for k, v in comp.items()}}
        return total
