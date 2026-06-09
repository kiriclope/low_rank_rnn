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
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.regularizer   = regularizer   # callable(model) -> scalar tensor, added to train loss
        self.hebb_lr       = hebb_lr       # three-factor Hebbian lr for reward input (0 = disabled)
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
        total_loss, total_n = 0.0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        use_hebb = self.hebb_lr > 0 and train

        with context:
            for X, y in loader:
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

                if not torch.isfinite(loss):
                    return float("nan")

                if train:
                    if self.regularizer is not None:
                        loss = loss + self.regularizer(self.model)
                    loss.backward()
                    # Block BPTT from touching the reward input (Hebbian-only)
                    if use_hebb and self.model.wi.weight.grad is not None:
                        self.model.wi.weight.grad[:, -1] = 0.0
                    self._zero_low_rank_grads()
                    self._zero_input_grads()
                    if self.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self._restore_frozen_weights()
                    if use_hebb:
                        self._hebb_update(y_pred, y, rates)

                total_loss += loss.item() * X.size(0)
                total_n    += X.size(0)

        if total_n == 0:
            raise ValueError("Dataloader produced zero examples.")
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
    GNG-stage loss. Identical to MaskedMultiTargetLoss for all channels and timesteps,
    except in the response window (t >= n_on[1]) of the decision channel:
      - nogo (target == 0): relu(pred)²  — penalise positive readout only.
      - go   (target  > 0): relu(thresh - pred)² when go_hinge_thresh is set,
                            else standard MSE toward +1.
    Trials with target < 0 (nogo_target=-1) are always trained with standard MSE.
    """

    def __init__(
        self,
        timing: TaskTiming,
        readout_index: int = -1,
        criterion: nn.Module | None = None,
        target_weight: float = 1.0,
        zero_weight: float = 1.0,
        go_hinge_thresh: float | None = None,
    ):
        super().__init__()
        self.timing          = timing
        self.readout_index   = readout_index
        self.criterion       = criterion or nn.MSELoss(reduction="none")
        self.target_weight   = target_weight
        self.zero_weight     = zero_weight
        self.go_hinge_thresh = go_hinge_thresh

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
                nogo_zero_mask = finite & resp_mask & (safe_tgt == 0)
                go_resp_mask   = finite & resp_mask & (safe_tgt > 0)
                other_mask     = finite & ~nogo_zero_mask & ~go_resp_mask

                nogo_hinge = self.masked_mean(torch.relu(safe_pred) ** 2, nogo_zero_mask)

                if self.go_hinge_thresh is not None:
                    go_loss = self.masked_mean(
                        torch.relu(self.go_hinge_thresh - safe_pred) ** 2, go_resp_mask)
                else:
                    go_loss = self.masked_mean(
                        self.criterion(safe_pred, safe_tgt), go_resp_mask)

                other_raw  = torch.where(safe_tgt == 0,
                                         self.criterion(safe_pred, torch.zeros_like(safe_pred)),
                                         self.criterion(safe_pred, safe_tgt))
                other_loss = self.masked_mean(other_raw, other_mask)
                loss = loss + self.zero_weight * nogo_hinge + self.target_weight * go_loss + self.zero_weight * other_loss
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
    ):
        super().__init__()
        self.timing          = timing
        self.readout_index   = readout_index
        self.criterion       = criterion or nn.MSELoss(reduction="none")
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
        gng_win_mask = finite & ((t >= n_on[2]) & (t < n_on[3]))[None, :]
        dpa_mask     = finite & (t >= n_off[3])[None, :]

        go_mask   = gng_win_mask & (safe_t > 0)   # target = +1 → go
        nogo_mask = gng_win_mask & (safe_t <= 0)  # target ≤  0 → nogo

        bl_loss = self.masked_mean(self.criterion(safe_p, torch.zeros_like(safe_p)), bl_mask)
        # go: hinge relu(thresh - pred)² once pred ≥ thresh; else MSE toward +1
        if self.go_hinge_thresh is not None:
            go_loss = self.masked_mean(
                torch.relu(self.go_hinge_thresh - safe_p) ** 2, go_mask)
        else:
            go_loss = self.masked_mean(self.criterion(safe_p, safe_t), go_mask)
        # nogo: hinge relu(pred)² when target==0; MSE when target<0
        nogo_raw  = torch.where(safe_t == 0,
                                torch.relu(safe_p) ** 2,
                                self.criterion(safe_p, safe_t))
        nogo_loss = self.masked_mean(nogo_raw, nogo_mask)
        gng_loss  = self.gng_go_weight * go_loss + self.gng_nogo_weight * nogo_loss
        dpa_loss  = self.masked_mean(self.criterion(safe_p, safe_t), dpa_mask)

        aux_loss = pred_dec.sum() * 0.0   # zero scalar carrying grad/device/dtype
        for ch in range(C):
            if ch == dec:
                continue
            aux_loss = aux_loss + self._aux_channel_loss(
                y_pred[..., ch].clone(), y[..., ch].clone()
            )

        self.last_components = {
            "dpa":      float(dpa_loss.detach()),
            "gng":      float(gng_loss.detach()),
            "gng_go":   float(go_loss.detach()),
            "gng_nogo": float(nogo_loss.detach()),
            "baseline": float(bl_loss.detach()),
            "aux":      float(aux_loss.detach()),
        }

        return (
            self.dpa_weight * dpa_loss
            + self.gng_weight * gng_loss
            + self.bl_weight  * bl_loss
            + self.aux_weight * aux_loss
        )


class ThresholdLoss(nn.Module):
    """
    Squared hinge loss: penalises predictions on the wrong side of a threshold,
    with zero gradient once the prediction is correctly above/below it.

    For target = +1 :  loss = relu(thresh - pred)²      (penalise if pred < thresh)
    For target = -1 :  loss = relu(thresh + pred)²      (penalise if pred > -thresh)
    For target =  0 :  loss = relu(|pred| - thresh)²    (penalise if |pred| > thresh)
    NaN targets are masked out (no gradient).

    Works as a drop-in for MaskedMultiTargetLoss.
    Supports any number of target values as long as they are +1, -1, 0, or nan.
    """

    def __init__(
        self,
        thresh: float = 0.5,
        target_weight: float = 1.0,
        zero_weight: float = 1.0,
    ):
        super().__init__()
        self.thresh        = thresh
        self.target_weight = target_weight
        self.zero_weight   = zero_weight

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

        pos_loss  = self.masked_mean(
            torch.relu(self.thresh - safe_pred) ** 2, pos_mask)
        neg_loss  = self.masked_mean(
            torch.relu(self.thresh + safe_pred) ** 2, neg_mask)
        zero_loss = self.masked_mean(
            torch.relu(safe_pred.abs() - self.thresh) ** 2, zero_mask)

        return self.target_weight * (pos_loss + neg_loss) + self.zero_weight * zero_loss

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for ch in range(y.shape[-1]):
            loss = loss + self.forward_channel(
                y_pred[..., ch].clone(), y[..., ch].clone())
        return loss
