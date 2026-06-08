import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_mean(tensor):
    """Returns mean or zero if tensor is empty."""
    if tensor.numel() == 0:
        return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    else:
        return tensor.mean()

class BCEOneClassLoss(nn.Module):
    # Your original BCEOneClassLoss goes here
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    # def forward(self, logits, targets, class_bal=1):
    #     bce = self.criterion(logits, targets.float())

    #     if class_bal == 0:
    #         mask_pos = (targets == 1)
    #         mask_neg = (targets == 0)
    #         pos_loss = safe_mean(bce[mask_pos])
    #         # For negatives: encourage sigmoid(logit) => 0.5 via MSE
    #         if mask_neg.any():
    #             neg_loss = safe_mean((torch.sigmoid(logits[mask_neg]) - 0.5) ** 2)
    #             return pos_loss + 0.1 * neg_loss
    #         else:
    #             return pos_loss

    #     return safe_mean(bce)

    def forward(self, logits, targets, class_bal=1):
        bce = self.criterion(logits, targets.float())

        if class_bal == 0:
            mask1 = (targets == 1)
            pos_loss = safe_mean(bce[mask1])
            mask0 = (targets == 0)
            # Encourage proba = 0.5 for class 0 (logit=0)
            neutral_loss = safe_mean(self.criterion(logits[mask0], torch.full_like(logits[mask0], 0.5)))
            return pos_loss + 0.01 * neutral_loss

        return safe_mean(bce)

    # def forward(self, logits, targets, class_bal=1):
    #     criterion = nn.BCEWithLogitsLoss(reduction='none')
    #     bce = criterion(logits, targets.float())

    #     if class_bal == 0:
    #         mask = (targets == 1)
    #         return safe_mean(bce[mask])

    #     return safe_mean(bce)

class SignBCELoss(nn.Module):
    def __init__(self, alpha=0.5, thresh=1.0, class_bal=0):
        super().__init__()
        self.alpha = alpha
        self.thresh = thresh
        self.class_bal = class_bal
        self.bce_with_logits = BCEOneClassLoss()

    def forward(self, readout, targets):
        # BCE loss (can be 0 if alpha==1)
        bce_loss = 0.0
        if self.alpha != 1.0:
            bce_loss = self.bce_with_logits(readout, targets, self.class_bal)

        sign_overlap = torch.sign(2 * targets - 1) * readout
        sign_loss = torch.zeros_like(sign_overlap)

        if self.alpha!=0:
            if self.class_bal == 0:
                # Penalize class 0 (targets==0) with |overlap|
                mask0 = (targets == 0)
                if mask0.sum() > 0:
                    sign_loss[mask0] = 0.1 * torch.abs(sign_overlap[mask0])
                    # Penalize class 1 (targets==1) with relu(thresh - overlap)
                mask1 = (targets == 1)
                if mask1.sum() > 0:
                    sign_loss[mask1] = F.relu(self.thresh - sign_overlap[mask1])
            else:
                sign_loss = F.relu(self.thresh - sign_overlap)

        # Combine safely
        loss = ((1 - self.alpha) * bce_loss + self.alpha * safe_mean(sign_loss))

        return loss
#+end_src

#+begin_src jupyter-python
import torch
import torch.nn as nn

class DualLoss(nn.Module):
      def __init__(self, alpha=1.0, thresh=5.0, stim_idx=[], gng_idx=[], cue_idx=[], test_idx=[], rwd_idx=-1, zero_idx=[], read_idx=[-1], class_bal=[0]):
            super(DualLoss, self).__init__()
            self.alpha = alpha
            self.thresh = thresh
            self.class_bal = class_bal

            # BL idx
            self.zero_idx = zero_idx
            # Sample idx
            self.stim_idx = torch.tensor(stim_idx, dtype=torch.int, device=DEVICE)
            # Go NoGo
            self.gng_idx= torch.tensor(gng_idx, dtype=torch.int, device=DEVICE)
            # rwd idx for DRT
            self.cue_idx = torch.tensor(cue_idx, dtype=torch.int, device=DEVICE)
            # rwd idx for DPA
            self.rwd_idx = torch.tensor(rwd_idx, dtype=torch.int, device=DEVICE)
            # test idx for DPA
            self.test_idx = torch.tensor(test_idx, dtype=torch.int, device=DEVICE)

            # readout idx
            self.read_idx = read_idx

            self.loss = SignBCELoss(self.alpha, self.thresh)
            self.l1loss = nn.SmoothL1Loss()


      def forward(self, readout, targets):

            zeros = torch.zeros_like(readout[:, self.zero_idx, 0])
            # custom zeros for readout
            loss = self.l1loss(readout[:, self.zero_idx, self.read_idx[0]], zeros)
            # zero memory only before stim
            if len(self.read_idx)>1:
                  loss += self.l1loss(readout[:, :self.stim_idx[0]-1, self.read_idx[1]], zeros[:, :self.stim_idx[0]-1])

            is_stim = (self.stim_idx.numel() != 0)
            is_gng = (self.gng_idx.numel() != 0)
            is_cue = (self.cue_idx.numel() != 0)
            is_test = (self.test_idx.numel() != 0)
            is_rwd = (self.rwd_idx.numel() != 0)

            if is_test:
                  self.loss.class_bal = 1
                  loss += self.loss(readout[:, self.test_idx, self.read_idx[-1]], targets[:, -1, :self.test_idx.shape[0]])

            if is_cue:
                  self.loss.class_bal = self.class_bal[3]
                  loss += self.loss(readout[:, self.cue_idx, self.read_idx[3]], targets[:, 2, :self.cue_idx.shape[0]])

            if is_gng:
                  self.loss.class_bal = 1
                  loss += self.loss(readout[:,  self.gng_idx, self.read_idx[2]], targets[:, 2, :self.gng_idx.shape[0]])

            if is_stim:
                  self.loss.class_bal = 1
                  loss += self.loss(readout[:,  self.stim_idx, self.read_idx[1]], targets[:, 1, :self.stim_idx.shape[0]])

            if is_rwd:
                  self.loss.class_bal = self.class_bal[0]
                  loss += self.loss(readout[:,  self.rwd_idx, self.read_idx[0]], targets[:, 0, :self.rwd_idx.shape[0]])

            return loss
