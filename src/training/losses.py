import torch
from torch import nn
from torch.nn import functional as F
import wandb


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
        reduction="mean",
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        if self.reduction == "mean":
            return -loss.mean()
        if self.reduction == "sum":
            return -loss.sum()

        return -loss


class HierarchicalBCELoss(nn.Module):
    def __init__(self, primary_weight=1.0, secondary_weight=0.5):
        super().__init__()
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight

    def forward(self, x, primary_targets, secondary_targets):
        primary_loss = F.binary_cross_entropy_with_logits(x, primary_targets)
        secondary_loss = F.binary_cross_entropy_with_logits(x, secondary_targets)
        return (
            self.primary_weight * primary_loss + self.secondary_weight * secondary_loss
        )


class DynamicWeightedBCELoss(nn.Module):
    def __init__(self, num_classes, momentum=0.9, temperature=2.0, min_weight=0.1):
        super().__init__()
        self.register_buffer('class_errors', torch.ones(num_classes))
        self.momentum = momentum
        self.temperature = temperature
        self.min_weight = min_weight
        self.num_classes = num_classes

    def forward(self, logits, targets):
        device = logits.device
        with torch.no_grad():
            # Calculate current errors per class
            probs = torch.sigmoid(logits)
            errors = torch.abs(probs - targets).mean(0)  # [num_classes]

            # Ensure class_errors is on the correct device
            self.class_errors = self.class_errors.to(device)

            # Update running average of errors
            self.class_errors = (
                self.momentum * self.class_errors +
                (1 - self.momentum) * errors
            )

            # Convert errors to weights using softmax with temperature
            weights = F.softmax(self.class_errors / self.temperature, dim=0)

            # Ensure minimum weight per class
            min_weight_tensor = torch.tensor(self.min_weight, device=device)
            weights = torch.maximum(weights, min_weight_tensor)
            weights = weights / weights.sum()  # Renormalize

            # Log weights and errors if wandb is initialized
            if wandb.run is not None:
                for i in range(self.num_classes):
                    wandb.log({
                        f'loss/class_{i}_error': self.class_errors[i].item(),
                        f'loss/class_{i}_weight': weights[i].item()
                    })

        # Apply dynamic weights to BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        weighted_loss = bce_loss * weights[None, :]
        return weighted_loss.mean()
