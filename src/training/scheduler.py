import numpy as np
from torch.optim import lr_scheduler


class WarmupCosineScheduler:
    """Custom scheduler with linear warmup and cosine decay."""

    def __init__(self, optimizer, warmup_epochs, warmup_factor, total_epochs, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.current_epoch = 0

        # Store initial learning rates for each parameter group
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        """Update learning rates based on current epoch."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.warmup_factor + (1 - self.warmup_factor) * (
                self.current_epoch / self.warmup_epochs
            )
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        # Update learning rate for each parameter group
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = max(base_lr * factor, self.min_lr)

    def get_last_lr(self):
        """Return current learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Return state dictionary for checkpointing."""
        return {"current_epoch": self.current_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state_dict):
        """Load state dictionary from checkpoint."""
        self.current_epoch = state_dict["current_epoch"]
        self.base_lrs = state_dict["base_lrs"]


def get_scheduler(optimizer, cfg):
    """Get learning rate scheduler with warmup and custom scheduling.

    Implements:
    1. Linear warmup for first few epochs
    2. Custom scheduling based on dataset characteristics
    3. Cosine decay for main training phase
    """
    if cfg.training.SCHEDULER == "CosineAnnealingLR":
        # Create scheduler with warmup
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.T_MAX,
            eta_min=cfg.training.MIN_LR,
        )
    elif cfg.training.SCHEDULER == "WarmupCosineScheduler":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=cfg.training.WARMUP_EPOCHS,
            warmup_factor=cfg.training.WARMUP_FACTOR,
            total_epochs=cfg.training.T_MAX,
            min_lr=cfg.training.MIN_LR,
        )
    elif cfg.training.SCHEDULER == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=cfg.training.MIN_LR,
            verbose=True,
        )
    elif cfg.training.SCHEDULER == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.EPOCHS // 3, gamma=0.5
        )
    elif cfg.training.SCHEDULER == "OneCycleLR":
        scheduler = None
    else:
        scheduler = None

    return scheduler
