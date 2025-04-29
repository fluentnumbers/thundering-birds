import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.efficientnet_attention import AttentionChannels, EfficientNetWithAttention
from efficientnet_pytorch import EfficientNet


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Initialize EfficientNetWithAttention model
        # self.backbone = EfficientNetWithAttention(
        #     num_classes=cfg.num_classes,
        #     efficientnet_version=cfg.model.model_name,
        #     kernel_size=cfg.model.kernel_size,
        #     cfar_scaling_factors=cfg.model.cfar_scaling_factors,
        # )
        self.attention = AttentionChannels(
            kernel_size=cfg.model.kernel_size,
            scaling_factors=cfg.model.cfar_scaling_factors,
        )
        self.efficientnet = EfficientNet.from_pretrained(
            cfg.model.model_name, num_classes=cfg.num_classes
        )

        backbone_out = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.feat_dim = backbone_out

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.3),  # Dropout after feature extraction
        #     nn.Linear(backbone_out, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(p=0.2),  # Dropout before final classification
        #     nn.Linear(512, cfg.num_classes),
        # )
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

        self.mixup_enabled = (
            hasattr(cfg.model, "mixup_alpha") and cfg.model.mixup_alpha > 0
        )
        if self.mixup_enabled:
            self.mixup_alpha = cfg.model.mixup_alpha

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        self.attention_outputs = self.attention(x)
        # x = torch.cat((x, x,x ), dim=1)
        features = self.efficientnet(self.attention_outputs)

        if isinstance(features, dict):
            features = features['features']

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(
                F.binary_cross_entropy_with_logits, logits, targets_a, targets_b, lam
            )
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
