import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from src.models.efficientnet_attention import (
    AttentionChannels,
    EfficientNetWithAttention,
)


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg=None, num_classes=None, model_name=None):
        super().__init__()

        # Allow initialization with either cfg or direct parameters
        if cfg is not None:
            self.num_classes = cfg.num_classes
            self.model_name = cfg.model.model_name
            self.cfar_scaling_factors = cfg.model.cfar_scaling_factors
            self.kernel_size = cfg.model.kernel_size
            self.mixup_alpha = cfg.model.mixup_alpha if hasattr(cfg.model, "mixup_alpha") else 0
        else:
            self.num_classes = num_classes
            self.model_name = model_name or 'efficientnet-b0'
            self.cfar_scaling_factors = (1, 2)
            self.kernel_size = (3, 3)
            self.mixup_alpha = 0

        self.attention = AttentionChannels(
            kernel_size=self.kernel_size,
            scaling_factors=self.cfar_scaling_factors,
        )

        self.efficientnet = EfficientNet.from_pretrained(
            self.model_name, num_classes=self.num_classes
        )

        backbone_out = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, self.num_classes)

        self.mixup_enabled = self.mixup_alpha > 0

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        self.attention_outputs = self.attention(x)
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

    @staticmethod
    def load_from_checkpoint(checkpoint_path, map_location=None):
        """Load model from checkpoint without requiring original config"""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Extract model parameters from checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            num_classes = checkpoint.get('num_classes', 206)  # Default to 206 if not found
            model_name = checkpoint.get('model_name', 'efficientnet-b0')
        else:
            state_dict = checkpoint
            num_classes = 206
            model_name = 'efficientnet-b0'

        # Create model instance
        model = BirdCLEFModel(num_classes=num_classes, model_name=model_name)
        model.load_state_dict(state_dict)
        return model

    def save_checkpoint(self, path):
        """Save model checkpoint with minimal dependencies"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'architecture': 'BirdCLEFModel'
        }
        torch.save(checkpoint, path)
