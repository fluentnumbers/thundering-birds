import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    """EfficientNet model for bird sound classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0",
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)
