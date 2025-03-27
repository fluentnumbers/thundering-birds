import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    """EfficientNet model for bird sound classification."""

    def __init__(self, num_classes: int, efficientnet_version: str = "efficientnet-b0"):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            efficientnet_version,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)


def create_model(
    num_classes: int,
    efficientnet_version: str = "efficientnet-b0",
) -> EfficientNetModel:
    """
    Factory function to create an EfficientNet model.

    Args:
        num_classes: Number of output classes
        efficientnet_version: Version of EfficientNet to use

    Returns:
        Initialized EfficientNetModel
    """
    return EfficientNetModel(
        num_classes=num_classes,
        efficientnet_version=efficientnet_version,
    )
