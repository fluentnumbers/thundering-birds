from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CFARLayer(nn.Module):
    """
    Constant False Alarm Rate (CFAR) detection layer.
    Implements a learnable 2D CFAR detection mechanism.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 5),
        scaling_factor: float = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size

        # Calculate padding for 'same' output size
        self.padding = (
            kernel_size[0] // 2,  # Vertical padding
            kernel_size[1] // 2,  # Horizontal padding
        )

        # Add activation
        self.activation = nn.ReLU()

        # Fixed uniform kernel for average noise estimation
        self.kernel = torch.ones(1, 1, *kernel_size) / (kernel_size[0] * kernel_size[1])
        self.kernel = nn.Parameter(self.kernel, requires_grad=True)

        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply manual padding for 'same' output size
        local_noise = F.conv2d(x, self.kernel, padding=self.padding)
        threshold = local_noise * self.scaling_factor
        detected = torch.where(x > threshold, x, torch.zeros_like(x))
        return self.activation(detected)


class AttentionChannels(nn.Module):
    """
    Generate attention channels from MEL spectrogram using CFAR detection.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        scaling_factors: Tuple[float, float] = (5, 20),
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)

        # CFAR layers
        self.cfar1 = CFARLayer(
            kernel_size=kernel_size, scaling_factor=scaling_factors[0]
        )
        self.cfar2 = CFARLayer(
            kernel_size=kernel_size, scaling_factor=scaling_factors[1]
        )

        # Convert to RGB-like channels with learned transformation
        self.to_rgb = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        # ImageNet normalization parameters
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        attention1 = self.cfar1(x)
        attention2 = self.cfar2(x)

        # Combine channels
        combined = torch.cat((x, attention1, attention2), dim=1)

        # Convert to RGB-like space
        rgb_like = self.to_rgb(combined)

        # Apply ImageNet normalization
        normalized = (rgb_like - self.mean) / self.std

        return normalized


class EfficientNetWithAttention(nn.Module):
    """
    EfficientNet model with CFAR-based attention mechanism for bird sound classification.
    """

    def __init__(
        self,
        num_classes: int,
        efficientnet_version: str = "efficientnet-b0",
        kernel_size: Tuple[int, int] = (3, 3),
        cfar_scaling_factors: Tuple[float, float] = (0.5, 0.7),
    ):
        super().__init__()

        self.attention = AttentionChannels(
            kernel_size=kernel_size,
            scaling_factors=cfar_scaling_factors,
        )

        # Load pre-trained EfficientNet
        self.efficientnet = EfficientNet.from_pretrained(
            efficientnet_version, num_classes=num_classes
        )

        # Add final classification layers with appropriate dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout after feature extraction
            nn.Linear(self.efficientnet._fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),  # Dropout before final classification
            nn.Linear(512, num_classes),
        )

        # Replace the original classifier
        self.efficientnet._fc = self.classifier

        # Register attention outputs as a buffer to avoid gradient computation
        self.register_buffer("attention_outputs", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store attention outputs as a buffer to avoid gradient computation
        self.attention_outputs = self.attention(x)
        return self.efficientnet(self.attention_outputs)

    def get_attention_outputs(self) -> Optional[torch.Tensor]:
        """
        Returns the stored attention outputs if available.

        Returns:
            Optional[torch.Tensor]: The 3-channel attention outputs (original + 2 attention channels)
        """
        return self.attention_outputs

    def clear_attention_outputs(self) -> None:
        """
        Clears the stored attention outputs to free up memory.
        Should be called after saving or using the attention outputs.
        """
        self.attention_outputs = None


def create_model(
    num_classes: int,
    efficientnet_version: str = "efficientnet-b0",
    kernel_size: Tuple[int, int] = (3, 3),
    cfar_scaling_factors: Tuple[float, float] = (0.5, 0.7),
) -> EfficientNetWithAttention:
    """
    Factory function to create an EfficientNetWithAttention model.

    Args:
        num_classes: Number of output classes
        efficientnet_version: Version of EfficientNet to use
        kernel_size: Size of the CFAR kernel
        cfar_scaling_factors: Initial scaling factors for the two CFAR layers

    Returns:
        Initialized EfficientNetWithAttention model
    """
    return EfficientNetWithAttention(
        num_classes=num_classes,
        efficientnet_version=efficientnet_version,
        kernel_size=kernel_size,
        cfar_scaling_factors=cfar_scaling_factors,
    )
