from typing import Optional, Tuple

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class CFARLayer(nn.Module):
    """
    Constant False Alarm Rate (CFAR) detection layer.
    Implements a learnable 2D CFAR detection mechanism.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 5),
        scaling_factor: float = 0.5,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = "same"

        # Initialize the convolution kernel for noise estimation

        # Fixed uniform kernel for average noise estimation
        kernel = torch.ones(1, 1, *kernel_size) / (kernel_size[0] * kernel_size[1])
        kernel = nn.Parameter(kernel, requires_grad=True)
        self.register_buffer("kernel", kernel)

        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed kernel noise estimation
        local_noise = nn.functional.conv2d(x, self.kernel, padding=self.padding)

        # Apply adaptive thresholding
        return torch.where(
            x > (local_noise * self.scaling_factor), x, torch.zeros_like(x)
        )


class AttentionChannels(nn.Module):
    """
    Generate attention channels from MEL spectrogram using CFAR detection.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        scaling_factors: Tuple[float, float] = (0.5, 0.7),
    ):
        super().__init__()
        self.cfar1 = CFARLayer(
            kernel_size=kernel_size,
            scaling_factor=scaling_factors[0],
        )
        self.cfar2 = CFARLayer(
            kernel_size=kernel_size,
            scaling_factor=scaling_factors[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate two attention channels
        attention1 = self.cfar1(x)
        attention2 = self.cfar2(x)

        # Combine original spectrogram with attention channels
        return torch.cat((x, attention1, attention2), dim=1)


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

        # Attention mechanism
        self.attention = AttentionChannels(
            kernel_size=kernel_size,
            scaling_factors=cfar_scaling_factors,
        )

        # Load pre-trained EfficientNet
        self.efficientnet = EfficientNet.from_pretrained(
            efficientnet_version, num_classes=num_classes
        )

        # Store attention outputs
        self.attention_outputs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate three-channel input using attention mechanism
        self.attention_outputs = self.attention(x)
        # Pass through EfficientNet
        return self.efficientnet(self.attention_outputs)

    def get_attention_outputs(self) -> Optional[torch.Tensor]:
        """
        Returns the stored attention outputs if available.

        Returns:
            Optional[torch.Tensor]: The 3-channel attention outputs (original + 2 attention channels)
        """
        return self.attention_outputs


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
