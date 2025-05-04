from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# Hardcoded configuration
INPUT_MODEL_PATH = r"C:\Users\andre\Documents\repositories\thundering-birds\training_run_20250429_123129\model_fold0_epoch26_best.pth"  # Replace with your model path
OUTPUT_MODEL_PATH = Path(INPUT_MODEL_PATH).parent / Path(INPUT_MODEL_PATH).stem
NUM_CLASSES = 10  # Replace with your number of classes

# Add safe globals for numpy types
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


class CFARLayer(nn.Module):
    """Constant False Alarm Rate (CFAR) detection layer."""

    def __init__(self, kernel_size=(5, 5), scaling_factor=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.activation = nn.ReLU()
        self.kernel = nn.Parameter(
            torch.ones(1, 1, *kernel_size) / (kernel_size[0] * kernel_size[1]),
            requires_grad=True,
        )
        self.scaling_factor = scaling_factor

    def forward(self, x):
        local_noise = F.conv2d(x, self.kernel, padding=self.padding)
        threshold = local_noise * self.scaling_factor
        detected = torch.where(x > threshold, x, torch.zeros_like(x))
        return self.activation(detected)


class AttentionChannels(nn.Module):
    """Generate attention channels from MEL spectrogram using CFAR detection."""

    def __init__(self, kernel_size=(3, 3), scaling_factors=(5, 20)):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.cfar1 = CFARLayer(
            kernel_size=kernel_size, scaling_factor=scaling_factors[0]
        )
        self.cfar2 = CFARLayer(
            kernel_size=kernel_size, scaling_factor=scaling_factors[1]
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        attention1 = self.cfar1(x)
        attention2 = self.cfar2(x)
        combined = torch.cat((x, attention1, attention2), dim=1)
        rgb_like = self.to_rgb(combined)
        normalized = (rgb_like - self.mean) / self.std
        return normalized


class StandaloneBirdCLEFModel(nn.Module):
    """EfficientNet model with CFAR-based attention mechanism."""

    def __init__(
        self,
        num_classes=206,
        efficientnet_version="efficientnet-b0",
        kernel_size=(3, 3),
        cfar_scaling_factors=(0.5, 0.7),
    ):
        super().__init__()

        self.attention = AttentionChannels(
            kernel_size=kernel_size,
            scaling_factors=cfar_scaling_factors,
        )

        self.efficientnet = EfficientNet.from_pretrained(
            efficientnet_version, num_classes=num_classes
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.efficientnet._fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        self.efficientnet._fc = self.classifier

    def forward(self, x):
        x = self.attention(x)
        return self.efficientnet(x)


def convert_state_dict(state_dict):
    """Convert state dict keys to match standalone model."""
    new_state_dict = {}
    for k, v in state_dict.items():
        # Handle different key patterns
        if k.startswith("model."):
            new_k = k[len("model.") :]  # Remove 'model.' prefix
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def main():
    print(f"Loading model from {INPUT_MODEL_PATH}")

    try:
        # Try loading with weights_only=False first
        checkpoint = torch.load(
            INPUT_MODEL_PATH, map_location="cpu", weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    except Exception as e:
        print(f"Error loading model with weights_only=False: {e}")
        try:
            # Fallback to weights_only=True with safe globals
            checkpoint = torch.load(
                INPUT_MODEL_PATH, map_location="cpu", weights_only=True
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        except Exception as e:
            print(f"Error loading model with weights_only=True: {e}")
            return

    # Create new model
    model = StandaloneBirdCLEFModel(
        num_classes=NUM_CLASSES,
        efficientnet_version="efficientnet-b0",
        kernel_size=(3, 3),
        cfar_scaling_factors=(0.5, 0.7),
    )

    # Convert and load state dict
    new_state_dict = convert_state_dict(state_dict)

    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("Successfully loaded weights")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # Save in standalone format
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "model_name": "efficientnet-b0",
        },
        OUTPUT_MODEL_PATH,
    )

    print(f"Saved standalone model to {OUTPUT_MODEL_PATH}")

    # Verify the saved model
    print("\nVerifying saved model...")
    try:
        checkpoint = torch.load(OUTPUT_MODEL_PATH)
        verify_model = StandaloneBirdCLEFModel(num_classes=checkpoint["num_classes"])
        verify_model.load_state_dict(checkpoint["model_state_dict"])

        # Set model to eval mode for verification
        verify_model.eval()

        # Test with dummy input (1-channel input for mel spectrogram)
        dummy_input = torch.randn(1, 1, 224, 224)  # Changed to 1-channel input
        with torch.no_grad():
            output = verify_model(dummy_input)
        print(f"Verification successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Verification failed: {e}")


if __name__ == "__main__":
    main()
