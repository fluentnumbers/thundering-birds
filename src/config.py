from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch


@dataclass
class ModelConfig:
    """Configuration for model-specific parameters."""

    name: str
    params: Dict


@dataclass
class Config:
    """Main configuration class for the training pipeline."""

    # Data paths
    DATA_ROOT: Path = Path("data/birdclef-2025")
    TRAIN_AUDIO_DIR: Path = DATA_ROOT / "train_audio"
    TRAIN_METADATA_PATH: Path = DATA_ROOT / "train_metadata.csv"
    TEST_AUDIO_DIR: Path = DATA_ROOT / "test_audio"
    TEST_METADATA_PATH: Path = DATA_ROOT / "test_metadata.csv"
    LOGS_DIR: Path = Path("logs")
    # Training parameters
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    EPOCHS: int = 5
    LR_MAX: float = 1e-3
    DEV_MODE: bool = True
    DEV_MODE_N_SAMPLES: int = 300

    # Model configuration
    MODEL_NAME: str = (
        "efficientnet_attention"  # Options: "efficientnet", "efficientnet_attention"
    )

    # Different configurations based on model type
    # Configuration for efficientnet
    if MODEL_NAME == "efficientnet":
        MODEL_CONFIG = {"efficientnet_version": "efficientnet-b0"}
        MAKE_RGB: bool = True
    # Configuration for efficientnet_attention
    elif MODEL_NAME == "efficientnet_attention":
        MODEL_CONFIG = {
            "efficientnet_version": "efficientnet-b0",
            "kernel_size": (5, 5),
            "cfar_scaling_factors": (1, 20),
        }
        MAKE_RGB: bool = False

    # Audio processing parameters
    SAMPLE_RATE: int = 32000
    DURATION: float = 5.0
    N_MELS: int = 128
    HOP_LENGTH: int = 512
    N_FFT: int = 1024
    FMIN: float = 50
    FMAX: float = 14000
    SEGMENT_DURATION: float = 5  # seconds
    NSAMPLES: int = SEGMENT_DURATION * SAMPLE_RATE
    PADMODE: str = "constant"
    UFOLD_OVERLAP: int = NSAMPLES // 2  # 2.5 seconds overlap

    # Dataset parameters
    N_CLASSES: Optional[int] = None  # Will be set during initialization
    TRAIN_VALID_SPLIT: float = 0.2

    # Logging and visualization
    LOG_DIR: Path = Path("logs")
    SAVE_SPECTROGRAMS: bool = True
    WANDB_PROJECT: str = "bird-sound-classification"

    def __post_init__(self):
        """Initialize paths and create necessary directories."""
        # Create necessary directories
        self.DATA_ROOT.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

        # Set number of classes if not provided
        if self.N_CLASSES is None:
            # This will be set during data loading
            self.N_CLASSES = 0

    @property
    def model_config(self) -> ModelConfig:
        """Get the model configuration as a ModelConfig object."""
        return ModelConfig(name=self.MODEL_NAME, params=self.MODEL_CONFIG)


def get_config() -> Config:
    """Factory function to create a Config instance."""
    return Config()
