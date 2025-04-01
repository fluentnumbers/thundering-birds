import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

# Global configuration variables
LOGS_DIR = Path("logs")


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
    TRAIN_AUDIO_PATH: Path = DATA_ROOT / "train_audio"
    METADATA_PATH: Path = DATA_ROOT / "train.csv"
    PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"

    # Training parameters
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION: bool = (
        torch.cuda.is_available()
    )  # Only enable if CUDA is available
    GRADIENT_ACCUMULATION_STEPS: int = (
        4 if torch.cuda.is_available() else 8  # CPU or low memory GPU
    )
    BATCH_SIZE: int = (
        128 if torch.cuda.is_available() else 64
    )  # Smaller batch size for CPU
    NUM_WORKERS: int = min(10, os.cpu_count() or 1)  # Safe default for num_workers
    EPOCHS: int = 3
    LR_MAX: float = 1e-3
    DEV_MODE: bool = True
    DEV_MODE_N_CLASSES: int = 3
    EARLY_STOPPING_PATIENCE: int = 10  # Number of epochs to wait before early stopping

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
            "cfar_scaling_factors": (1, 2),
        }
        MAKE_RGB: bool = False

    # Audio processing parameters
    REMOVE_VOICE: bool = True
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
    SAVE_SPECTROGRAMS: bool = False
    SAVE_SPECTROGRAMS_N_SAMPLES: int = 3
    WANDB_PROJECT: str = "bird-sound-classification"

    # Distributed Training Configuration - only enabled if CUDA is available
    DISTRIBUTED_TRAINING: bool = (
        torch.cuda.is_available()
    )  # Enable distributed training only if CUDA available
    WORLD_SIZE: int = (
        -1 if not torch.cuda.is_available() else torch.cuda.device_count()
    )  # Total number of processes
    LOCAL_RANK: int = -1  # GPU device ID for each process, set by environment
    DIST_BACKEND: str = (
        "nccl" if torch.cuda.is_available() else "gloo"
    )  # Use NCCL for GPU, gloo for CPU
    DIST_URL: str = "env://"  # URL used to establish distributed training

    @property
    def model_config(self) -> ModelConfig:
        """Get the model configuration as a ModelConfig object."""
        return ModelConfig(name=self.MODEL_NAME, params=self.MODEL_CONFIG)


def get_config() -> Config:
    """Factory function to create a Config instance."""
    return Config()
