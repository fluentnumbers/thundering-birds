import os
from typing import Dict, Optional, Tuple

import albumentations as albu
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MelSpectrogramTransform:
    """Computes the Mel Spectogram of an audio sample."""

    def __init__(self, config):
        self.to_melspectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            f_max=config.FMAX,
            f_min=config.FMIN,
            n_mels=config.N_MELS,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.etol = 1e-8
        logger.debug("Initialized MelSpectrogramTransform")

    def __call__(self, audio_sample: torch.Tensor) -> torch.Tensor:
        if torch.isnan(audio_sample).any():
            mean_value = torch.nanmean(audio_sample)
            audio_sample = torch.nan_to_num(audio_sample, nan=mean_value)
            logger.warning(
                f"Found NaN values in audio sample, replaced with mean: {mean_value}"
            )

        output = self.to_melspectogram(audio_sample)
        output = librosa.power_to_db(output, ref=np.max)
        output = (output - output.min()) / (output.max() - output.min() + self.etol)

        return torch.tensor(output)


class BirdSoundDataset(Dataset):
    """Dataset class for bird sound spectrograms using precomputed data stored on disk."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        augmentation: Optional[albu.Compose] = None,
        mode: str = "train",
    ):
        self.metadata_df = metadata_df
        self.augmentation = augmentation
        self.mode = mode
        logger.info(
            f"Initialized BirdSoundDataset with {len(metadata_df)} samples in {mode} mode"
        )
        if augmentation:
            logger.debug("Using data augmentation")

        self.total_samples = len(metadata_df)

        # Cache for loaded batch data
        self.current_batch_idx = None
        self.current_batch_data = None

    def _load_batch(self, batch_idx: int):
        """Load a batch of spectrograms from disk."""
        if self.current_batch_idx != batch_idx:
            batch_file = self.metadata_df.iloc[0]["batch_file"]
            self.current_batch_data = torch.load(batch_file)
            self.current_batch_idx = batch_idx

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get batch information for this index
        row = self.metadata_df.iloc[index]
        batch_idx = row["batch_idx"]
        sample_idx = row["sample_idx"]

        # Load batch if needed
        self._load_batch(batch_idx)

        # Get spectrogram and label
        spec = self.current_batch_data["spectrograms"][sample_idx]
        label = self.current_batch_data["labels"][sample_idx]

        # Apply augmentations if any
        if self.augmentation and self.mode == "train":
            spec = torch.tensor(self.augmentation(image=spec.numpy())["image"])

        return spec, label


def collate_fn(batch):
    """Custom collate function to handle batching of spectrograms.

    Args:
        batch: List of tuples (mel_spec, label) where mel_spec has shape [3, H, W]
              and label is a scalar

    Returns:
        Tuple of (inputs, labels) where inputs has shape [batch_size, 3, H, W]
        and labels has shape [batch_size]
    """
    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Stack inputs and labels
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)

    return inputs, labels


def get_transforms(mode: str) -> albu.Compose:
    """Get augmentation transforms based on mode."""
    return None
