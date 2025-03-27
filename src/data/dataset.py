import os
from typing import Dict, Tuple

import albumentations as albu
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class MelSpectrogramTransform:
    """Computes the Mel Spectogram of an audio sample."""

    def __init__(self, config):
        self.to_melspectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.FS,
            n_fft=config.N_FFT,
            hop_length=config.WIN_LAP,
            f_max=config.MAX_FREQ,
            f_min=config.MIN_FREQ,
            n_mels=config.N_MELS,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.etol = 1e-8

    def __call__(self, audio_sample: torch.Tensor) -> torch.Tensor:
        if torch.isnan(audio_sample).any():
            mean_value = torch.nanmean(audio_sample)
            audio_sample = torch.nan_to_num(audio_sample, nan=mean_value)

        output = self.to_melspectogram(audio_sample)
        output = librosa.power_to_db(output, ref=np.max)
        output = (output - output.min()) / (output.max() - output.min() + self.etol)

        return torch.tensor(output)


class BirdSoundDataset(Dataset):
    """Dataset class for bird sound spectrograms using precomputed data."""

    def __init__(
        self,
        spectrograms: torch.Tensor,
        labels: torch.Tensor,
        augmentation=None,
        mode="train",
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augmentation = augmentation
        self.mode = mode
        self.total_samples = len(spectrograms)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = self.spectrograms[index]
        label = self.labels[index]

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
