import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import MelSpectrogramTransform


def load_metadata(config) -> pd.DataFrame:
    """Load and prepare metadata."""
    metadata_df = pd.read_csv(f"{config.DATA_ROOT}/train.csv")

    # Add full filepath
    metadata_df["filepath"] = metadata_df["filename"].apply(
        lambda x: os.path.join(config.DATA_ROOT, "train_audio", x)
    )

    # Get unique labels and create label mapping
    labels = sorted(metadata_df["primary_label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    metadata_df["target"] = metadata_df["primary_label"].map(label2id)

    config.N_CLASSES = len(labels)

    # For development: limit to a fractions of  samples while maintaining class distribution
    if config.DEV_MODE:
        metadata_df = metadata_df.groupby("primary_label", group_keys=False).apply(
            lambda x: x.sample(
                n=min(config.DEV_MODE_N_SAMPLES // config.N_CLASSES, len(x))
            )
        )

    return metadata_df


def preprocess_dataset(
    metadata_df: pd.DataFrame, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute all segments and mel spectrograms for the dataset.

    Args:
        metadata_df: DataFrame containing metadata
        config: Configuration object

    Returns:
        Tuple of (precomputed_spectrograms, labels) where spectrograms has shape [n_total_segments, 3, 224, 224]
        and labels has shape [n_total_segments]
    """
    mel_transform = MelSpectrogramTransform(config)

    # Initialize lists to store results
    all_spectrograms = []
    all_labels = []
    total_size_mb = 0

    # Process each audio file
    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Processing audio files"
    ):
        # Load audio
        audio_data, _ = librosa.load(row.filepath, sr=config.FS)
        audio_tensor = torch.tensor(audio_data)

        # Pad if necessary
        nsamples = audio_tensor.shape[-1]
        rsamples = nsamples % config.nsamples
        audio_tensor = torch.nn.functional.pad(
            audio_tensor, (0, config.nsamples - rsamples), mode=config.padmode
        )

        # Calculate number of segments
        n_segments = (len(audio_tensor) - config.nsamples) // config.ufoldoverlap + 1

        # Process each segment
        for segment_idx in range(n_segments):
            start_idx = segment_idx * config.ufoldoverlap
            audio_segment = audio_tensor[start_idx : start_idx + config.nsamples]

            # Convert to mel spectrogram
            mel_spec = mel_transform(audio_segment)

            # Resize to 224x224
            mel_spec = torch.tensor(cv2.resize(mel_spec.numpy(), (224, 224)))

            # Add channel dimension and repeat to 3 channels
            mel_spec = mel_spec.unsqueeze(0).repeat(3, 1, 1)

            # Append to lists
            all_spectrograms.append(mel_spec)
            all_labels.append(row.target)

            # Log memory usage every 1000 segments
            if len(all_spectrograms) % 1000 == 0:
                current_size = sum(
                    spec.element_size() * spec.nelement() for spec in all_spectrograms
                )
                current_size_mb = current_size / (1024 * 1024)
                total_size_mb = current_size_mb

    # Convert lists to tensors
    spectrograms = torch.stack(all_spectrograms)
    labels = torch.tensor(all_labels)

    return spectrograms, labels
