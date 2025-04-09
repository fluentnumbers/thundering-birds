import logging
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MelSpectrogramTransform:
    """Transform audio to mel spectrogram with configurable parameters."""

    def __init__(self, config):
        """Initialize the transform with configuration parameters.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.to_melspectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["SAMPLE_RATE"],
            n_fft=config["N_FFT"],
            hop_length=config["HOP_LENGTH"],
            n_mels=config["N_MELS"],
            f_min=config["FMIN"],
            f_max=config["FMAX"],
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.etol = 1e-8
        logger.debug("Initialized MelSpectrogramTransform")

    def __call__(self, audio_sample: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram.

        Args:
            audio_sample: Audio tensor of shape (n_samples,)

        Returns:
            Mel spectrogram tensor of shape (n_mels, time)
        """
        if torch.isnan(audio_sample).any():
            mean_value = torch.nanmean(audio_sample)
            audio_sample = torch.nan_to_num(audio_sample, nan=mean_value)
            logger.warning(
                f"Found NaN values in audio sample, replaced with mean: {mean_value}"
            )

        mel_spec = self.to_melspectogram(audio_sample)
        mel_spec = self.to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.min()) / (
            mel_spec.max() - mel_spec.min() + self.etol
        )

        return mel_spec.clone().detach()


def load_metadata(config) -> pd.DataFrame:
    """Load and prepare metadata."""
    metadata_df = pd.read_csv(config.METADATA_PATH)

    # Add full filepath
    metadata_df["filepath"] = metadata_df["filename"].apply(
        lambda x: os.path.join(config.TRAIN_AUDIO_PATH, x)
    )

    return metadata_df


def process_audio_file(
    row: pd.Series,
    config: Dict,
) -> Dict:
    """Process a single audio file and return its segments.

    Args:
        row: DataFrame row containing file information
        config: Dictionary containing configuration parameters

        config: Configuration object
        use_voice_removal: Whether to use voice removal

    Returns:
        Dictionary containing processed segments and metadata
    """
    try:
        # Force CPU device for preprocessing
        device = torch.device("cpu")

        # Initialize mel transform inside the worker
        mel_transform = MelSpectrogramTransform(config)
        mel_transform.to_melspectogram = mel_transform.to_melspectogram.to(device)
        mel_transform.to_db = mel_transform.to_db.to(device)

        # Load audio
        audio_data, _ = librosa.load(row.filepath, sr=config["SAMPLE_RATE"])
        if audio_data.size == 0:
            logger.warning(f"Empty audio data for {row.filename}")
            return {
                "segments": [],
                "success": False,
                "error": f"Empty audio data for {row.filename}",
            }
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32, device=device)

        # Pad if necessary
        nsamples = audio_tensor.shape[-1]
        rsamples = nsamples % config["NSAMPLES"]
        audio_tensor = torch.nn.functional.pad(
            audio_tensor, (0, config["NSAMPLES"] - rsamples), mode=config["PADMODE"]
        )

        # Calculate number of segments
        n_segments = (len(audio_tensor) - config["NSAMPLES"]) // config[
            "UFOLD_OVERLAP"
        ] + 1

        segments = []
        for segment_idx in range(n_segments):
            start_idx = segment_idx * config["UFOLD_OVERLAP"]
            audio_segment = audio_tensor[start_idx : start_idx + config["NSAMPLES"]]
            # Skip segments that are mostly silence (50% or more zeros)
            zero_count = torch.sum(audio_segment == 0).item()
            if zero_count / audio_segment.numel() >= config["SILENCE_THRESHOLD"]:
                logger.debug(
                    f"Segment {segment_idx} of {row.filename} is mostly silence ({zero_count / audio_segment.numel() * 100:.1f}% zeros)"
                )
                continue

            # Convert to mel spectrogram
            mel_spec = mel_transform(audio_segment)

            # Resize to 224x224
            mel_spec = torch.tensor(
                cv2.resize(mel_spec.numpy(), (224, 224)), device=device
            )

            # Add channel dimension and optionally repeat to 3 channels for RGB
            mel_spec = mel_spec.unsqueeze(0)
            if config["MAKE_RGB"]:
                mel_spec = mel_spec.repeat(3, 1, 1)

            segments.append(
                {
                    "spectrogram": mel_spec,
                    "file_idx": row.name,
                    "segment_idx": segment_idx,
                    "filename": row.filename,
                    "primary_label": row.primary_label,
                }
            )

        return {
            "segments": segments,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {"segments": [], "success": False, "error": str(e)}
