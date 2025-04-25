import time
from typing import Dict

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


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


def process_audio_file(
    row: pd.Series,
    config: Dict,
    verbose: bool = False,
) -> Dict:
    """Process a single audio file and return its segments.

    Args:
        row: DataFrame row containing file information
        config: Dictionary containing configuration parameters
        verbose: Whether to log timing information

    Returns:
        Dictionary containing processed segments and metadata
    """
    start_time = time.time()
    try:
        # Force CPU device for preprocessing
        device = torch.device("cpu")

        # Initialize mel transform inside the worker
        mel_transform = MelSpectrogramTransform(config)
        mel_transform.to_melspectogram = mel_transform.to_melspectogram.to(device)
        mel_transform.to_db = mel_transform.to_db.to(device)

        # Load audio
        audio_data, _ = librosa.load(row.filepath, sr=config["SAMPLE_RATE"])
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

            # Check for silence and apply circular padding if needed
            zero_count = torch.sum(audio_segment == 0).item()
            silence_pct = zero_count / audio_segment.numel() * 100
            is_padded = False

            if silence_pct > 0:
                logger.debug(
                    f"Segment {segment_idx} of {row.filename} contains {silence_pct * 100:.1f}% silence, applying circular padding"
                )
                # Apply circular padding to the non-silent parts
                non_silent_mask = audio_segment != 0
                if torch.any(non_silent_mask):
                    non_silent_parts = audio_segment[non_silent_mask]
                    # Calculate how many times we need to repeat to fill the segment
                    repeat_factor = int(
                        np.ceil(len(audio_segment) / len(non_silent_parts))
                    )
                    # Create circularly padded audio
                    padded_audio = torch.tile(non_silent_parts, (repeat_factor,))[
                        : len(audio_segment)
                    ]
                    audio_segment = padded_audio
                    is_padded = True

            min_val = torch.min(audio_segment)
            max_val = torch.max(audio_segment)
            normalized_audio = (audio_segment - min_val) / (
                max_val - min_val
            ) * 2 - 1  # Scale to [-1, 1]

            # Calculate signal power (mean squared amplitude)
            signal_power = torch.mean(normalized_audio**2)

            # Calculate noise power (variance of the signal)
            noise_power = torch.var(normalized_audio)

            # Calculate SNR in dB (avoid division by zero)
            if noise_power == 0:
                snr_db = float("inf")
            else:
                snr_db = 10 * torch.log10(signal_power / noise_power)
                if torch.isnan(snr_db) or torch.isinf(snr_db):
                    snr_db = float("-inf")

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
                    "secondary_labels": row.secondary_labels,
                    "signal_power": signal_power.item(),
                    "noise_power": noise_power.item(),
                    "snr_db": snr_db if isinstance(snr_db, float) else snr_db.item(),
                    "rating": row.rating,
                    "silence_pct": silence_pct,
                    "is_padded": is_padded,
                }
            )

        end_time = time.time()
        processing_time = end_time - start_time  # in seconds
        if verbose:
            logger.info(
                f"Processed {row.filename} in {processing_time:.2f} seconds ({n_segments} segments)"
            )

        return {
            "segments": segments,
            "success": True,
            "error": None,
            "processing_time": processing_time,
        }
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        if verbose:
            logger.info(
                f"Failed to process {row.filename} in {processing_time:.2f} seconds: {e}"
            )
        return {
            "segments": [],
            "success": False,
            "error": str(e),
            "processing_time": processing_time,
        }
