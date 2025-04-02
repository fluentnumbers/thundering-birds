import logging
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from requests import HTTPError

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# List of authors whose recordings contain human voice
AUTHORS_WITH_VOICE = []
COLLECTIONS_WITH_VOICE = ["CSA"]


def load_silero_vad(config):
    """Load Silero VAD model and utilities.

    Args:
        config: Dictionary containing configuration parameters

    Returns:
        Tuple of (model, utils) or (None, None) if loading fails
    """
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )

        # Force model to CPU
        model = model.cpu()
        logger.debug("Successfully loaded Silero VAD model on CPU")
        return model, utils
    except HTTPError as e:
        logger.error(f"Failed to download model: {e}")
        logger.error("Please check your internet connection and try again")
        return None, None
    except Exception as e:
        logger.error(f"Error loading Silero VAD model: {e}")
        return None, None


class SileroVADRemover:
    """Remove voice from audio using Silero VAD."""

    def __init__(self, config):
        """Initialize the voice remover.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.device = torch.device("cpu")
        self.model, self.utils = load_silero_vad(config)
        if self.model is None:
            logger.error("Failed to initialize Silero VAD model")
            return

        self.model = self.model.to(self.device)
        self.model.eval()

        # Get the speech timestamps function from utils
        self.get_speech_timestamps = self.utils[0]

        # Set resampling parameters
        self.target_fs = 16000  # Silero VAD works best at 16kHz
        self.orig_fs = config["SAMPLE_RATE"]
        self.rsratio = self.orig_fs // self.target_fs

        # Initialize resampler
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.orig_fs,
            new_freq=self.target_fs,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="sinc_interp_kaiser",
            beta=8.555504641634386,
        ).to(self.device)

    def __call__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Remove voice from audio if detected.

        Args:
            audio: Audio tensor of shape (n_samples,)

        Returns:
            Tuple of (processed_audio, has_voice)
        """
        if self.model is None:
            logger.warning("Silero VAD model not initialized, returning original audio")
            return audio, False

        with torch.no_grad():
            # Resample audio to 16kHz for VAD
            audio_16k = self.resampler(audio)

            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_16k, self.model, sampling_rate=self.target_fs, return_seconds=True
            )

            if not speech_timestamps:
                return audio, False

            # Convert timestamps to original sample rate
            speech_samples = []
            for ts in speech_timestamps:
                start_sample = int(ts["start"] * self.orig_fs)
                end_sample = int(ts["end"] * self.orig_fs)
                speech_samples.append((start_sample, end_sample))

            # Remove speech segments
            processed_audio = audio.clone()
            for start, end in speech_samples:
                # Replace speech with silence
                processed_audio[start:end] = 0

            return processed_audio, True


def should_remove_voice(author: str, collection: str) -> bool:
    """Check if voice removal should be applied for a given author and collection."""
    return collection in COLLECTIONS_WITH_VOICE
