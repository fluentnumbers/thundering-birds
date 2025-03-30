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


def load_silero_vad():
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )
        logger.info("Successfully loaded Silero VAD model")
        return model, utils
    except HTTPError as e:
        logger.error(f"Failed to download model: {e}")
        logger.error("Please check your internet connection and try again")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error loading Silero VAD model: {e}")
        return None, None


class SileroVADRemover:
    def __init__(self, config):
        self.target_fs = 16000  # silero works @ 8khz/16khz
        self.orig_fs = config.SAMPLE_RATE
        self.rsratio = self.orig_fs // self.target_fs

        self.maxmergingtime_s = 1
        self.maxdistance = self.maxmergingtime_s * config.SAMPLE_RATE

        # fast resampler - similar config as librosa : from torchaudio docs
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=config.SAMPLE_RATE,
            new_freq=self.target_fs,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="sinc_interp_kaiser",
            beta=8.555504641634386,
        )

        torch.set_num_threads(1)
        self.model, (self.get_speech_timestamps, _, _, _, _) = load_silero_vad()

    def remove_voice(
        self, audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Remove voice from audio using Silero VAD.

        Args:
            audio: Audio tensor of shape [n_samples]

        Returns:
            Tuple of (processed_audio, original_audio, has_voice)
        """
        rsaudiosample = self.resampler(audio)
        speech_timestamps = self.get_speech_timestamps(rsaudiosample, self.model)
        has_voice = len(speech_timestamps) > 0

        maskspeech = torch.ones(size=(audio.numel(),))
        prev = -np.inf

        for st in speech_timestamps:
            if st["start"] - prev < self.maxdistance:
                st["start"] = prev
            maskspeech[self.rsratio * st["start"] : self.rsratio * st["end"]] = 0
            prev = st["end"]

        # Remove detected voice blocks
        newsample = audio[maskspeech == 1]

        return newsample, audio, has_voice

    def __call__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Process audio and return voice-removed version and voice detection status."""
        processed_audio, _, has_voice = self.remove_voice(audio)
        return processed_audio, has_voice


def should_remove_voice(author: str, collection: str) -> bool:
    """Check if voice removal should be applied for a given author and collection."""
    return collection in COLLECTIONS_WITH_VOICE
