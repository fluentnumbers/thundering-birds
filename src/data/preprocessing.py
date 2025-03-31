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
from src.data.voice_removal import SileroVADRemover

logger = logging.getLogger(__name__)


def load_metadata(config) -> pd.DataFrame:
    """Load and prepare metadata."""
    metadata_df = pd.read_csv(config.METADATA_PATH)

    # Add full filepath
    metadata_df["filepath"] = metadata_df["filename"].apply(
        lambda x: os.path.join(config.TRAIN_AUDIO_PATH, x)
    )

    return metadata_df


def preprocess_and_save_dataset(
    metadata_df: pd.DataFrame,
    config,
    output_dir: Path,
    batch_size: int,
) -> Tuple[Path, pd.DataFrame]:
    """Preprocess audio files and save spectrograms to disk in batches.

    Args:
        metadata_df: DataFrame containing metadata
        config: Configuration object
        output_dir: Directory to save processed spectrograms
        batch_size: Number of spectrograms in each batch

    Returns:
        Tuple of (output_dir, processed_metadata_df) containing paths to saved spectrograms
    """
    mel_transform = MelSpectrogramTransform(config)

    # Initialize voice remover if needed
    voice_remover = None
    if config.REMOVE_VOICE:
        voice_remover = SileroVADRemover(config)
        logger.info(
            "Voice removal enabled - will detect and remove voice from recordings"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store batch data
    current_batch_spectrograms = []
    current_batch_labels = []
    current_batch_indices = []
    batch_file_counter = 0
    processed_with_voice = 0

    # Process each audio file
    for idx, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Processing audio files",
        unit="file",
    ):
        # Load audio
        audio_data, _ = librosa.load(row.filepath, sr=config.SAMPLE_RATE)
        audio_tensor = torch.tensor(audio_data)

        # Check for voice and remove if needed
        if voice_remover:
            audio_tensor, has_voice = voice_remover(audio_tensor)
            if has_voice:
                processed_with_voice += 1

        # Pad if necessary
        nsamples = audio_tensor.shape[-1]
        rsamples = nsamples % config.NSAMPLES
        audio_tensor = torch.nn.functional.pad(
            audio_tensor, (0, config.NSAMPLES - rsamples), mode=config.PADMODE
        )

        # Calculate number of segments
        n_segments = (len(audio_tensor) - config.NSAMPLES) // config.UFOLD_OVERLAP + 1

        # Process each segment
        for segment_idx in range(n_segments):
            start_idx = segment_idx * config.UFOLD_OVERLAP
            audio_segment = audio_tensor[start_idx : start_idx + config.NSAMPLES]

            # Convert to mel spectrogram
            mel_spec = mel_transform(audio_segment)

            # Resize to 224x224
            mel_spec = torch.tensor(cv2.resize(mel_spec.numpy(), (224, 224)))

            # Add channel dimension and optionally repeat to 3 channels for RGB
            mel_spec = mel_spec.unsqueeze(0)
            if config.MAKE_RGB:
                mel_spec = mel_spec.repeat(3, 1, 1)

            # Append to current batch
            current_batch_spectrograms.append(mel_spec)
            current_batch_labels.append(row.target)
            current_batch_indices.append((idx, segment_idx))

            # Save batch when it reaches batch_size
            if len(current_batch_spectrograms) >= batch_size:
                # Stack current batch
                batch_data = {
                    "spectrograms": torch.stack(current_batch_spectrograms),
                    "labels": torch.tensor(current_batch_labels),
                    "indices": current_batch_indices,
                }

                # Save batch file
                batch_file = output_dir / f"batch_{batch_file_counter}.pickle"
                torch.save(batch_data, batch_file)

                # Reset batch lists
                current_batch_spectrograms = []
                current_batch_labels = []
                current_batch_indices = []
                batch_file_counter += 1

    # Save any remaining samples
    if current_batch_spectrograms:
        batch_data = {
            "spectrograms": torch.stack(current_batch_spectrograms),
            "labels": torch.tensor(current_batch_labels),
            "indices": current_batch_indices,
        }
        batch_file = output_dir / f"batch_{batch_file_counter}.pickle"
        torch.save(batch_data, batch_file)

    # Create a DataFrame with batch file information
    processed_metadata = []
    for batch_file in sorted(output_dir.glob("batch_*.pickle")):
        batch_data = torch.load(batch_file)
        for idx, (file_idx, segment_idx) in enumerate(batch_data["indices"]):
            processed_metadata.append(
                {
                    "batch_file": str(batch_file),
                    "batch_idx": batch_file.stem.split("_")[
                        1
                    ],  # Extract the batch number
                    "sample_idx": idx,
                    "file_idx": file_idx,
                    "segment_idx": segment_idx,
                    "label": batch_data["labels"][idx].item(),
                }
            )

    # Log final statistics
    if config.REMOVE_VOICE:
        logger.info(
            f"Found and removed voice from {processed_with_voice} recordings ({processed_with_voice/len(metadata_df)*100:.1f}%)"
        )

    processed_metadata_df = pd.DataFrame(processed_metadata)
    return output_dir, processed_metadata_df
