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


def process_audio_file(
    row: pd.Series,
    config: Dict,
    use_voice_removal: bool = False,
) -> Dict:
    """Process a single audio file and return its segments.

    Args:
        row: DataFrame row containing file information
        config: Dictionary containing configuration parameters
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
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32, device=device)

        # Initialize voice remover inside the worker process if needed
        has_voice = False
        if use_voice_removal:
            voice_remover = SileroVADRemover(config)
            audio_tensor, has_voice = voice_remover(audio_tensor)

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
                    "label": row.target,
                    "file_idx": row.name,
                    "segment_idx": segment_idx,
                    "filename": row.filename,
                    "primary_label": row.primary_label,
                }
            )

        return {
            "segments": segments,
            "has_voice": has_voice,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {"segments": [], "has_voice": False, "success": False, "error": str(e)}


def save_batch(batch_data: Dict, output_path: Path, batch_idx: int) -> Dict:
    """Save a batch of spectrograms to disk.

    Args:
        batch_data: Dictionary containing batch information
        output_path: Directory to save the batch
        batch_idx: Index of the batch

    Returns:
        Dictionary containing metadata about saved samples
    """
    batch_file = output_path / f"batch_{batch_idx}.pickle"
    torch.save(batch_data, batch_file)

    metadata = []
    for idx, (file_idx, segment_idx) in enumerate(batch_data["indices"]):
        metadata.append(
            {
                "batch_file": str(batch_file),
                "batch_idx": batch_idx,
                "sample_idx": idx,
                "file_idx": file_idx,
                "segment_idx": segment_idx,
                "label": batch_data["labels"][idx].item(),
                "filename": batch_data["filenames"][idx],
                "primary_label": batch_data["class_names"][idx],
            }
        )
    return metadata


def preprocess_and_save_dataset(
    metadata_df: pd.DataFrame,
    config,
    output_dir: Path,
    batch_size: int,
    n_workers: int = None,
) -> Tuple[Path, pd.DataFrame]:
    """Preprocess audio files in parallel and save spectrograms to disk in batches.
    Uses a streaming approach to save batches as they become ready to prevent memory issues.

    Args:
        metadata_df: DataFrame containing metadata
        config: Configuration object
        output_dir: Directory to save processed spectrograms
        batch_size: Number of spectrograms in each batch
        n_workers: Number of worker processes (defaults to CPU count - 1)

    Returns:
        Tuple of (output_dir, processed_metadata_df)
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Log if voice removal is enabled
    if config.REMOVE_VOICE:
        logger.info(
            "Voice removal enabled - will detect and remove voice from recordings"
        )

    # Create a pool of workers
    pool = mp.Pool(n_workers)

    # Create a simplified config for preprocessing that doesn't include distributed training objects
    preprocess_config = {
        "SAMPLE_RATE": config.SAMPLE_RATE,
        "NSAMPLES": config.NSAMPLES,
        "PADMODE": config.PADMODE,
        "UFOLD_OVERLAP": config.UFOLD_OVERLAP,
        "MAKE_RGB": config.MAKE_RGB,
        "REMOVE_VOICE": config.REMOVE_VOICE,
        "N_MELS": config.N_MELS,
        "HOP_LENGTH": config.HOP_LENGTH,
        "N_FFT": config.N_FFT,
        "FMIN": config.FMIN,
        "FMAX": config.FMAX,
    }

    process_func = partial(
        process_audio_file,
        config=preprocess_config,
        use_voice_removal=config.REMOVE_VOICE,
    )

    # Initialize counters and storage
    current_batch = []
    current_batch_idx = 0
    processed_with_voice = 0
    failed_files = []
    all_metadata = []

    try:
        # Process files in groups to control memory usage
        num_files_per_group = 100  # Process 10 files at a time
        for group_st_idx in tqdm(
            range(0, len(metadata_df), num_files_per_group),
            desc=f"Processing audio files in groups of {num_files_per_group}",
            unit="group",
        ):
            group_end_idx = min(group_st_idx + num_files_per_group, len(metadata_df))
            group_df = metadata_df.iloc[group_st_idx:group_end_idx]

            # Process current chunk
            chunk_results = []
            for result in pool.imap(
                process_func, [row for _, row in group_df.iterrows()]
            ):
                if result["success"]:
                    chunk_results.extend(result["segments"])
                    if result["has_voice"]:
                        processed_with_voice += 1
                else:
                    failed_files.append(result["error"])

            # Sort segments by file_idx and segment_idx for reproducibility
            chunk_results.sort(key=lambda x: (x["file_idx"], x["segment_idx"]))

            # Add to current batch and save if full
            for segment in chunk_results:
                current_batch.append(segment)
                if len(current_batch) >= batch_size:
                    # Prepare batch data
                    batch_data = {
                        "spectrograms": torch.stack(
                            [s["spectrogram"] for s in current_batch]
                        ),
                        "labels": torch.tensor([s["label"] for s in current_batch]),
                        "indices": [
                            (s["file_idx"], s["segment_idx"]) for s in current_batch
                        ],
                        "filenames": [s["filename"] for s in current_batch],
                        "class_names": [s["primary_label"] for s in current_batch],
                    }

                    # Save batch and collect metadata
                    batch_metadata = save_batch(
                        batch_data, output_dir, current_batch_idx
                    )
                    all_metadata.extend(batch_metadata)

                    # Clear current batch and increment counter
                    current_batch = []
                    current_batch_idx += 1

            # Clear chunk results to free memory
            chunk_results = None

    except Exception as e:
        logger.error(f"Error during parallel processing: {e}")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()

    # Save any remaining segments as the final batch
    if current_batch:
        batch_data = {
            "spectrograms": torch.stack([s["spectrogram"] for s in current_batch]),
            "labels": torch.tensor([s["label"] for s in current_batch]),
            "indices": [(s["file_idx"], s["segment_idx"]) for s in current_batch],
            "filenames": [s["filename"] for s in current_batch],
            "class_names": [s["primary_label"] for s in current_batch],
        }
        batch_metadata = save_batch(batch_data, output_dir, current_batch_idx)
        all_metadata.extend(batch_metadata)

    # Log statistics
    if config.REMOVE_VOICE:
        logger.info(
            f"Found and removed voice from {processed_with_voice} recordings "
            f"({processed_with_voice/len(metadata_df)*100:.1f}%)"
        )

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        for error in failed_files[:5]:  # Log first 5 errors
            logger.warning(f"Error: {error}")

    processed_metadata_df = pd.DataFrame(all_metadata)
    return output_dir, processed_metadata_df
