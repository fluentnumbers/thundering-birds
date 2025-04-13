import json
import logging
import multiprocessing as mp
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from src.data.preprocessing import process_audio_file
from src.data.voice_removal import SileroVADRemover
from src.train_notebook import CFG, set_seed
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Global variable to store the preprocessing config per process
_process_config = None


def init_process(cfg: CFG) -> None:
    """Initialize the preprocessing configuration for each process.

    Args:
        cfg: Configuration object containing preprocessing parameters
    """
    global _process_config
    try:
        # Create a simplified config for preprocessing
        _process_config = {
            "SAMPLE_RATE": cfg.preprocessing.SAMPLE_RATE,
            "NSAMPLES": cfg.preprocessing.NSAMPLES,
            "PADMODE": cfg.preprocessing.PADMODE,
            "UFOLD_OVERLAP": cfg.preprocessing.UFOLD_OVERLAP,
            "MAKE_RGB": cfg.preprocessing.MAKE_RGB,
            "N_MELS": cfg.preprocessing.N_MELS,
            "HOP_LENGTH": cfg.preprocessing.HOP_LENGTH,
            "N_FFT": cfg.preprocessing.N_FFT,
            "FMIN": cfg.preprocessing.FMIN,
            "FMAX": cfg.preprocessing.FMAX,
        }
        logger.debug(
            f"Initialized preprocessing config in process {mp.current_process().name}"
        )
    except Exception as e:
        logger.error(
            f"Error initializing preprocessing config in process {mp.current_process().name}: {e}"
        )
        raise


def process_single_file(
    args: Tuple[pd.Series, Path, Path],
) -> Tuple[str, bool, Optional[str], List[Dict]]:
    """Process a single audio file to generate spectrograms and metadata.

    Args:
        args: Tuple containing (row, data_folder_path, output_folder_path)

    Returns:
        Tuple of (filename, success, error, metadata)
    """
    row, src_folder, dst_folder = args
    try:
        if _process_config is None:
            raise RuntimeError("Preprocessing config not initialized in process")

        # Process the audio file
        row["filepath"] = src_folder / row["filename"]
        result = process_audio_file(row, _process_config)

        if not result["success"]:
            # Create metadata entry for failed file
            metadata = [
                {
                    "audio_file": str(row["filename"]),
                    "segment_file": None,
                    "segment_idx": -1,
                    "primary_label": row.get("primary_label", None),
                    "secondary_labels": row.get("secondary_labels", None),
                    "signal_power": None,
                    "noise_power": None,
                    "snr_db": None,
                    "rating": row.get("rating", None),
                    "silence_pct": None,
                    "is_padded": None,
                    "processing_time": result["processing_time"],
                    "success": False,
                    "error": result["error"],
                    "n_segments": 0,
                }
            ]
            return row["filename"], False, result["error"], metadata

        # Create output directory that mirrors source structure
        relative_path = Path(row["filename"]).parent
        output_dir = dst_folder / relative_path
        output_dir.mkdir(exist_ok=True, parents=True)

        # Calculate average processing time per segment
        n_segments = len(result["segments"])
        avg_processing_time = (
            result["processing_time"] / n_segments if n_segments > 0 else 0
        )

        # Process each segment and collect metadata
        metadata = []
        for segment in result["segments"]:
            # Create output path for the segment in the same structure
            segment_path = (
                output_dir
                / f"{segment['primary_label']}_{Path(row['filename']).stem}_segment_{segment['segment_idx']}.pt"
            )

            # Save the spectrogram tensor
            torch.save(segment["spectrogram"], segment_path)

            # Create metadata entry
            metadata.append(
                {
                    "audio_file": str(row["filename"]),
                    "segment_file": str(segment_path.relative_to(dst_folder)),
                    "segment_idx": segment["segment_idx"],
                    "primary_label": segment["primary_label"],
                    "secondary_labels": segment["secondary_labels"],
                    "signal_power": segment["signal_power"],
                    "noise_power": segment["noise_power"],
                    "snr_db": segment["snr_db"],
                    "rating": segment["rating"],
                    "silence_pct": segment["silence_pct"],
                    "is_padded": segment["is_padded"],
                    "processing_time": avg_processing_time,
                    "success": True,
                    "error": None,
                    "n_segments": n_segments,
                }
            )

        return row["filename"], True, None, metadata
    except Exception as e:
        logger.error(f"Error processing {row['filename']}: {e}")
        # Create metadata entry for failed file
        metadata = [
            {
                "audio_file": str(row["filename"]),
                "segment_file": None,
                "segment_idx": -1,
                "primary_label": row.get("primary_label", None),
                "secondary_labels": row.get("secondary_labels", None),
                "signal_power": None,
                "noise_power": None,
                "snr_db": None,
                "rating": row.get("rating", None),
                "silence_pct": None,
                "is_padded": None,
                "processing_time": None,
                "success": False,
                "error": str(e),
                "n_segments": 0,
            }
        ]
        return row["filename"], False, str(e), metadata


def generate_spectrograms_whole_dataset(
    df: pd.DataFrame,
    src_folder: Path,
    dst_folder: Path,
    cfg: CFG,
    n_workers: Optional[int] = None,
) -> Tuple[bool, List[Tuple[str, str]], pd.DataFrame]:
    """Generate spectrograms from audio files in parallel and create metadata.

    Args:
        df: DataFrame containing file information
        src_folder: Path to input audio files
        dst_folder: Path to output spectrograms (will maintain same folder structure)
        cfg: Configuration object
        n_workers: Number of worker processes (defaults to CPU count - 1)

    Returns:
        Tuple of (success, failed_files, metadata_df)
    """
    logger.info("Generating mel spectrograms from audio files...")
    start_time = time.time()

    # Create output directory
    dst_folder.mkdir(exist_ok=True, parents=True)

    # Set number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    logger.info(f"Using {n_workers} worker processes")

    # Prepare arguments for each file
    process_args = [(row, src_folder, dst_folder) for _, row in df.iterrows()]

    # Track failed files and collect metadata
    failed_files = []
    all_metadata = []

    # Process files in parallel with process initialization
    try:
        with mp.Pool(n_workers, initializer=init_process, initargs=(cfg,)) as pool:
            # Use imap_unordered with chunksize for better performance and progress updates
            chunksize = max(1, len(process_args) // (n_workers * 100))
            results = list(
                tqdm(
                    pool.imap_unordered(
                        process_single_file, process_args, chunksize=chunksize
                    ),
                    total=len(df),
                    desc="Generating training inputs (spectrograms) from audio files",
                    unit="files",
                )
            )
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise

    # Process results to track failed files and collect metadata
    for filename, success, error, metadata in results:
        if not success:
            failed_files.append((filename, error))
        else:
            all_metadata.extend(metadata)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(all_metadata)

    # Save metadata to CSV
    metadata_path = dst_folder / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")

    # Save failed files information
    if failed_files:
        logger.warning(
            f"Failed to process {len(failed_files)}/{len(df)} files ({len(failed_files)/len(df)*100:.1f}%)"
        )
        # Log first 10 failures
        for filename, error in failed_files[:10]:
            logger.warning(f"Failed file: {filename}, Error: {error}")
        if len(failed_files) > 10:
            logger.warning(f"... and {len(failed_files) - 10} more failures")

        # Save detailed failed files information
        failed_files_path = dst_folder / "failed_files.json"
        failed_info = {
            "total_files": len(df),
            "failed_count": len(failed_files),
            "failed_percentage": (len(failed_files) / len(df)) * 100,
            "failed_files": dict(failed_files),
        }
        with open(failed_files_path, "w") as f:
            json.dump(failed_info, f, indent=2)
        logger.info(f"Saved failed files information to {failed_files_path}")

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.1f} seconds")

    return True, failed_files, metadata_df


if __name__ == "__main__":
    cfg = CFG()
    cfg.update_machine_settings(machine="local")
    set_seed(cfg.seed)
    train_df = pd.read_csv(cfg.dirs.train_csv)
    # train_df = train_df.head(1000)  # Uncomment for testing with a small subset

    src_folder = cfg.dirs.DATA_ROOT / "train_audio_no_voice"
    dst_folder = cfg.dirs.DATA_ROOT / "train_audio_no_voice_spectrograms"
    success, failed_files, metadata_df = generate_spectrograms_whole_dataset(
        train_df,
        src_folder,
        dst_folder,
        cfg,
        n_workers=12,  # Uncomment to use single worker for debugging
    )
    logger.info(f"Success: {success}")
    logger.info(f"Metadata shape: {metadata_df.shape}")
