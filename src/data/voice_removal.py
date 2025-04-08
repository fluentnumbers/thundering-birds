import json
import logging
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

from src.data.save_ogg import save_ogg_via_wav
from src.train_notebook import CFG, set_seed
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_process_vad = None


class SileroVADRemover:
    """Remove voice from audio using Silero VAD."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the voice remover.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.device = torch.device("cpu")
        self.model = None  # Initialize to None for proper error handling

        try:
            self.model = load_silero_vad()
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD model: {e}")
            raise Exception(f"Failed to initialize Silero VAD model: {e}")

        # Set resampling parameters
        self.target_fs = 16000  # Silero VAD works best at 16kHz
        self.orig_fs = config["SAMPLE_RATE"]
        self.rsratio = self.orig_fs // self.target_fs
        self.min_speech_gap = 1.5  # seconds

        # Initialize resampler
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.orig_fs,
            new_freq=self.target_fs,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="sinc_interp_kaiser",
            beta=8.555504641634386,
        ).cpu()  # Force resampler to CPU

    def __call__(
        self, audio: torch.Tensor
    ) -> Tuple[torch.Tensor, bool, List[Dict[str, float]]]:
        """Remove voice from audio if detected.

        Args:
            audio: Audio tensor of shape (n_samples,)

        Returns:
            Tuple of (processed_audio, has_voice, timestamps)
        """
        if self.model is None:
            logger.warning("Silero VAD model not initialized, returning original audio")
            return audio, False, []

        with torch.no_grad():
            # Resample audio to 16kHz for VAD
            audio_16k = self.resampler(audio)

            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                audio_16k, self.model, sampling_rate=self.target_fs, return_seconds=True
            )

            if not speech_timestamps:
                return audio, False, []

            merged_timestamps = []

            if speech_timestamps:
                current_segment = speech_timestamps[0]

                for i in range(1, len(speech_timestamps)):
                    if (
                        speech_timestamps[i]["start"] - current_segment["end"]
                        < self.min_speech_gap
                    ):
                        current_segment["end"] = speech_timestamps[i]["end"]
                    else:
                        merged_timestamps.append(current_segment)
                        current_segment = speech_timestamps[i]

                merged_timestamps.append(current_segment)
                speech_timestamps = merged_timestamps

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

            return processed_audio, True, speech_timestamps


def init_process(cfg: CFG) -> None:
    """Initialize the VAD model for each process.

    Args:
        cfg: Configuration object containing preprocessing parameters
    """
    global _process_vad
    try:
        _process_vad = SileroVADRemover(cfg.preprocessing)
        if _process_vad.model is None:
            raise RuntimeError("Failed to initialize VAD model")
        logger.debug(f"Initialized VAD model in process {mp.current_process().name}")
    except Exception as e:
        logger.error(
            f"Error initializing VAD model in process {mp.current_process().name}: {e}"
        )
        raise


def process_single_file(
    args: Tuple[pd.Series, Path, Path],
) -> Tuple[str, bool, bool, Optional[str], Optional[List[Dict[str, float]]]]:
    """Process a single audio file for voice removal.

    Args:
        args: Tuple containing (row, data_folder_path, output_folder_path)

    Returns:
        Tuple of (filename, has_voice, success, error, timestamps)
    """
    row, src_dir, dst_dir = args
    try:
        if _process_vad is None:
            raise RuntimeError("VAD model not initialized in process")

        audio_path = Path(src_dir) / row["filename"]
        output_file_path = dst_dir / row["filename"]
        output_file_path.parent.mkdir(exist_ok=True, parents=True)

        # Load audio with timeout
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Error loading audio file {row['filename']}: {e}")
            return row["filename"], False, False, f"Error loading audio: {str(e)}", None

        # Process audio with timeout
        try:
            audio_tensor = waveform.squeeze(0)  # Remove channel dimension
            audio, has_voice, timestamps = _process_vad(audio_tensor)
        except Exception as e:
            logger.error(f"Error processing audio {row['filename']}: {e}")
            return (
                row["filename"],
                False,
                False,
                f"Error processing audio: {str(e)}",
                None,
            )

        # Save output with timeout
        try:
            if has_voice:
                audio_np = audio.cpu().numpy().astype(np.float32)
                save_ogg_via_wav(_process_vad.orig_fs, audio_np, output_file_path)
            else:
                shutil.copy(str(audio_path), str(output_file_path))
        except Exception as e:
            logger.error(f"Error saving output for {row['filename']}: {e}")
            return row["filename"], False, False, f"Error saving output: {str(e)}", None

        return row["filename"], has_voice, True, None, timestamps
    except Exception as e:
        logger.error(f"Unexpected error processing {row['filename']}: {e}")
        return row["filename"], False, False, str(e), None


def remove_voice_whole_dataset(
    df: pd.DataFrame,
    src_dir: Path,
    dst_dir: Path,
    cfg: CFG,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Remove voice from whole dataset using Silero VAD in parallel.

    Args:
        df: DataFrame containing file information
        data_folder_path: Path to input audio files
        cfg: Configuration object
        n_workers: Number of worker processes (defaults to CPU count - 1)

    Returns:
        DataFrame with updated has_voice column
    """
    logger.info("Removing voice from whole dataset using Silero VAD...")
    start_time = time.time()
    dst_dir.mkdir(exist_ok=True, parents=True)
    # Check which files already exist in the output directory
    existing_files = set()
    for file_path in dst_dir.rglob("*"):
        if file_path.is_file():
            # Get relative path from output_folder_path and normalize separators
            rel_path = file_path.relative_to(dst_dir)
            # Convert to string and normalize path separators
            normalized_path = str(rel_path).replace("\\", "/")
            existing_files.add(normalized_path)

    # Filter out files that already exist
    files_to_process = []
    for _, row in df.iterrows():
        # Normalize the filename path
        normalized_filename = str(row["filename"]).replace("\\", "/")
        if normalized_filename not in existing_files:
            files_to_process.append(row)
        else:
            logger.debug(f"Skipping {normalized_filename} - already processed")

    if not files_to_process:
        logger.info("All files have already been processed")
        return df

    logger.info(
        f"Found {len(files_to_process)} files to process out of {len(df)} total files"
    )

    # Set number of workers (reduce if memory issues)
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)  # Leave 1 core free
    logger.info(f"Using {n_workers} worker processes")

    # Prepare arguments for each file
    process_args = [(row, src_dir, dst_dir) for row in files_to_process]

    # Initialize DataFrame column
    df["has_voice"] = False

    # Process files in parallel with process initialization
    try:
        with mp.Pool(
            n_workers,
            initializer=init_process,
            initargs=(cfg,),
            maxtasksperchild=100,  # Restart workers after 100 tasks
        ) as pool:
            # Use imap_unordered with chunksize for better performance and progress updates
            chunksize = max(
                1, len(process_args) // (n_workers * 10)
            )  # Smaller chunksize
            results = list(
                tqdm(
                    pool.imap_unordered(
                        process_single_file, process_args, chunksize=chunksize
                    ),
                    total=len(files_to_process),
                    desc="Removing voice",
                    unit="files",
                )
            )
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise

    # Process results
    failed_files = []
    voice_info = {}
    for filename, has_voice, success, error, timestamps in results:
        if success:
            df.loc[df["filename"] == filename, "has_voice"] = has_voice
            voice_info[filename] = {
                "has_voice": has_voice,
                "timestamps": timestamps if timestamps else [],
            }
        else:
            failed_files.append((filename, error))
            logger.error(f"Error processing {filename}: {error}")

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.1f} seconds")
    logger.info(f"Processed files saved to {dst_dir}")

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        # Save failed files information
        failed_files_path = dst_dir / "failed_files.json"
        with open(failed_files_path, "w") as f:
            json.dump(dict(failed_files), f, indent=2)
        logger.info(f"Saved failed files information to {failed_files_path}")

    # Save voice detection results and timestamps to a JSON file
    voice_info_path = dst_dir / "voice_detection_results.json"
    logger.info(f"Saving voice detection results to {voice_info_path}")
    with open(voice_info_path, "w") as f:
        json.dump(voice_info, f, indent=2)

    logger.info(
        f"Found voice in {df['has_voice'].sum()} out of {len(df)} files ({df['has_voice'].mean()*100:.1f}%)"
    )
    return df


if __name__ == "__main__":
    cfg = CFG()
    set_seed(cfg.seed)
    train_df = pd.read_csv(cfg.train_csv)
    # train_df = train_df[train_df["primary_label"] == "65373"]
    # train_df = train_df[:100]

    REMOVE_VOICE = True
    if REMOVE_VOICE:
        df = remove_voice_whole_dataset(
            train_df,
            cfg.DATA_ROOT / "train_audio",
            cfg.DATA_ROOT / "train_audio_no_voice",
            cfg,
            # n_workers=11,
        )
