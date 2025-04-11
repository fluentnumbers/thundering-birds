import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data.preprocessing import process_audio_file
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, species_ids, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.species_ids = species_ids
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # Memory management settings
        self.max_memory_gb = cfg.training.MAX_CACHE_MEMORY_GB
        self.current_memory_usage = mp.Value("d", 0.0)  # Shared memory counter
        self.files_in_memory = mp.Manager().dict()  # Shared dictionary
        self.segments_loaded_this_epoch = (
            mp.Manager().list()
        )  # Shared list (used as set)

        if "filepath" not in self.df.columns:
            self.df["filepath"] = self.cfg.dirs.train_datadir + "/" + self.df.filename

        if "samplename" not in self.df.columns:
            self.df["samplename"] = self.df.filename.map(
                lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
            )

        # Create cache directory using config path
        self.cache_dir = Path(self.cfg.dirs.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize file segments tracking
        self.total_segments = 0
        self.file_indices = []  # List of (filename, segment_idx) pairs

        # Vectorized check for cache existence
        cache_paths = {
            row["filename"]: self._get_cache_path(row["filename"])
            for _, row in self.df.iterrows()
        }
        cache_exists = {
            filename: path.exists() for filename, path in cache_paths.items()
        }
        logger.info(
            f"Cache exists for {sum(cache_exists.values())} files out of {len(self.df)}"
        )

        # Create or load cache status file
        cache_status_file = self.cache_dir / f"cache_status_full.json"
        CACHE_LOADED = False
        if cache_status_file.exists():
            # Load existing cache status
            with open(cache_status_file, "r") as f:
                cache_status = json.load(f)

            # Update file_indices from cache status
            for _, row in self.df.iterrows():
                filename = row["filename"]
                if filename in cache_status and cache_exists[filename]:
                    num_segments = cache_status[filename]["num_segments"]
                    self.file_indices.extend(
                        [(filename, i) for i in range(num_segments)]
                    )
                    self.total_segments += num_segments
            CACHE_LOADED = True

        cached_files = []
        files_to_process = []
        if not CACHE_LOADED:
            cache_status = {}
            # Split files into cached and to-process
            time_start = time.time()

            for _, row in tqdm(
                self.df.iterrows(),
                total=len(self.df),
                desc="Loading cached files",
                unit="file",
            ):
                filename = row["filename"]
                if cache_exists[filename]:
                    try:
                        # Load just the first segment to get the count
                        segments = torch.load(cache_paths[filename])
                        self.total_segments += len(segments)
                        self.file_indices.extend(
                            [(filename, i) for i in range(len(segments))]
                        )
                        cached_files.append(filename)
                        cache_status[filename] = {
                            "num_segments": len(segments),
                            "last_modified": os.path.getmtime(cache_paths[filename]),
                            "primary_label": row["primary_label"],
                        }
                    except Exception as e:
                        logger.warning(
                            f"Failed to load cache for {cache_paths[filename]}, will reprocess: {e}"
                        )
                        files_to_process.append((filename, row, self.cfg))
                else:
                    files_to_process.append((filename, row, self.cfg))

        if not files_to_process:
            logger.info(
                f"All files for mode {self.mode} are already cached, no processing needed"
            )
            return

        logger.info(
            f"Processing {len(files_to_process)} files out of {len(self.df)} total files"
        )
        time_start = time.time()
        with mp.Pool(
            processes=min(self.cfg.preprocessing.NUM_WORKERS, mp.cpu_count())
        ) as pool:
            # Process files in batches to manage memory
            batch_size = 100
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i : i + batch_size]
                results = list(
                    tqdm(
                        pool.imap(process_file_worker, batch),
                        total=len(batch),
                        desc=f"Processing batch {i//batch_size + 1}/{(len(files_to_process) + batch_size - 1)//batch_size}",
                        unit=f"{batch_size} files",
                    )
                )

                # Save results and update indices
                for filename, segments in results:
                    if segments:  # Only process if we got valid segments
                        # Save to cache
                        self._save_to_cache(filename, segments)

                        # Update total segments and indices
                        self.total_segments += len(segments)
                        self.file_indices.extend(
                            [(filename, i) for i in range(len(segments))]
                        )

                # Clear memory after each batch
                gc.collect()
        logger.info(
            f"Processed {len(files_to_process)} files in {time.time() - time_start:.1f}s"
        )

    def _get_cache_path(self, filename):
        """Get the cache path for a file"""
        # Extract class name from filename (part before first dash)
        class_name = filename.split("/")[0]
        # Create path with class subdirectory
        return (
            self.cache_dir
            / class_name
            / f"{Path(filename.replace('/', '-')).with_suffix('.pt')}"
        )

    def _load_from_cache(self, filename):
        """Load segments from cache if available"""
        cache_path = self._get_cache_path(filename)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_path}: {e}")
                return None
        return None

    def _save_to_cache(self, filename, segments):
        """Save segments to cache"""
        cache_path = self._get_cache_path(filename)
        # Create parent directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(segments, cache_path)
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_path}: {e}")

    def _estimate_segment_memory(self, segments):
        """Estimate memory usage of segments in bytes"""
        total_bytes = 0
        for segment in segments:
            if isinstance(segment, dict) and "spectrogram" in segment:
                total_bytes += (
                    segment["spectrogram"].element_size()
                    * segment["spectrogram"].nelement()
                )
        return total_bytes

    def _load_segments_to_memory(self, filename):
        """Load segments into memory if possible"""
        if filename in self.files_in_memory:
            return self.files_in_memory[filename][0]

        segments = self._load_from_cache(filename)
        if segments is None:
            return None

        # Estimate memory usage
        segment_size = self._estimate_segment_memory(segments)

        with self.current_memory_usage.get_lock():
            if (
                self.current_memory_usage.value + segment_size
                <= self.max_memory_gb * 1024**3
            ):
                self.files_in_memory[filename] = (segments, segment_size)
                self.current_memory_usage.value += segment_size
                if filename not in self.segments_loaded_this_epoch:
                    self.segments_loaded_this_epoch.append(filename)
                return segments
            else:
                return segments

    def cleanup_after_epoch(self):
        """Clean up memory after each epoch"""
        # Clear all segments from memory
        self.files_in_memory.clear()
        with self.current_memory_usage.get_lock():
            self.current_memory_usage.value = 0
        self.segments_loaded_this_epoch[:] = []  # Clear the list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        # Get the file and segment index from our pre-computed list
        filename, segment_idx = self.file_indices[idx]

        # Try to get segments from memory or load them
        segments = self._load_segments_to_memory(filename)
        if segments is None:
            logger.error(f"Failed to load segments for {filename}")
            return {
                "melspec": torch.zeros((1, 224, 224), dtype=torch.float32),
                "target": torch.zeros(self.num_classes, dtype=torch.float32),
                "filename": filename,
                "segment_idx": 0,
            }

        segment = segments[segment_idx]
        row = self.df[self.df["filename"] == filename].iloc[0]

        target = self.encode_label(row["primary_label"])

        if "secondary_labels" in row and row["secondary_labels"] not in [
            "[" "]",
            "['']",
            None,
            np.nan,
        ]:
            if isinstance(row["secondary_labels"], str):
                secondary_labels = eval(row["secondary_labels"])
            else:
                secondary_labels = row["secondary_labels"]

            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return {
            "melspec": segment["spectrogram"],
            "target": torch.tensor(target, dtype=torch.float32),
            "filename": filename,
            "segment_idx": segment_idx,
        }

    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target


def process_file_worker(args):
    """Worker function for processing files in parallel"""
    filename, row, cfg = args
    try:
        result = process_audio_file(row, cfg.preprocessing)
        if result["success"] and result["segments"]:
            # Convert segments to tensors
            tensor_segments = []
            for segment in result["segments"]:
                if isinstance(segment["spectrogram"], np.ndarray):
                    tensor_segments.append(
                        {
                            "spectrogram": torch.from_numpy(segment["spectrogram"]),
                            "filename": segment.get("filename", row["filename"]),
                            "segment_idx": segment.get("segment_idx", 0),
                        }
                    )
                else:
                    tensor_segments.append(segment)
            return filename, tensor_segments
    except Exception as e:
        logger.warning(f"Failed to process {row['samplename']}: {e}")

    # Return zero segment if processing fails
    zero_segment = {
        "spectrogram": torch.zeros((1, 224, 224), dtype=torch.float32),
        "filename": row["filename"],
        "segment_idx": 0,
    }
    return filename, [zero_segment]


def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms and segments"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}

    result = {key: [] for key in batch[0].keys()}

    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    for key in result:
        if key == "target" and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == "melspec" and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
        elif key == "segment_idx":
            result[key] = torch.tensor(result[key])

    return result
