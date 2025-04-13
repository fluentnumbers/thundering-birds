import gc
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data.preprocessing import process_audio_file
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_segments_info(cfg):
    """Load and filter metadata for audio segments.

    Args:
        cfg: Configuration object with paths and settings

    Returns:
        DataFrame containing filtered metadata for audio segments
    """
    df_cache = pd.read_csv(Path(cfg.dirs.cache_dir) / "metadata.csv")
    # Convert snr_db to numeric, coercing errors to NaN
    df_cache["snr_db"] = pd.to_numeric(df_cache["snr_db"], errors="coerce")

    df_cache = df_cache[df_cache["silence_pct"] < 100]
    df_cache = df_cache[~df_cache["rating"].isin([0.5, 1.0, 1.5])]
    df_cache = df_cache[df_cache["signal_power"] > 0.01]
    df_cache = df_cache[df_cache["snr_db"] > 0]

    # Assert that we have the expected number of classes (206)
    unique_classes = df_cache["primary_label"].nunique()
    if unique_classes != 206:
        raise ValueError(f"Expected 206 unique classes, but found {unique_classes}")

    # Log statistics about the filtered dataset
    logger.info(
        f"After filtering segmetns metadata: {len(df_cache)} segments from {df_cache['audio_file'].nunique()} files and {df_cache['primary_label'].nunique()} classes"
    )

    return df_cache


class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, species_ids, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.class_ids = species_ids
        self.num_classes = len(self.class_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_ids)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.classes_present_in_df = self.df["primary_label"].unique().tolist()

        # Initialize sample usage tracking
        self.sample_usage = {}  # Regular dictionary
        self.current_epoch = 0

        self.cache_dir = Path(self.cfg.dirs.cache_dir)
        # loading prefiltered metadata about cached segments
        self.metadata_df = load_segments_info(self.cfg)

        # df maybe train or test set,  so, first keep only files that are present in the metadata (not completely filtered out due to low SNR, etc.)
        self.df = self.df[self.df["filename"].isin(self.metadata_df["audio_file"])]
        self.df = self.df.reset_index(drop=True)

        # now remove metadata rows, which represent files which are not in the df
        self.metadata_df = self.metadata_df[
            self.metadata_df["audio_file"].isin(self.df["filename"].unique())
        ]
        self.metadata_df = self.metadata_df.reset_index(drop=True)
        self.metadata_df["index"] = range(len(self.metadata_df))

        self.metadata_df["cache_path"] = self.metadata_df["segment_file"].map(
            lambda x: Path(self.cache_dir / x)
        )

        self.metadata_lookup_files = {}
        self.metadata_lookup_classes = {}

        # Build lookup for files
        unique_files = self.metadata_df["audio_file"].unique()
        for f in unique_files:
            file_df = self.metadata_df[self.metadata_df["audio_file"] == f]
            self.metadata_lookup_files[f] = {
                "n_segments": len(file_df),
                "indices": file_df["index"].tolist(),
                "cache_paths": file_df["cache_path"].tolist(),
            }

        # Build lookup for classes
        for c in self.class_ids:
            class_df = self.metadata_df[self.metadata_df["primary_label"] == c]
            self.metadata_lookup_classes[c] = {
                "n_segments": len(class_df),
                "n_files": len(class_df["audio_file"].unique()),
                "indices": class_df["index"].tolist(),
                "files": class_df["audio_file"].unique().tolist(),
                "cache_paths": class_df["cache_path"].tolist(),
            }

        # Set fixed number of samples per epoch
        self.samples_per_epoch = cfg.training.SAMPLES_PER_EPOCH
        if self.samples_per_epoch is None or self.samples_per_epoch > len(
            self.metadata_df
        ):
            self.samples_per_epoch = len(self.metadata_df)

        # Initialize random state for reproducibility
        self.rng = np.random.RandomState(cfg.seed)

        # Pre-compute class weights for sampling
        self.class_weights = np.ones(len(self.class_ids)) / len(self.class_ids)
        # Create a mask for classes not present in the current dataset
        mask = ~np.isin(self.class_ids, self.classes_present_in_df)
        self.class_weights[mask] = 0
        # Normalize weights to sum to 1
        if self.class_weights.sum() > 0:
            self.class_weights = self.class_weights / self.class_weights.sum()
        else:
            raise ValueError("No valid classes found in the dataset")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # First sample a class uniformly
        class_idx = self.rng.choice(len(self.class_ids), p=self.class_weights)
        class_label = self.class_ids[class_idx]

        # Get all segments for this class
        class_info = self.metadata_lookup_classes[class_label]
        n_segments = class_info["n_segments"]

        # Sample a random segment from this class
        segment_idx = self.rng.randint(0, n_segments)
        cache_path = class_info["cache_paths"][segment_idx]

        # Track sample usage
        sample_key = f"{class_label}_{segment_idx}"
        if sample_key not in self.sample_usage[self.current_epoch]:
            self.sample_usage[self.current_epoch][sample_key] = 0
        self.sample_usage[self.current_epoch][sample_key] += 1

        # Log if sample is used more than once
        if self.sample_usage[self.current_epoch][sample_key] > 1:
            logger.debug(
                f"Sample {sample_key} used {self.sample_usage[self.current_epoch][sample_key]} times in epoch {self.current_epoch}"
            )

        # Load the segment
        try:
            spectrogram = torch.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load segment from {cache_path}: {e}")
            spectrogram = torch.zeros((1, 224, 224), dtype=torch.float32)

        # Create target vector
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[class_idx] = 1.0

        return {
            "melspec": spectrogram,
            "target": target,
            "class_label": class_label,
            "segment_idx": segment_idx,
        }

    def set_epoch(self, epoch):
        """Set random seed for this epoch and reset usage tracking"""
        self.current_epoch = epoch
        self.rng = np.random.RandomState(self.cfg.seed + epoch)
        self.sample_usage[epoch] = {}  # Reset usage tracking for new epoch

    def get_usage_stats(self, epoch=None):
        """Get statistics about sample usage for a specific epoch or all epochs"""
        if epoch is not None:
            usage = dict(self.sample_usage.get(epoch, {}))  # Convert to regular dict
            if not usage:
                return None

            # Initialize per-class statistics
            class_stats = {}
            for class_label in self.class_ids:
                class_samples = {
                    k: v for k, v in usage.items() if k.startswith(f"{class_label}_")
                }
                class_stats[class_label] = {
                    "total_samples": len(class_samples),
                    "unique_samples": len(set(class_samples.keys())),
                    "max_usage": max(class_samples.values()) if class_samples else 0,
                    "min_usage": min(class_samples.values()) if class_samples else 0,
                    "avg_usage": (
                        sum(class_samples.values()) / len(class_samples)
                        if class_samples
                        else 0
                    ),
                    "samples_used_once": sum(
                        1 for v in class_samples.values() if v == 1
                    ),
                    "samples_used_multiple": sum(
                        1 for v in class_samples.values() if v > 1
                    ),
                }

            # Overall statistics
            overall_stats = {
                "total_samples": len(usage),
                "unique_samples": len(set(usage.keys())),
                "max_usage": max(usage.values()),
                "min_usage": min(usage.values()),
                "avg_usage": sum(usage.values()) / len(usage),
                "samples_used_once": sum(1 for v in usage.values() if v == 1),
                "samples_used_multiple": sum(1 for v in usage.values() if v > 1),
                "class_stats": class_stats,
            }

            return overall_stats
        else:
            return {e: self.get_usage_stats(e) for e in self.sample_usage.keys()}

    def log_usage_stats(self, epoch=None):
        """Log detailed usage statistics for an epoch"""
        stats = self.get_usage_stats(epoch)
        if stats is None:
            logger.warning(f"No usage statistics available for epoch {epoch}")
            return

        logger.info(f"\n{'='*50}")
        logger.info(
            f"Usage Statistics for Epoch {epoch if epoch is not None else 'All'}"
        )
        logger.info(f"{'='*50}")

        # Log overall statistics
        logger.info("\nOverall Statistics:")
        logger.info(f"Total samples used: {stats['total_samples']}")
        logger.info(f"Unique samples: {stats['unique_samples']}")
        logger.info(f"Max usage per sample: {stats['max_usage']}")
        logger.info(f"Min usage per sample: {stats['min_usage']}")
        logger.info(f"Avg usage per sample: {stats['avg_usage']:.2f}")
        logger.info(f"Samples used once: {stats['samples_used_once']}")
        logger.info(f"Samples used multiple times: {stats['samples_used_multiple']}")

        # Log per-class statistics
        logger.info("\nPer-Class Statistics:")
        for class_label, class_stat in stats["class_stats"].items():
            logger.info(f"\nClass: {class_label}")
            logger.info(f"  Total samples: {class_stat['total_samples']}")
            logger.info(f"  Unique samples: {class_stat['unique_samples']}")
            logger.info(f"  Max usage: {class_stat['max_usage']}")
            logger.info(f"  Min usage: {class_stat['min_usage']}")
            logger.info(f"  Avg usage: {class_stat['avg_usage']:.2f}")
            logger.info(f"  Used once: {class_stat['samples_used_once']}")
            logger.info(f"  Used multiple: {class_stat['samples_used_multiple']}")

        logger.info(f"{'='*50}\n")


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
                            "rating": segment.get("rating", float("nan")),
                            "signal_power": segment.get("signal_power", float("nan")),
                            "noise_power": segment.get("noise_power", float("nan")),
                            "snr_db": segment.get("snr_db", float("nan")),
                            "silence_pct": segment.get("silence_pct", 0),
                            "is_padded": segment.get("is_padded", False),
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
        "rating": float("nan"),
        "signal_power": float("nan"),
        "noise_power": float("nan"),
        "snr_db": float("nan"),
        "silence_pct": 0,
        "is_padded": False,
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
