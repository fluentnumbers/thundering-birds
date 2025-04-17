import ctypes
import gc
import json
import multiprocessing as mp
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.auto import tqdm

from src.data.preprocessing import process_audio_file
from src.data.processing import align_df_and_metadata, normalize_values_by_group
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_cache_metadata(cfg):
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

    # Calculate the number of segments per class
    class_segment_counts = df_cache["primary_label"].value_counts()

    # Find bottom 10% of classes by number of segments
    bottom_10_percent_classes = class_segment_counts[
        class_segment_counts <= np.percentile(class_segment_counts.values, 10)
    ].index.tolist()

    # Find top 10% of classes by number of segments
    top_10_percent_classes = class_segment_counts[
        class_segment_counts >= np.percentile(class_segment_counts.values, 90)
    ].index.tolist()

    logger.info(
        f"Bottom 10% classes (excluded from filtering): {len(bottom_10_percent_classes)}"
    )
    logger.info(f"Top 10% classes (stricter filtering): {len(top_10_percent_classes)}")

    # Create a copy of the dataframe for filtering
    df_filtered = df_cache.copy()

    # Apply stricter filtering only to top 10% classes
    stricter_filter_mask = (
        (df_filtered["primary_label"].isin(top_10_percent_classes))
        & (df_filtered["signal_power"] > 0.005)  # Stricter signal power threshold
        & (df_filtered["snr_db"] > 0.1)  # Stricter SNR threshold
        & ~(df_filtered["rating"].isin([0.5, 1, 1.5]))
    )

    # Keep all segments from bottom 10% classes
    keep_mask = df_filtered["primary_label"].isin(bottom_10_percent_classes)

    # Apply normal filtering to the middle 80% classes
    middle_classes_mask = (
        ~(
            df_filtered["primary_label"].isin(bottom_10_percent_classes)
            | df_filtered["primary_label"].isin(top_10_percent_classes)
        )
        & (df_filtered["signal_power"] > 0.002)
        & (df_filtered["snr_db"] > 0)
        & ~(df_filtered["rating"].isin([0.5, 1, 1.5]))
    )

    # Combine the masks to get the final dataframe
    df_cache = pd.concat(
        [
            df_filtered[keep_mask],  # Bottom 10% (no filtering)
            df_filtered[middle_classes_mask],  # Middle 80% (normal filtering)
            df_filtered[
                stricter_filter_mask & ~keep_mask
            ],  # Top 10% (stricter filtering)
        ]
    )

    # Log the filtering results
    logger.info(f"Original segments count: {len(df_filtered)}")
    logger.info(f"After class-specific filtering: {len(df_cache)}")
    logger.info(f"Segments from bottom 10% classes: now {sum(keep_mask)}, before ")
    logger.info(
        f"Segments from top 10% classes after filtering: now {sum(stricter_filter_mask & ~keep_mask)}, before "
    )

    # Assert that we have the expected number of classes (206)
    unique_classes = df_cache["primary_label"].nunique()
    if unique_classes != 206:
        raise ValueError(f"Expected 206 unique classes, but found {unique_classes}")

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
        self.segments_weights_by_rating = (df["rating"].values + 1) / 6

        self.class_weights = self.compute_class_weights()
        self.rng = np.random.RandomState(cfg.seed)
        self.cache_dir = Path(self.cfg.dirs.cache_dir)
        self.df, self.metadata_df = align_df_and_metadata(
            self.df, load_cache_metadata(cfg)
        )
        self.metadata_df["index"] = range(len(self.metadata_df))

        # Fix path handling - ensure paths are relative to cache_dir
        self.metadata_df["cache_path"] = self.metadata_df.apply(
            lambda row: self.cache_dir
            / row["primary_label"]
            / f"{row['primary_label']}_{Path(row['filename']).stem}_segment_{row['segment_idx']}.pt",
            axis=1,
        )

        # Initialize tracking for all segment usage (only for training)
        if self.mode == "train":
            self.segment_usage_stats_per_epoch = {}

            # Create shared memory structures for segment usage tracking
            self._shared_segment_counts = {}
            self._shared_segment_indices = {}
            self.segment_usage_stats_per_class = {}

            for class_label in self.class_ids:
                class_segments = self.metadata_df[
                    self.metadata_df["primary_label"] == class_label
                ]["segment_file"].unique()

                # Create shared array for this class's segment counts
                n_segments = len(class_segments)
                shared_arr = mp.Array(ctypes.c_uint32, n_segments)

                # Create mapping from segment file to index in shared array
                segment_to_idx = {seg: idx for idx, seg in enumerate(class_segments)}

                self._shared_segment_counts[class_label] = shared_arr
                self._shared_segment_indices[class_label] = segment_to_idx

                # Create view dictionary that maps segment files to their usage counts
                self.segment_usage_stats_per_class[class_label] = {
                    segment_file: 0 for segment_file in class_segments
                }

            self.segment_usage_stats_lock = mp.Lock()

            # Initialize sample usage tracking with per-class locks
            self._class_locks = {
                class_label: mp.Lock() for class_label in self.class_ids
            }

            # Generate one permutation per class that we'll use across epochs
            self._class_permutations = {}
            self._class_permutation_indices = {}

            # Track how many times we've had to regenerate permutations
            self._permutation_reset_counts = {
                class_label: 0 for class_label in self.class_ids
            }

        # For validation, create a deterministic order of all segments
        if self.mode != "train":
            self.all_segment_indices = []
            for class_label in self.class_ids:
                class_segments = self.metadata_df[
                    self.metadata_df["primary_label"] == class_label
                ]
                for _, row in class_segments.iterrows():
                    self.all_segment_indices.append(
                        {
                            "class_label": class_label,
                            "segment_file": row["segment_file"],
                            "cache_path": row["cache_path"],
                            "segment_idx": row["segment_idx"],
                        }
                    )
        else:
            # Initialize samples per epoch for training
            self.samples_per_epoch = (
                cfg.training.SAMPLES_PER_EPOCH
                if cfg.training.SAMPLES_PER_EPOCH < len(self.metadata_df)
                else len(self.metadata_df)
            )

            self.metadata_lookup_files = {}
            self.metadata_lookup_classes = {}

            # Build lookup for files
            unique_files = self.metadata_df["filename"].unique()
            for f in unique_files:
                file_df = self.metadata_df[self.metadata_df["filename"] == f]
                self.metadata_lookup_files[f] = {
                    "n_segments": len(file_df),
                    "indices": file_df["index"].tolist(),
                    "cache_paths": file_df["cache_path"].tolist(),
                }

            # Build lookup for classes and initialize permutations
            for c in self.class_ids:
                class_df = self.metadata_df[self.metadata_df["primary_label"] == c]
                n_segments = len(class_df)
                self.metadata_lookup_classes[c] = {
                    "n_segments": n_segments,
                    "n_files": len(class_df["filename"].unique()),
                    "indices": class_df["index"].tolist(),
                    "files": class_df["filename"].unique().tolist(),
                    "cache_paths": class_df["cache_path"].tolist(),
                    "segments": class_df["segment_file"].tolist(),
                }
                # Initialize permutation for this class
                self._class_permutations[c] = self.rng.permutation(n_segments)
                self._class_permutation_indices[c] = 0

            # Create signal power lookup using normalized per-file values
        self.signal_power_lookup = dict(
            zip(
                self.metadata_df["cache_path"].astype(str),
                normalize_values_by_group(
                    self.metadata_df["signal_power"],
                    self.metadata_df["filename"],
                ),
            )
        )

        logger.info(
            f"{self.mode} dataset loaded with {len(self.metadata_df['filename'].unique())} files, {len(self.metadata_df)} samples and {len(self.class_ids)} classes;"
            + (
                f" {self.samples_per_epoch} samples per epoch ({self.samples_per_epoch // self.cfg.training.BATCH_SIZE} batches ({self.cfg.training.BATCH_SIZE}) per epoch)"
                if self.mode == "train"
                else ""
            )
        )

    def get_segment_usage_stats(self, epoch) -> Dict:
        """Get data for plotting sampling histogram."""
        usage_stats = {}
        with self.segment_usage_stats_lock:
            for class_label in self.class_ids:
                # Update the dictionary view with current shared memory values
                shared_arr = self._shared_segment_counts[class_label]
                segment_to_idx = self._shared_segment_indices[class_label]

                for segment_file, idx in segment_to_idx.items():
                    self.segment_usage_stats_per_class[class_label][segment_file] = (
                        shared_arr[idx]
                    )

                class_usage = np.array(
                    list(self.segment_usage_stats_per_class[class_label].values())
                )
                usage_stats[class_label] = {
                    "mean_usage": np.mean(class_usage),
                    "max_usage": np.max(class_usage),
                    "unused_segments": (class_usage == 0).sum(),
                    "total_segments": self.metadata_lookup_classes[class_label][
                        "n_segments"
                    ],
                    "total_segments_drawn": np.sum(class_usage),
                }
            self.segment_usage_stats_per_epoch[epoch] = usage_stats

            histogram_data = {
                "classes": self.class_ids,
                "total_segments_drawn": np.array(
                    [usage_stats[c]["total_segments_drawn"] for c in self.class_ids]
                ),
                "total_segments": np.asarray(
                    [usage_stats[c]["total_segments"] for c in self.class_ids]
                ),
                "mean_usage_per_class": np.asarray(
                    [usage_stats[c]["mean_usage"] for c in self.class_ids]
                ),
                "max_usage_per_class": np.asarray(
                    [usage_stats[c]["max_usage"] for c in self.class_ids]
                ),
                "unused_segments_per_class": np.asarray(
                    [usage_stats[c]["unused_segments"] for c in self.class_ids]
                ),
            }
            return histogram_data

    def compute_class_weights(self):
        class_weights = np.ones(len(self.class_ids)) / len(self.class_ids)
        mask = ~np.isin(self.class_ids, self.classes_present_in_df)
        class_weights[mask] = 0
        if class_weights.sum() > 0:
            class_weights = class_weights / class_weights.sum()
        else:
            raise ValueError("No valid classes found in the dataset")
        return class_weights

    def get_next_segment_idx(self, class_label):
        """Get next unused sample index from the class's permutation"""
        with self._class_locks[class_label]:
            idx = self._class_permutation_indices[class_label]
            n_segments = self.metadata_lookup_classes[class_label]["n_segments"]

            # If we've used all segments, generate new permutation with a different seed
            if idx >= n_segments:
                # Create a new seed combining class label hash, reset count, and base seed
                combined_seed = (
                    self.cfg.seed
                    + (hash(class_label) & 0xFFFFFFFF)
                    + self._permutation_reset_counts[class_label] * 65537
                ) & 0xFFFFFFFF

                # Create a new RNG instance with the combined seed
                permutation_rng = np.random.RandomState(combined_seed)
                self._class_permutations[class_label] = permutation_rng.permutation(
                    n_segments
                )
                idx = 0
                self._permutation_reset_counts[class_label] += 1

            # Get segment index from permutation and increment position
            segment_idx = self._class_permutations[class_label][idx]
            self._class_permutation_indices[class_label] = idx + 1

            return segment_idx

    def __len__(self):
        """Return dataset length based on mode"""
        if self.mode == "train":
            return self.samples_per_epoch
        else:
            return len(self.all_segment_indices)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Existing training logic
            class_label = self.rng.choice(self.class_ids, p=self.class_weights)
            segment_idx = self.get_next_segment_idx(class_label)
            cache_path = self.metadata_lookup_classes[class_label]["cache_paths"][
                segment_idx
            ]
            segment_file = self.metadata_lookup_classes[class_label]["segments"][
                segment_idx
            ]

            # Update usage tracking with thread safety using shared memory
            with self.segment_usage_stats_lock:
                shared_arr = self._shared_segment_counts[class_label]
                segment_idx_in_arr = self._shared_segment_indices[class_label][
                    segment_file
                ]
                shared_arr[segment_idx_in_arr] += 1
                num_times_used = shared_arr[segment_idx_in_arr]
        else:
            # Validation mode: use deterministic ordering
            segment_info = self.all_segment_indices[idx]
            class_label = segment_info["class_label"]
            segment_file = segment_info["segment_file"]
            cache_path = segment_info["cache_path"]
            segment_idx = segment_info["segment_idx"]

        spectrogram = torch.load(cache_path)

        if self.mode == "train" and num_times_used > 1:
            spectrogram = self.apply_spec_augmentations(
                spectrogram, num_times_used=num_times_used
            )

        # Create target vectors
        primary_target, secondary_target, target = self._create_targets(
            self.metadata_df[self.metadata_df["cache_path"] == cache_path].iloc[0]
        )

        return {
            "melspec": spectrogram,
            "target": target,
            "primary_target": primary_target,
            "secondary_target": secondary_target,
            "class_label": class_label,
            "segment_file": segment_file,
            "segment_idx": segment_idx,
            "signal_power_weight": self.signal_power_lookup.get(str(cache_path), 0.1),
        }

    def _create_empty_sample(self, class_idx, class_label):
        """Create an empty sample with zeros when loading fails"""
        empty_spectrogram = torch.zeros((1, 224, 224), dtype=torch.float32)
        primary_target = torch.zeros(self.num_classes, dtype=torch.float32)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        primary_target[class_idx] = 1.0
        secondary_target = torch.zeros(self.num_classes, dtype=torch.float32)
        return {
            "melspec": empty_spectrogram,
            "target": target,
            "primary_target": primary_target,
            "secondary_target": secondary_target,
            "class_label": class_label,
            "segment_idx": -1,
            "signal_power_weight": 0.1,  # minimum weight for failed samples
        }

    def apply_spec_augmentations(self, spec, num_times_used):
        """Apply augmentations to spectrogram with usage-based intensity

        Args:
            spec (torch.Tensor): Input spectrogram
            num_times_used (int): Number of times the segment has been used
        """
        # Get usage count for this segment if tracking info is available
        usage_factor = min(1.0 + (num_times_used * 0.1), 2.0)  # Cap at 2.0

        # Time masking (horizontal stripes) with scaled width
        if random.random() < 0.5:
            num_masks = random.randint(1, max(2, int(3 * usage_factor)))
            for _ in range(num_masks):
                width = random.randint(5, int(20 * usage_factor))
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start : start + width] = 0

        # Frequency masking (vertical stripes) with scaled height
        if random.random() < 0.5:
            num_masks = random.randint(1, max(2, int(3 * usage_factor)))
            for _ in range(num_masks):
                height = random.randint(5, int(20 * usage_factor))
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start : start + height, :] = 0

        # Random brightness/contrast with scaled intensity
        if random.random() < 0.5:
            gain = random.uniform(1.0 - 0.2 * usage_factor, 1.0 + 0.2 * usage_factor)
            bias = random.uniform(-0.1 * usage_factor, 0.1 * usage_factor)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        # Frequency shift (pitch shift simulation) for heavily used samples
        if random.random() < 0.3 * usage_factor:
            shift = random.randint(-2, 2)
            spec = torch.roll(spec, shifts=shift, dims=1)

        # Time stretching for heavily used samples
        if random.random() < 0.3 * usage_factor:
            stretch_factor = random.uniform(0.9, 1.1)
            orig_size = spec.shape[2]
            stretched_size = int(orig_size * stretch_factor)
            if stretched_size != orig_size:
                spec = torch.nn.functional.interpolate(
                    spec.unsqueeze(0),
                    size=(spec.shape[1], stretched_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if stretched_size > orig_size:
                    spec = spec[:, :, :orig_size]
                else:
                    spec = torch.nn.functional.pad(
                        spec, (0, orig_size - stretched_size)
                    )

        # Add noise for heavily used samples
        if random.random() < 0.3 * usage_factor:
            noise = torch.randn_like(spec) * (0.02 * usage_factor)
            spec = spec + noise
            spec = torch.clamp(spec, 0, 1)

        return spec

    def _create_targets(
        self,
        segment_metadata,
    ):
        """
        Create primary and secondary target vectors with label smoothing and weighting.

        Args:
            segment_metadata: Metadata for the current segment

        Returns:
            tuple: (primary_target, secondary_target, target) tensors
        """
        num_classes = self.num_classes
        primary_idx = self.label_to_idx[segment_metadata["primary_label"]]
        secondary_labels = segment_metadata["secondary_labels"]
        USE_SMOOTHING = self.cfg.training.USE_LABEL_SMOOTHING
        primary_smoothing = (
            self.cfg.training.PRIMARY_LABEL_SMOOTHING if self.mode == "train" else 0.0
        )
        secondary_weight = (
            self.cfg.training.SECONDARY_LABEL_WEIGHT if self.mode == "train" else 0.0
        )

        # Create primary target with label smoothing
        if not USE_SMOOTHING or self.mode != "train":
            primary_target = torch.zeros(num_classes, dtype=torch.float32)
            primary_target[primary_idx] = 1.0
        else:
            # Initialize with small uniform distribution for smoothing
            primary_target = torch.full(
                (num_classes,),
                primary_smoothing / (num_classes - 1),
                dtype=torch.float32,
            )
            # Set primary label with smoothing
            primary_target[primary_idx] = 1.0 - primary_smoothing
        target = primary_target.clone()

        # Create secondary target
        secondary_target = torch.zeros(num_classes, dtype=torch.float32)
        if secondary_labels and secondary_weight > 0:
            if isinstance(secondary_labels, str):
                # Handle string format (e.g., "[label1,label2]")
                secondary_labels = (
                    eval(secondary_labels)
                    if secondary_labels.startswith("[")
                    else [secondary_labels]
                )

            for label in secondary_labels:
                if label in self.label_to_idx:
                    sec_idx = self.label_to_idx[label]
                    if (
                        sec_idx != primary_idx
                    ):  # Don't include primary label in secondary targets
                        secondary_target[sec_idx] = secondary_weight
                        target[sec_idx] += secondary_weight

        # Normalize primary target to ensure sum is 1
        if primary_target.sum() > 0:
            primary_target = primary_target / primary_target.sum()

        # Normalize target to ensure sum is 1
        if target.sum() > 0:
            target = target / target.sum()

        return primary_target, secondary_target, target


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
        if key in ["primary_target", "secondary_target", "target"] and isinstance(
            result[key][0], torch.Tensor
        ):
            result[key] = torch.stack(result[key])
        elif key == "melspec" and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
        elif key == "segment_idx":
            result[key] = torch.tensor(result[key])
        elif key == "signal_power_weight":
            result[key] = torch.tensor(result[key], dtype=torch.float32)

    return result
