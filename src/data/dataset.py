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

    # Initial filtering of silence segments - do this first to reduce data size
    df_cache = df_cache[df_cache["silence_pct"] < 100]

    # Calculate class distributions once
    class_segment_counts = df_cache["primary_label"].value_counts()
    bottom_threshold = np.percentile(class_segment_counts.values, 10)
    top_threshold = np.percentile(class_segment_counts.values, 90)

    # Classify classes using boolean masks
    bottom_classes = set(
        class_segment_counts[class_segment_counts <= bottom_threshold].index
    )
    top_classes = set(class_segment_counts[class_segment_counts >= top_threshold].index)
    middle_classes = (
        set(df_cache["primary_label"].unique()) - bottom_classes - top_classes
    )

    # Create masks for each class category
    is_bottom = df_cache["primary_label"].isin(bottom_classes)
    is_middle = df_cache["primary_label"].isin(middle_classes)
    is_top = df_cache["primary_label"].isin(top_classes)

    # Process bottom classes - simple filtering
    bottom_mask = is_bottom & (df_cache["signal_power"] > 0) & (df_cache["snr_db"] > 0)

    # Process middle classes efficiently
    middle_mask = (
        is_middle
        & ~df_cache["rating"].isin([0.5, 1, 1.5])
        & (df_cache["signal_power"] > 0.001)
        & (df_cache["snr_db"] > 0)
    )

    # For middle and top classes, rank within each file and keep top 50%
    def get_top_half_mask(group, pct_to_keep=0.5, not_more_than_n=None):
        """Keep top half of segments per file"""
        n = len(group)
        if not_more_than_n is not None:
            n = min(n * pct_to_keep, not_more_than_n)
        return group["signal_power"].rank(ascending=False) <= n

    def get_top_n_mask(group, n_segments_to_keep=5):
        """Keep top n segments per file"""
        return group["signal_power"].rank(ascending=False) <= n_segments_to_keep

    # Process middle classes ranking
    middle_filtered = df_cache[middle_mask].copy()
    if not middle_filtered.empty:
        middle_ranks = middle_filtered.groupby("filename").apply(
            get_top_half_mask, pct_to_keep=0.5, not_more_than_n=10
        )
        middle_ranks = middle_ranks.reset_index(
            level=0, drop=True
        )  # Drop filename from index
        middle_mask = middle_mask & middle_ranks.reindex(
            df_cache.index, fill_value=False
        )

    top_mask = (
        is_top
        & (df_cache["signal_power"] > 0.001)
        & (df_cache["snr_db"] > 0)
        & ~(df_cache["rating"].isin([0.5, 1, 1.5]))
    )

    # Process top classes ranking
    top_filtered = df_cache[top_mask].copy()
    if not top_filtered.empty:
        top_ranks = top_filtered.groupby("filename").apply(
            get_top_half_mask, pct_to_keep=0.5, not_more_than_n=10
        )
        top_ranks = top_ranks.reset_index(
            level=0, drop=True
        )  # Drop filename from index
        top_mask = top_mask & top_ranks.reindex(df_cache.index, fill_value=False)
    else:
        top_mask = top_mask

    # Combine all masks efficiently
    final_mask = bottom_mask | middle_mask | top_mask
    df_cache = df_cache[final_mask]

    # Log the filtering results
    logger.info("*" * 100)
    logger.info(
        f"Quality-based segments filtering: {len(df_cache)} left from {len(df_cache[final_mask])}"
    )
    logger.info(
        f"Segments from bottom 10% classes: now {sum(df_cache['primary_label'].isin(bottom_classes))}, before {sum(is_bottom)}"
    )
    logger.info(
        f"Segments from middle 80% classes: now {sum(df_cache['primary_label'].isin(middle_classes))}, before {sum(is_middle)}"
    )
    logger.info(
        f"Segments from top 10% classes: now {sum(df_cache['primary_label'].isin(top_classes))}, before {sum(is_top)}"
    )
    logger.info("-" * 100)

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

            self.class_weights = self.compute_class_weights(
                weight_type=self.cfg.training.SAMPLING_CLASSES_WEIGHTS
            )
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

    def compute_class_weights(self, weight_type="uniform"):
        """Compute class weights based on different strategies.

        Args:
            weight_type (str): Type of weighting strategy to use
                - "uniform": Equal weights for all classes (default)
                - "segments": Weights proportional to number of segments per class
                - "segments_inverse": Weights inversely proportional to square root of segments per class
                - "files": Weights proportional to number of files per class
                - "files_inverse": Weights inversely proportional to square root of files per class

        Returns:
            np.ndarray: Array of class weights
        """
        if weight_type == "uniform":
            class_weights = np.ones(len(self.class_ids)) / len(self.class_ids)
            mask = ~np.isin(self.class_ids, self.classes_present_in_df)
            class_weights[mask] = 0
            if class_weights.sum() > 0:
                class_weights = class_weights / class_weights.sum()
            else:
                raise ValueError("No valid classes found in the dataset")
            return class_weights

        elif weight_type == "segments":
            # Get segment counts from metadata_lookup_classes
            segment_counts = np.zeros(len(self.class_ids))
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    segment_counts[i] = self.metadata_lookup_classes[cls]["n_segments"]

            # Compute weights proportional to segment counts
            total_segments = segment_counts.sum()
            if total_segments == 0:
                raise ValueError("No segments found in the dataset")

            # Direct weighting: more segments = higher weight
            class_weights = segment_counts / total_segments

            # Set weight to 0 for classes not present
            mask = ~np.isin(self.class_ids, self.classes_present_in_df)
            class_weights[mask] = 0

            # Log class weights for debugging
            logger.info("\nClass weights based on segment distribution (direct):")
            logger.info("Class | Segments | Weight")
            logger.info("-" * 30)
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    logger.info(
                        f"{cls} | {int(segment_counts[i]):^8d} | {class_weights[i]:.4f}"
                    )

            return class_weights

        elif weight_type == "segments_inverse":
            # Get segment counts from metadata_lookup_classes
            segment_counts = np.zeros(len(self.class_ids))
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    segment_counts[i] = self.metadata_lookup_classes[cls]["n_segments"]

            # Compute weights using square root transformation
            sqrt_segments = np.sqrt(segment_counts)
            total_sqrt = sqrt_segments.sum()
            if total_sqrt == 0:
                raise ValueError("No segments found in the dataset")

            # Inverse weighting with square root: more segments = lower weight, but less extreme
            class_weights = total_sqrt / (len(self.class_ids) * sqrt_segments)

            # Normalize weights
            class_weights = class_weights / class_weights.sum()

            # Set weight to 0 for classes not present
            mask = ~np.isin(self.class_ids, self.classes_present_in_df)
            class_weights[mask] = 0

            # Log class weights for debugging
            logger.info("\nClass weights based on segment distribution (inverse):")
            logger.info("Class | Segments | Weight")
            logger.info("-" * 30)
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    logger.info(
                        f"{cls} | {int(segment_counts[i]):^8d} | {class_weights[i]:.4f}"
                    )

            return class_weights

        elif weight_type == "files":
            # Get file counts from metadata_lookup_classes
            file_counts = np.zeros(len(self.class_ids))
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    file_counts[i] = self.metadata_lookup_classes[cls]["n_files"]

            # Compute weights proportional to file counts
            total_files = file_counts.sum()
            if total_files == 0:
                raise ValueError("No files found in the dataset")

            # Direct weighting: more files = higher weight
            class_weights = file_counts / total_files

            # Set weight to 0 for classes not present
            mask = ~np.isin(self.class_ids, self.classes_present_in_df)
            class_weights[mask] = 0

            # Log class weights for debugging
            logger.info("\nClass weights based on file distribution (direct):")
            logger.info("Class | Files | Weight")
            logger.info("-" * 30)
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    logger.info(
                        f"{cls} | {int(file_counts[i]):^5d} | {class_weights[i]:.4f}"
                    )

            return class_weights

        elif weight_type == "files_inverse":
            # Get file counts from metadata_lookup_classes
            file_counts = np.zeros(len(self.class_ids))
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    file_counts[i] = self.metadata_lookup_classes[cls]["n_files"]

            # Compute weights using square root transformation
            sqrt_files = np.sqrt(file_counts)
            total_sqrt = sqrt_files.sum()
            if total_sqrt == 0:
                raise ValueError("No files found in the dataset")

            # Inverse weighting with square root: more files = lower weight, but less extreme
            class_weights = total_sqrt / (len(self.class_ids) * sqrt_files)

            # Normalize weights
            class_weights = class_weights / class_weights.sum()

            # Set weight to 0 for classes not present
            mask = ~np.isin(self.class_ids, self.classes_present_in_df)
            class_weights[mask] = 0

            # Log class weights for debugging
            logger.info("\nClass weights based on file distribution (inverse):")
            logger.info("Class | Files | Weight")
            logger.info("-" * 30)
            for i, cls in enumerate(self.class_ids):
                if cls in self.classes_present_in_df:
                    logger.info(
                        f"{cls} | {int(file_counts[i]):^5d} | {class_weights[i]:.4f}"
                    )

            return class_weights

        else:
            raise ValueError(
                f"Unknown weight_type: {weight_type}. Use 'uniform', 'segments', 'segments_inverse', 'files', or 'files_inverse'"
            )

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
            segment_idx = self.rng.randint(
                0, self.metadata_lookup_classes[class_label]["n_segments"]
            )
            cache_path = self.metadata_lookup_classes[class_label]["cache_paths"][
                segment_idx
            ]
            segment_file = self.metadata_lookup_classes[class_label]["segments"][
                segment_idx
            ]
        else:
            # Validation mode: use deterministic ordering
            segment_info = self.all_segment_indices[idx]
            class_label = segment_info["class_label"]
            segment_file = segment_info["segment_file"]
            cache_path = segment_info["cache_path"]
            segment_idx = segment_info["segment_idx"]

        spectrogram = torch.load(cache_path)

        num_times_used = 0
        if self.mode == "train" and random.random() < 0:
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
        Label smoothing is only applied when secondary labels are present, to avoid
        unnecessarily reducing confidence in the primary label when we're certain about it.

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

        # Initialize targets
        primary_target = torch.zeros(num_classes, dtype=torch.float32)
        secondary_target = torch.zeros(num_classes, dtype=torch.float32)

        # Process secondary labels first to determine if we should apply smoothing
        has_valid_secondary = False
        if secondary_labels and secondary_weight > 0:
            if isinstance(secondary_labels, str):
                # Handle string format (e.g., "[label1,label2]")
                secondary_labels = (
                    eval(secondary_labels)
                    if secondary_labels.startswith("[")
                    else [secondary_labels]
                )

            # Collect valid secondary labels (excluding primary label)
            valid_secondary_indices = []
            for label in secondary_labels:
                if label in self.label_to_idx:
                    sec_idx = self.label_to_idx[label]
                    if (
                        sec_idx != primary_idx
                    ):  # Don't include primary label in secondary targets
                        valid_secondary_indices.append(sec_idx)
                        secondary_target[sec_idx] = secondary_weight
                        has_valid_secondary = True

        # Apply label smoothing only if we have valid secondary labels and smoothing is enabled
        if USE_SMOOTHING and self.mode == "train" and has_valid_secondary:
            # Apply smoothing only to classes with secondary labels
            smoothing_weight = primary_smoothing / max(len(valid_secondary_indices), 1)
            primary_target[primary_idx] = 1.0 - primary_smoothing
            for sec_idx in valid_secondary_indices:
                primary_target[sec_idx] = smoothing_weight
        else:
            # No smoothing - full confidence in primary label
            primary_target[primary_idx] = 1.0

        # Create combined target
        target = primary_target.clone()
        if has_valid_secondary:
            target += secondary_target
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
