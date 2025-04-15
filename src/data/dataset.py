import gc
import json
import os
import random
import threading
import time
from pathlib import Path

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
    df_cache = df_cache[~df_cache["rating"].isin([0.5, 1.0, 1.5])]
    df_cache = df_cache[df_cache["signal_power"] > 0.01]
    df_cache = df_cache[df_cache["snr_db"] > 0]

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

        # Initialize sample usage tracking with per-class locks
        self.current_epoch = 0
        self._class_locks = {
            class_label: threading.Lock() for class_label in self.class_ids
        }

        # Pre-generate permutations for each class
        self._class_permutations = {}
        self._class_permutation_indices = {}

        self.cache_dir = Path(self.cfg.dirs.cache_dir)

        self.df, self.metadata_df = align_df_and_metadata(
            self.df, load_cache_metadata(cfg)
        )
        self.samples_per_epoch = (
            cfg.training.SAMPLES_PER_EPOCH
            if cfg.training.SAMPLES_PER_EPOCH < len(self.metadata_df)
            else len(self.metadata_df)
        )
        self.metadata_df["index"] = range(len(self.metadata_df))

        # Fix path handling - ensure paths are relative to cache_dir
        self.metadata_df["cache_path"] = self.metadata_df.apply(
            lambda row: self.cache_dir
            / row["primary_label"]
            / f"{row['primary_label']}_{Path(row['filename']).stem}_segment_{row['segment_idx']}.pt",
            axis=1,
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
            }
            # Initialize permutation for this class
            self._class_permutation_indices[c] = 0
            self._class_permutations[c] = self.rng.permutation(n_segments)

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
            f"{self.mode} dataset loaded with {len(self.metadata_lookup_files)} files, {len(self.metadata_df)} samples and {len(self.metadata_lookup_classes)} classes; {self.samples_per_epoch} samples per epoch"
        )

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
        """Get next segment index from the pre-generated permutation"""
        with self._class_locks[class_label]:
            idx = self._class_permutation_indices[class_label]
            n_segments = self.metadata_lookup_classes[class_label]["n_segments"]

            # If we've used all segments, generate new permutation
            if idx >= n_segments:
                self._class_permutations[class_label] = self.rng.permutation(n_segments)
                idx = 0

            # Get segment index from permutation
            segment_idx = self._class_permutations[class_label][idx]
            self._class_permutation_indices[class_label] = idx + 1

            return segment_idx

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # First sample a class uniformly
        class_idx = self.rng.choice(len(self.class_ids), p=self.class_weights)
        class_label = self.class_ids[class_idx]

        # Get all segments for this class
        class_info = self.metadata_lookup_classes[class_label]
        n_segments = class_info["n_segments"]

        if n_segments == 0:
            logger.error(f"No segments found for class {class_label}")
            return self._create_empty_sample(class_idx, class_label)

        # Get next segment index from permutation
        segment_idx = self.get_next_segment_idx(class_label)
        cache_path = class_info["cache_paths"][segment_idx]
        spectrogram = torch.load(cache_path)

        if (
            self.mode == "train"
            and random.random() < self.cfg.training.AUGMENTATION_PROB
        ):
            spectrogram = self.apply_spec_augmentations(spectrogram)

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
            "segment_idx": segment_idx,
            "signal_power_weight": self.signal_power_lookup.get(str(cache_path), 0.1),
        }

    def _create_empty_sample(self, class_idx, class_label):
        """Create an empty sample with zeros when loading fails"""
        empty_spectrogram = torch.zeros((1, 224, 224), dtype=torch.float32)
        primary_target = torch.zeros(self.num_classes, dtype=torch.float32)
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

    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""

        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start : start + width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start : start + height, :] = 0

        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
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
