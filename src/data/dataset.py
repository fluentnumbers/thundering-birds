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

        self.cache_dir = Path(self.cfg.dirs.cache_dir)
        if not self.cache_dir.exists() or self.cache_dir.empty():
            logger.error(f"Cache directory {self.cache_dir} does not exist")
            raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist")
        else:
            self.metadata_df = pd.read_csv(self.cache_dir / "metadata.csv")

            # Filter out rows with 100% silence
            self.metadata_df = self.metadata_df[self.metadata_df["silence_pct"] < 100]
            # Filter out rows with low ratings (0.5, 1.0, 1.5)
            self.metadata_df = self.metadata_df[
                ~self.metadata_df["rating"].isin([0.5, 1.0, 1.5])
            ]
            # Filter out rows with low signal power
            self.metadata_df = self.metadata_df[self.metadata_df["signal_power"] > 0.01]
            # Filter out rows with low SNR
            self.metadata_df = self.metadata_df[self.metadata_df["snr_db"] > 0]

            self.metadata_df = self.metadata_df.reset_index(drop=True)
            self.metadata_df["index"] = range(len(self.metadata_df))
            self.metadata_df["cache_path"] = Path(
                self.cache_dir / self.metadata_df["segment_file"]
            )
            self.metadata_lookup_files = {}

            unique_files = self.metadata_df["audio_file"].unique()
            for f in unique_files:
                file_df = self.metadata_df[self.metadata_df["audio_file"] == f]
                self.metadata_lookup_files[f] = {
                    "n_segments": len(file_df),
                    "indices": file_df["index"].tolist(),
                    "cache_paths": file_df["cache_path"].tolist(),
                }
            self.metadata_lookup_classes = {}
            unique_classes = self.metadata_df["primary_label"].unique()
            for c in unique_classes:
                class_df = self.metadata_df[self.metadata_df["primary_label"] == c]
                self.metadata_lookup_classes[c] = {
                    "n_segments": len(class_df),
                    "n_files": len(class_df["audio_file"].unique()),
                    "indices": class_df["index"].tolist(),
                    "files": class_df["audio_file"].unique().tolist(),
                    "cache_paths": class_df["cache_path"].tolist(),
                }
        self.total_segments = len(self.metadata_df)

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        filename, segment_idx = self.file_indices[idx]

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
