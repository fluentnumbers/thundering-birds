# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25


import gc
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv
from easydict import EasyDict
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.preprocessing import process_audio_file
from src.models.efficientnet_attention import EfficientNetWithAttention
from src.utils.logger import WandbLogger, setup_logger

warnings.filterwarnings("ignore")
LOGS_DIR = Path("logs")


# Create run directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = LOGS_DIR / f"training_run_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(__name__, run_dir)


@dataclass
class CFG:

    preprocessing = EasyDict()
    preprocessing.SAMPLE_RATE = 32000
    preprocessing.PADMODE = "constant"
    preprocessing.N_FFT = 1024
    preprocessing.HOP_LENGTH = 512
    preprocessing.N_MELS = 128
    preprocessing.FMIN = 50
    preprocessing.FMAX = 14000
    preprocessing.SEGMENT_DURATION = 5  # seconds
    preprocessing.NSAMPLES = preprocessing.SEGMENT_DURATION * preprocessing.SAMPLE_RATE
    preprocessing.UFOLD_OVERLAP = preprocessing.NSAMPLES // 2  # 2.5 seconds overlap
    preprocessing.MAKE_RGB = False
    preprocessing.SILENCE_THRESHOLD = (
        0.8  # if more than 50% of the segment is silence, skip it
    )

    seed = 42
    apex = False
    print_freq = 100
    num_workers = 10
    DATA_ROOT: Path = Path("data/birdclef-2025")

    train_datadir = (DATA_ROOT / "train_audio_no_voice").as_posix()
    train_csv = (DATA_ROOT / "train.csv").as_posix()
    test_soundscapes = (DATA_ROOT / "test_soundscapes").as_posix()
    taxonomy_csv = (DATA_ROOT / "taxonomy.csv").as_posix()
    cache_dir = (DATA_ROOT / "cache").as_posix()  # Add cache directory to config

    model_name = "efficientnet-b0"
    pretrained = True
    in_channels = 1
    kernel_size = (3, 3)
    cfar_scaling_factors = (1, 2)

    LOAD_DATA = False
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (224, 224)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 20
    batch_size = 64
    criterion = "BCEWithLogitsLoss"

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    optimizer = "AdamW"
    lr = 5e-4
    weight_decay = 1e-5

    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5

    mixup_alpha = 0

    debug = False

    def update_debug_settings(self):
        if self.debug:
            self.debug_n_classes = 5
            self.epochs = 40
            self.selected_folds = [0, 1, 2, 3, 4]


def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
        mel_spec_db.max() - mel_spec_db.min() + 1e-8
    )

    return mel_spec_norm


def process_file_worker(args):
    """Worker function for processing files in parallel"""
    file_idx, row, cfg = args
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
            return file_idx, tensor_segments
    except Exception as e:
        logger.warning(f"Failed to process {row['samplename']}: {e}")

    # Return zero segment if processing fails
    zero_segment = {
        "spectrogram": torch.zeros((1, 224, 224), dtype=torch.float32),
        "filename": row["filename"],
        "segment_idx": 0,
    }
    return file_idx, [zero_segment]


class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, species_ids, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.species_ids = species_ids
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if "filepath" not in self.df.columns:
            self.df["filepath"] = self.cfg.train_datadir + "/" + self.df.filename

        if "samplename" not in self.df.columns:
            self.df["samplename"] = self.df.filename.map(
                lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
            )

        # Create cache directory using config path
        self.cache_dir = Path(self.cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize file segments tracking
        self.file_segments = {}  # Maps file_idx to list of segments
        self.total_segments = 0
        self.file_indices = []  # List of (file_idx, segment_idx) pairs

        # First, check which files need processing
        files_to_process = []
        for file_idx, row in self.df.iterrows():
            cache_path = self._get_cache_path(file_idx)
            if not cache_path.exists():
                files_to_process.append((file_idx, row, self.cfg))
            else:
                # Just count segments from cache without loading them
                try:
                    # Load just the first segment to get the count
                    segments = torch.load(cache_path)
                    self.total_segments += len(segments)
                    self.file_indices.extend(
                        [(file_idx, i) for i in range(len(segments))]
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load cache for {cache_path}, will reprocess: {e}"
                    )
                    files_to_process.append((file_idx, row, self.cfg))

        if not files_to_process:
            logger.info("All files are already cached, no processing needed")
            return

        logger.info(
            f"Processing {len(files_to_process)} files out of {len(self.df)} total files"
        )

        # Process files in batches to manage memory
        batch_size = 100  # Process 100 files at a time
        total_batches = (len(files_to_process) + batch_size - 1) // batch_size

        with mp.Pool(processes=np.min([self.cfg.num_workers, mp.cpu_count()])) as pool:
            # Create a progress bar for the overall process
            with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(files_to_process))
                    batch_args = files_to_process[start_idx:end_idx]

                    # Process current batch
                    batch_results = list(pool.imap(process_file_worker, batch_args))

                    # Save results and update indices
                    for file_idx, segments in batch_results:
                        # Save to cache
                        self._save_to_cache(file_idx, segments)

                        # Update total segments and indices
                        self.total_segments += len(segments)
                        self.file_indices.extend(
                            [(file_idx, i) for i in range(len(segments))]
                        )

                    # Update progress bar
                    pbar.update(len(batch_args))

                    # Clear memory after each batch
                    gc.collect()

    def _get_cache_path(self, file_idx):
        """Get the cache path for a file"""
        row = self.df.iloc[file_idx]
        return self.cache_dir / f"{row['samplename']}.pt"

    def _load_from_cache(self, file_idx):
        """Load segments from cache if available"""
        cache_path = self._get_cache_path(file_idx)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_path}: {e}")
                return None
        return None

    def _save_to_cache(self, file_idx, segments):
        """Save segments to cache"""
        cache_path = self._get_cache_path(file_idx)
        try:
            torch.save(segments, cache_path)
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_path}: {e}")

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        # Get the file and segment index from our pre-computed list
        file_idx, segment_idx = self.file_indices[idx]

        # Load segments from cache if not in memory
        if file_idx not in self.file_segments:
            cache_path = self._get_cache_path(file_idx)
            try:
                segments = torch.load(cache_path)
                self.file_segments[file_idx] = segments
            except Exception as e:
                logger.error(f"Failed to load cache for {cache_path}: {e}")
                # Return zero segment if loading fails
                return {
                    "melspec": torch.zeros((1, 224, 224), dtype=torch.float32),
                    "target": torch.zeros(self.num_classes, dtype=torch.float32),
                    "filename": self.df.iloc[file_idx]["filename"],
                    "segment_idx": 0,
                }

        segments = self.file_segments[file_idx]
        segment = segments[segment_idx]
        row = self.df.iloc[file_idx]

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
            "filename": row["filename"],
            "segment_idx": segment_idx,
        }

    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target


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


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Initialize EfficientNetWithAttention model
        self.model = EfficientNetWithAttention(
            num_classes=cfg.num_classes,
            efficientnet_version=cfg.model_name,
            kernel_size=cfg.kernel_size,
            cfar_scaling_factors=cfg.cfar_scaling_factors,
        )

        self.mixup_enabled = hasattr(cfg, "mixup_alpha") and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        logits = self.model(x)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(
                F.binary_cross_entropy_with_logits, logits, targets_a, targets_b, lam
            )
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_optimizer(model, cfg):

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")

    return optimizer


def get_scheduler(optimizer, cfg):

    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True,
        )
    elif cfg.scheduler == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.epochs // 3, gamma=0.5)
    elif cfg.scheduler == "OneCycleLR":
        scheduler = None
    else:
        scheduler = None

    return scheduler


def get_criterion(cfg):

    if cfg.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")

    return criterion


def calculate_metrics(targets, outputs):
    """Calculate AUC and F1 scores for all classes"""
    num_classes = targets.shape[1]
    aucs = []
    f1s = []

    probs = 1 / (1 + np.exp(-outputs))
    preds = (probs > 0.5).astype(int)

    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            class_f1 = f1_score(targets[:, i], preds[:, i], zero_division=0)
            aucs.append(class_auc)
            f1s.append(class_f1)

    return {
        "auc": np.mean(aucs) if aucs else 0.0,
        "f1": np.mean(f1s) if f1s else 0.0,
        "aucs": aucs,
        "f1s": f1s,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    enumerate_loader = enumerate(loader)
    pbar = tqdm(enumerate_loader, total=len(loader), desc="Training")

    for step, batch in pbar:
        inputs = batch["melspec"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs, loss = outputs
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())

        pbar.set_postfix(
            {
                "train_loss": np.mean(losses[-10:]) if losses else 0,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, metrics


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for batch in pbar:
            inputs = batch["melspec"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())

        pbar.set_postfix(
            {
                "val_loss": np.mean(losses[-10:]) if losses else 0,
            }
        )

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, metrics


def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""

    if cfg.debug:
        cfg.update_debug_settings()
        # Filter the dataframe to keep only the top 3 classes
        class_counts = df["primary_label"].value_counts().sort_index()
        top_3_classes = class_counts[class_counts >= 4][
            : cfg.debug_n_classes
        ].index.tolist()

        df = df[df["primary_label"].isin(top_3_classes)]
        logger.info(
            f"Filtered training data to {len(df)} audio files from {cfg.debug_n_classes} classes"
        )
    species_ids = df["primary_label"].unique().tolist()
    cfg.num_classes = len(species_ids)

    logger.info("Will generate spectrograms on-the-fly during training.")
    if "filepath" not in df.columns:
        df["filepath"] = cfg.train_datadir + "/" + df.filename
    if "samplename" not in df.columns:
        df["samplename"] = df.filename.map(
            lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
        )

    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    best_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["primary_label"])):
        if fold not in cfg.selected_folds:
            continue

        logger.info(f'\n{"="*30} Fold {fold} {"="*30}')

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        logger.info(f"Training set: {len(train_df)} audio files")
        logger.info(f"Validation set: {len(val_df)} audio files")

        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, species_ids, mode="train")
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, species_ids, mode="valid")

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)

        if cfg.scheduler == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                pct_start=0.1,
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)

        best_auc = 0
        best_f1 = 0
        best_epoch = 0

        for epoch in range(cfg.epochs):
            logger.info(f"\nEpoch {epoch+1}/{cfg.epochs}")

            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
            )

            val_loss, val_metrics = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and not isinstance(
                scheduler, lr_scheduler.OneCycleLR
            ):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Log metrics to wandb with fold grouping
            wandb_logger.log(
                {
                    "epoch": epoch + 1,
                    "fold": fold,
                    "train_loss": train_loss,
                    "train_auc": train_metrics["auc"],
                    "train_f1": train_metrics["f1"],
                    "val_loss": val_loss,
                    "val_auc": val_metrics["auc"],
                    "val_f1": val_metrics["f1"],
                    "learning_rate": (
                        scheduler.get_last_lr()[0] if scheduler else cfg.lr
                    ),
                    "_step": epoch + 1,
                    "_group": f"fold_{fold}",
                }
            )

            logger.debug(
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_metrics['auc']:.4f}, Train F1: {train_metrics['f1']:.4f}"
            )
            logger.debug(
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}, Val F1: {val_metrics['f1']:.4f}"
            )

            # Save model if either AUC or F1 improves
            if val_metrics["auc"] > best_auc or val_metrics["f1"] > best_f1:
                best_auc = max(best_auc, val_metrics["auc"])
                best_f1 = max(best_f1, val_metrics["f1"])
                best_epoch = epoch + 1
                logger.info(
                    f"New best metrics - AUC: {best_auc:.4f}, F1: {best_f1:.4f} at epoch {best_epoch}"
                )

                # Save model checkpoint in run directory
                checkpoint_path = run_dir / f"model_fold{fold}_epoch{epoch+1}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "epoch": epoch,
                        "val_auc": val_metrics["auc"],
                        "val_f1": val_metrics["f1"],
                        "train_auc": train_metrics["auc"],
                        "train_f1": train_metrics["f1"],
                        "cfg": cfg,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        best_scores.append({"auc": best_auc, "f1": best_f1})
        logger.info(
            f"\nBest metrics for fold {fold}: AUC: {best_auc:.4f}, F1: {best_f1:.4f} at epoch {best_epoch}"
        )

        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation Results:")
    for fold, scores in enumerate(best_scores):
        logger.info(
            f"Fold {cfg.selected_folds[fold]}: AUC: {scores['auc']:.4f}, F1: {scores['f1']:.4f}"
        )
    logger.info(f"Mean AUC: {np.mean([s['auc'] for s in best_scores]):.4f}")
    logger.info(f"Mean F1: {np.mean([s['f1'] for s in best_scores]):.4f}")
    logger.info("=" * 60)

    # Save final results
    results = {
        "best_scores": best_scores,
        "mean_auc": float(np.mean([s["auc"] for s in best_scores])),
        "mean_f1": float(np.mean([s["f1"] for s in best_scores])),
        "std_auc": float(np.std([s["auc"] for s in best_scores])),
        "std_f1": float(np.std([s["f1"] for s in best_scores])),
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved results to {run_dir / 'results.json'}")

    # Finish wandb run
    wandb_logger.finish()


if __name__ == "__main__":
    load_dotenv(".env")
    # Initialize wandb logger with run directory
    wandb_logger = WandbLogger(f"training_run_{timestamp}", run_dir)
    cfg = CFG()
    set_seed(cfg.seed)

    logger.info("Loading training data...")
    train_df = pd.read_csv(cfg.train_csv)

    logger.info("Starting training...")

    logger.info("Will generate spectrograms on-the-fly during training")

    run_training(train_df, cfg)

    logger.info("Training complete!")
