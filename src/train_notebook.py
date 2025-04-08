# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25


import gc
import json
import logging
import math
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
from sklearn.metrics import roc_auc_score
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
        0.5  # if more than 50% of the segment is silence, skip it
    )

    seed = 42
    apex = False
    print_freq = 100
    num_workers = 1
    DATA_ROOT: Path = Path("data/birdclef-2025")

    train_datadir = (DATA_ROOT / "train_audio_no_voice").as_posix()
    train_csv = (DATA_ROOT / "train.csv").as_posix()
    test_soundscapes = (DATA_ROOT / "test_soundscapes").as_posix()
    submission_csv = (DATA_ROOT / "sample_submission.csv").as_posix()
    taxonomy_csv = (DATA_ROOT / "taxonomy.csv").as_posix()

    model_name = "efficientnet-b0"
    pretrained = True
    in_channels = 1
    kernel_size = (3, 3)
    cfar_scaling_factors = (1, 2)

    LOAD_DATA = False
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 20
    batch_size = 128
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

    mixup_alpha = 0.5

    debug = True

    def update_debug_settings(self):
        if self.debug:
            self.debug_n_classes = 3
            self.epochs = 20
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


# def process_audio_file2(audio_path, cfg):
#     """Process a single audio file to get the mel spectrogram"""
#     try:
#         audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

#         target_samples = int(cfg.TARGET_DURATION * cfg.FS)

#         if len(audio_data) < target_samples:
#             n_copy = math.ceil(target_samples / len(audio_data))
#             if n_copy > 1:
#                 audio_data = np.concatenate([audio_data] * n_copy)

#         # Extract center 5 seconds
#         start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
#         end_idx = min(len(audio_data), start_idx + target_samples)
#         center_audio = audio_data[start_idx:end_idx]

#         if len(center_audio) < target_samples:
#             center_audio = np.pad(
#                 center_audio, (0, target_samples - len(center_audio)), mode="constant"
#             )

#         mel_spec = audio2melspec(center_audio, cfg)

#         if mel_spec.shape != cfg.TARGET_SHAPE:
#             mel_spec = cv2.resize(
#                 mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR
#             )

#         return mel_spec.astype(np.float32)

#     except Exception as e:
#         logger.error(f"Error processing {audio_path}: {e}")
#         return None


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

        # Initialize segment tracking
        self.segment_indices = []
        for idx, row in self.df.iterrows():
            # If generating on-the-fly, we'll need to process the file to know
            # For now, we'll assume a reasonable number of segments
            n_segments = 1  # This will be updated when the file is processed
            self.segment_indices.extend(
                [(idx, seg_idx) for seg_idx in range(n_segments)]
            )

    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        file_idx, segment_idx = self.segment_indices[idx]
        row = self.df.iloc[file_idx]
        samplename = row["samplename"]
        spec = None

        # Process the file and get all segments
        result = process_audio_file(row, self.cfg.preprocessing)
        if result["success"] and result["segments"]:
            # Update the number of segments for this file
            n_segments = len(result["segments"])
            # Update segment indices for this file
            self.segment_indices = [
                (f_idx, s_idx)
                for f_idx, s_idx in self.segment_indices
                if f_idx != file_idx
            ] + [(file_idx, s_idx) for s_idx in range(n_segments)]
            # Get the requested segment
            spec = result["segments"][segment_idx]["spectrogram"]
        elif not result["success"]:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                logger.warning(
                    f"Warning: Spectrogram for {samplename} segment {segment_idx} not found and could not be generated"
                )

        # spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        # if self.mode == "train" and random.random() < self.cfg.aug_prob:
        # spec = self.apply_spec_augmentations(spec)

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
            "melspec": spec,
            "target": torch.tensor(target, dtype=torch.float32),
            "filename": row["filename"],
            "segment_idx": segment_idx,
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


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):

    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    enumerate_loader = enumerate(loader)
    pbar = tqdm(enumerate_loader, total=len(loader), desc="Training")

    for step, batch in pbar:

        if isinstance(batch["melspec"], list):
            batch_outputs = []
            batch_losses = []

            for i in range(len(batch["melspec"])):
                inputs = batch["melspec"][i].unsqueeze(0).to(device)
                target = batch["target"][i].unsqueeze(0).to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()

                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())

            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch["target"].numpy()

        else:
            inputs = batch["melspec"].to(device)
            logger.info(f"Inputs shape: {inputs.shape}")
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
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def validate(model, loader, criterion, device):

    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch["melspec"], list):
                batch_outputs = []
                batch_losses = []

                for i in range(len(batch["melspec"])):
                    inputs = batch["melspec"][i].unsqueeze(0).to(device)
                    target = batch["target"][i].unsqueeze(0).to(device)

                    output = model(inputs)
                    loss = criterion(output, target)

                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())

                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch["target"].numpy()

            else:
                inputs = batch["melspec"].to(device)
                targets = batch["target"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def calculate_auc(targets, outputs):

    num_classes = targets.shape[1]
    aucs = []

    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):

        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)

    return np.mean(aucs) if aucs else 0.0


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

    spectrograms = None

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
        best_epoch = 0

        for epoch in range(cfg.epochs):
            logger.info(f"\nEpoch {epoch+1}/{cfg.epochs}")

            train_loss, train_auc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
            )

            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

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
                    "epoch": epoch + 1,  # Same epoch axis for all folds
                    "fold": fold,
                    "train_loss": train_loss,
                    "train_auc": train_auc,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "learning_rate": (
                        scheduler.get_last_lr()[0] if scheduler else cfg.lr
                    ),
                    "_step": epoch + 1,  # For consistent x-axis
                    "_group": f"fold_{fold}",  # Group by fold
                }
            )

            logger.info(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                logger.info(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

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
                        "val_auc": val_auc,
                        "train_auc": train_auc,
                        "cfg": cfg,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        best_scores.append(best_auc)
        logger.info(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")

        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        logger.info(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    logger.info(f"Mean AUC: {np.mean(best_scores):.4f}")
    logger.info("=" * 60)

    # Save final results
    results = {
        "best_scores": best_scores,
        "mean_auc": float(np.mean(best_scores)),
        "std_auc": float(np.std(best_scores)),
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
