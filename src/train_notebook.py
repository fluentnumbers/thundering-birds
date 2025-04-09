# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25


import gc
import json
import multiprocessing as mp
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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
# LOGS_DIR = Path("/dbfs/RAW/W00001_Data_Unrestricted/Andrejs/birdclef-2025/logs/")

# Create global logger
logger = setup_logger(__name__)


@dataclass
class CFG:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dirs = EasyDict()
    dirs.DATA_ROOT = Path("data/birdclef-2025")
    # dirs.DATA_ROOT = Path("/dbfs/RAW/W00001_Data_Unrestricted/Andrejs/birdclef-2025/")
    dirs.train_datadir = (dirs.DATA_ROOT / "train_audio_no_voice").as_posix()
    dirs.train_csv = (dirs.DATA_ROOT / "train.csv").as_posix()
    dirs.test_soundscapes = (dirs.DATA_ROOT / "test_soundscapes").as_posix()
    dirs.taxonomy_csv = (dirs.DATA_ROOT / "taxonomy.csv").as_posix()
    dirs.cache_dir = (dirs.DATA_ROOT / "cache").as_posix()

    training = EasyDict()
    training.DEBUG = True if device == "cpu" else False
    training.EPOCHS = 20
    training.N_FOLD = 5
    training.SELECTED_FOLDS = [0, 1, 2, 3, 4]
    training.NUM_WORKERS = 1
    training.SAVE_INTERMEDIATE_MODEL = True
    training.EARLY_STOPPING_METRIC = "f1"  # f1 auc
    training.EARLY_STOPPING_MIN_DELTA = 0.01
    training.EARLY_STOPPING_PATIENCE = 10
    training.BATCH_SIZE = 128 if device == "cuda" else 64
    training.OPTIMIZER = "AdamW"
    training.LR = 5e-4
    training.WEIGHT_DECAY = 1e-5
    training.SCHEDULER = "CosineAnnealingLR"
    training.CRITERION = "BCEWithLogitsLoss"
    training.MIN_LR = 1e-6
    training.T_MAX = training.EPOCHS
    training.AUG_PROB = 0.5

    def update_debug_settings(self):
        if self.training.DEBUG:
            self.training.DEBUG_N_CLASSES = 6
            self.training.EPOCHS = 30
            # self.training.SELECTED_FOLDS = [0, 1, 2, 3, 4]

    model = EasyDict()
    model.model_name = "efficientnet-b0"
    model.kernel_size = (3, 3)
    model.cfar_scaling_factors = (1, 2)
    model.mixup_alpha = 0
    preprocessing = EasyDict()
    # preprocessing.LOAD_DATA_STRICT = True if device == "cuda" else False
    preprocessing.NUM_WORKERS = 10
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


class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, species_ids, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.species_ids = species_ids
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

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
        self.file_segments = {}  # Maps file_idx to list of segments
        self.total_segments = 0
        self.file_indices = []  # List of (file_idx, segment_idx) pairs

        # Vectorized check for cache existence
        cache_paths = [
            self._get_cache_path(file_idx) for file_idx in range(len(self.df))
        ]
        cache_exists = [path.exists() for path in cache_paths]
        logger.info(f"Cache exists for {sum(cache_exists)} files out of {len(self.df)}")

        # Split files into cached and to-process
        time_start = time.time()
        cached_files = []
        files_to_process = []

        for file_idx, (row, is_cached) in tqdm(
            enumerate(zip(self.df.iterrows(), cache_exists)),
            total=len(self.df),
            desc="Loading cached files",
            unit="file",
        ):
            if is_cached:
                try:
                    # Load just the first segment to get the count
                    segments = torch.load(cache_paths[file_idx])
                    self.total_segments += len(segments)
                    self.file_indices.extend(
                        [(file_idx, i) for i in range(len(segments))]
                    )
                    cached_files.append(file_idx)
                except Exception as e:
                    logger.warning(
                        f"Failed to load cache for {cache_paths[file_idx]}, will reprocess: {e}"
                    )
                    files_to_process.append((file_idx, row[1], self.cfg))
            else:
                files_to_process.append((file_idx, row[1], self.cfg))

        if not files_to_process:
            logger.info(
                f"All files for mode {self.mode} are already cached, no processing needed [{time.time() - time_start:.1f}s]"
            )
            return

        logger.info(
            f"Processing {len(files_to_process)} files out of {len(self.df)} total files"
        )
        time_start = time.time()
        # Process files in parallel using multiprocessing
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
                for file_idx, segments in results:
                    if segments:  # Only process if we got valid segments
                        # Save to cache
                        self._save_to_cache(file_idx, segments)

                        # Update total segments and indices
                        self.total_segments += len(segments)
                        self.file_indices.extend(
                            [(file_idx, i) for i in range(len(segments))]
                        )

                # Clear memory after each batch
                gc.collect()
        logger.info(
            f"Processed {len(files_to_process)} files in {time.time() - time_start:.1f}s"
        )

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
            efficientnet_version=cfg.model.model_name,
            kernel_size=cfg.model.kernel_size,
            cfar_scaling_factors=cfg.model.cfar_scaling_factors,
        )

        self.mixup_enabled = (
            hasattr(cfg.model, "mixup_alpha") and cfg.model.mixup_alpha > 0
        )
        if self.mixup_enabled:
            self.mixup_alpha = cfg.model.mixup_alpha

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

    if cfg.training.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.LR,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    elif cfg.training.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.LR,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    elif cfg.training.OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.training.LR,
            momentum=0.9,
            weight_decay=cfg.training.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.training.OPTIMIZER} not implemented")

    return optimizer


def get_scheduler(optimizer, cfg):

    if cfg.training.SCHEDULER == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.EPOCHS, eta_min=cfg.training.MIN_LR
        )
    elif cfg.training.SCHEDULER == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=cfg.training.MIN_LR,
            verbose=True,
        )
    elif cfg.training.SCHEDULER == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.EPOCHS // 3, gamma=0.5
        )
    elif cfg.training.SCHEDULER == "OneCycleLR":
        scheduler = None
    else:
        scheduler = None

    return scheduler


def get_criterion(cfg):

    if cfg.training.CRITERION == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.training.CRITERION} not implemented")

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


def train_one_epoch(
    model, loader, optimizer, criterion, device, scheduler=None, scaler=None
):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    # Calculate gradient accumulation steps
    grad_accum_steps = max(
        1, 128 // cfg.training.BATCH_SIZE
    )  # Target effective batch size of 128

    enumerate_loader = enumerate(loader)
    pbar = tqdm(enumerate_loader, total=len(loader), desc="Training", unit="batches")

    optimizer.zero_grad()  # Zero gradients at the start of epoch

    for step, batch in pbar:
        inputs = batch["melspec"].to(
            device, non_blocking=True
        )  # Use non-blocking transfers
        targets = batch["target"].to(device, non_blocking=True)

        # Forward pass with mixed precision if using GPU
        if device == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs, loss = outputs
                else:
                    loss = criterion(outputs, targets)
                loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        else:
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation

        # Backward pass with mixed precision if using GPU
        if device == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer only after accumulating gradients
        if (step + 1) % grad_accum_steps == 0:
            if device == "cuda" and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss.item() * grad_accum_steps)  # Scale back the loss for logging

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

    pbar = tqdm(loader, desc="Validation", unit="batches")
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

            # Update progress bar with current batch loss
            pbar.set_postfix(
                {
                    "val_loss": f"{np.mean(losses[-10:]):.2f}",
                }
            )

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, metrics


def run_training(cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"training_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting training run in {run_dir}")

    # Initialize wandb group for all folds
    wandb_group = f"train_{cfg.device.upper()}_{timestamp}"

    # Load training data
    logger.info("Loading training data...")
    df = pd.read_csv(cfg.dirs.train_csv)

    if cfg.training.DEBUG:
        cfg.update_debug_settings()
        # Filter the dataframe to keep only the top 3 classes
        class_counts = df["primary_label"].value_counts().sort_index()
        top_3_classes = class_counts[class_counts >= 4][
            : cfg.training.DEBUG_N_CLASSES
        ].index.tolist()

        df = df[df["primary_label"].isin(top_3_classes)]
        logger.info(
            f"Filtered training data to {len(df)} audio files from {cfg.training.DEBUG_N_CLASSES} classes"
        )
    species_ids = df["primary_label"].unique().tolist()
    cfg.num_classes = len(species_ids)

    if "filepath" not in df.columns:
        df["filepath"] = cfg.dirs.train_datadir + "/" + df.filename
    if "samplename" not in df.columns:
        df["samplename"] = df.filename.map(
            lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
        )

    skf = StratifiedKFold(
        n_splits=cfg.training.N_FOLD,
        shuffle=True,
        random_state=cfg.seed,
    )

    best_scores = []

    # Initialize gradient scaler for mixed precision training if using GPU
    scaler = torch.cuda.amp.GradScaler() if cfg.device == "cuda" else None

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["primary_label"])):
        if fold not in cfg.training.SELECTED_FOLDS:
            continue

        logger.info(f"\n{'='*30} Fold {fold} {'='*30}")

        # Initialize wandb run for this fold
        wandb_logger = WandbLogger(
            f"fold_{fold}",
            run_dir,
            group=wandb_group,
            tags=[f"fold_{fold}"],
            config={
                "batch_size": cfg.training.BATCH_SIZE,
                "learning_rate": cfg.training.LR,
                "epochs": cfg.training.EPOCHS,
                "model": cfg.model.model_name,
                "optimizer": cfg.training.OPTIMIZER,
                "scheduler": cfg.training.SCHEDULER,
                "criterion": cfg.training.CRITERION,
                "early_stopping_metric": cfg.training.EARLY_STOPPING_METRIC,
                "early_stopping_patience": cfg.training.EARLY_STOPPING_PATIENCE,
            },
        )

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        logger.info(f"Training set: {len(train_df)} audio files")
        logger.info(f"Validation set: {len(val_df)} audio files")

        train_dataset = BirdCLEFDataset(train_df, cfg, species_ids, mode="train")
        val_dataset = BirdCLEFDataset(val_df, cfg, species_ids, mode="valid")
        # raise ValueError("Stop here")

        # Create DataLoaders with proper worker configuration
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=False,  # Disable persistent workers
            prefetch_factor=2,
            collate_fn=collate_fn,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=False,  # Disable persistent workers
            prefetch_factor=2,
            collate_fn=collate_fn,
        )

        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)

        if cfg.training.SCHEDULER == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.training.LR,
                steps_per_epoch=len(train_loader),
                epochs=cfg.training.EPOCHS,
                pct_start=0.1,
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)

        best_epoch = 0
        best_model_state = None
        best_optimizer_state = None
        best_scheduler_state = None

        # Early stopping variables
        no_improvement_epochs = 0
        best_metric = 0

        for epoch in range(cfg.training.EPOCHS):
            logger.info(f"Epoch {epoch+1}/{cfg.training.EPOCHS}")

            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
                scaler,  # Pass scaler to train_one_epoch
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
                        scheduler.get_last_lr()[0] if scheduler else cfg.training.LR
                    ),
                }
            )

            # Check for early stopping
            current_metric = val_metrics[cfg.training.EARLY_STOPPING_METRIC]
            if current_metric > best_metric + cfg.training.EARLY_STOPPING_MIN_DELTA:
                best_metric = current_metric
                no_improvement_epochs = 0

                # Save best model state
                best_model_state = model.state_dict().copy()
                best_optimizer_state = optimizer.state_dict().copy()
                best_scheduler_state = (
                    scheduler.state_dict().copy() if scheduler else None
                )
                best_epoch = epoch + 1

                logger.info(
                    f"New best {cfg.training.EARLY_STOPPING_METRIC}: {best_metric:.3f} at epoch {best_epoch}"
                )

                # Save model checkpoint when metrics improve
                checkpoint_path = run_dir / f"model_fold{fold}_epoch{epoch+1}_best.pth"

                # Delete previous checkpoints for this fold
                for old_checkpoint in run_dir.glob(f"model_fold{fold}_*_best.pth"):
                    if old_checkpoint != checkpoint_path:
                        old_checkpoint.unlink()
                        logger.debug(f"Deleted old checkpoint {old_checkpoint}")

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
                logger.debug(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                no_improvement_epochs += 1
                logger.info(
                    f"No improvement in {cfg.training.EARLY_STOPPING_METRIC} for {no_improvement_epochs} epochs"
                )

            # Check for early stopping
            if no_improvement_epochs >= cfg.training.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                # Load best model state
                model.load_state_dict(best_model_state)
                optimizer.load_state_dict(best_optimizer_state)
                if scheduler and best_scheduler_state:
                    scheduler.load_state_dict(best_scheduler_state)
                break

        best_scores.append(
            {"auc": val_metrics["auc"], "f1": val_metrics["f1"], "epoch": best_epoch}
        )
        logger.info(
            f"Best metrics for fold {fold}: AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f} at epoch {best_epoch}"
        )

        # Proper cleanup at the end of each fold
        del train_loader
        del val_loader
        gc.collect()
        torch.cuda.empty_cache()

        # Finish wandb run for this fold
        wandb_logger.finish()

    logger.info("Cross-Validation Results:")
    for fold, scores in enumerate(best_scores):
        logger.info(f"Fold {fold}: AUC: {scores['auc']:.4f}, F1: {scores['f1']:.4f}")
    logger.info(f"Mean AUC: {np.mean([s['auc'] for s in best_scores]):.4f}")
    logger.info(f"Mean F1: {np.mean([s['f1'] for s in best_scores]):.4f}")

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


if __name__ == "__main__":
    load_dotenv(".env")

    cfg = CFG()
    set_seed(cfg.seed)

    run_training(cfg)

    logger.info("Training complete!")
