# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25


import gc
import json
import multiprocessing as mp
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import CFG, set_seed
from src.data.dataset import BirdCLEFDataset, collate_fn, load_segments_info
from src.models.birdclef_model import BirdCLEFModel
from src.training.losses import AsymmetricLossMultiLabel
from src.training.metrics import (
    analyze_class_performance,
    calculate_class_metrics,
    plot_class_metrics,
    plot_confusion_matrix,
)
from src.utils.logger import WandbLogger, setup_logger

# Add at the beginning of your script, before any PyTorch imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

warnings.filterwarnings("ignore")
LOGS_DIR = Path("logs")
# LOGS_DIR = Path("/dbfs/RAW/W00001_Data_Unrestricted/Andrejs/birdclef-2025/logs/")

# Create global logger
logger = setup_logger(__name__)


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
    elif cfg.training.CRITERION == "AsymmetricLossMultiLabel":
        criterion = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,
            eps=1e-8,
            disable_torch_grad_focal_loss=False,
            reduction="mean",
        )
    else:
        raise NotImplementedError(f"Criterion {cfg.training.CRITERION} not implemented")

    return criterion


def calculate_metrics(targets, outputs, thresholds=None):
    """Calculate AUC and F1 scores for all classes"""
    num_classes = targets.shape[1]
    aucs = []
    f1s = []

    probs = 1 / (1 + np.exp(-outputs))
    if thresholds is None:
        thresholds = np.array([0.5] * num_classes)

    # Use class-specific thresholds
    preds = (probs > thresholds[:, None].T).astype(int)

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


def log_memory_usage():
    # Log system RAM memory
    ram = psutil.virtual_memory()
    logger.info(
        f"System RAM: {ram.used/1024**3:.0f}GB / "
        f"{ram.total/1024**3:.0f}GB / "
        f"{ram.available/1024**3:.0f}GB "
        f"(Used / Total / Free)"
    )

    # Log GPU memory if available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        free = total - allocated
        stats = torch.cuda.memory_stats()
        fragmentation = stats["allocated_bytes.all.current"] / (
            stats["reserved_bytes.all.current"] + 1
        )
        logger.info(
            f"GPU Memory: {allocated/1024**2:.0f}MB / "
            f"{reserved/1024**2:.0f}MB / "
            f"{total/1024**2:.0f}MB / "
            f"{free/1024**2:.0f}MB "
            f"(Allocated / Cached / Total / Free) /"
            f"Fragmentation: {fragmentation:.2%}"
        )


def verify_class_distribution(df, skf, logger):
    """
    Verify that all classes are present in both train and validation sets for each fold.

    Args:
        df: DataFrame containing the dataset
        skf: StratifiedKFold object with the split configuration
        logger: Logger object for output

    Returns:
        bool: True if all folds have the same classes in train and validation sets
    """
    all_folds_balanced = True
    logger.info(
        f"Total number of classes in dataset: {len(df['primary_label'].unique())}"
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["primary_label"])):
        train_classes = set(df.iloc[train_idx]["primary_label"].unique())
        val_classes = set(df.iloc[val_idx]["primary_label"].unique())

        if train_classes != val_classes:
            all_folds_balanced = False
            missing_in_train = val_classes - train_classes
            missing_in_val = train_classes - val_classes

            logger.warning(f"Fold {fold_idx} has class distribution mismatch!")
            if missing_in_train:
                logger.warning(f"Classes missing in training set: {missing_in_train}")
            if missing_in_val:
                logger.warning(f"Classes missing in validation set: {missing_in_val}")
    return all_folds_balanced


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    scaler=None,
    cfg=None,
    species_ids=None,
):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    # Initialize timing variables
    data_loading_time = 0
    training_time = 0
    batch_times = []

    # Use gradient accumulation steps from config
    grad_accum_steps = cfg.training.GRAD_ACCUM_STEPS

    enumerate_loader = enumerate(loader)
    pbar = tqdm(enumerate_loader, total=len(loader), desc="Training", unit="batch")

    optimizer.zero_grad()  # Zero gradients at the start of epoch

    for step, batch in pbar:
        batch_start = time.time()

        # Data loading
        inputs = batch["melspec"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        data_loading_time += time.time() - batch_start

        # Training step
        train_start = time.time()

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
            optimizer.zero_grad(set_to_none=True)

        training_time += time.time() - train_start
        batch_times.append(time.time() - batch_start)

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss.item() * grad_accum_steps)  # Scale back the loss for logging

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Update progress bar with timing info
        pbar.set_postfix(
            {
                "train_loss": np.mean(losses[-10:]) if losses else 0,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum": f"{step % grad_accum_steps + 1}/{grad_accum_steps}",
                "data_time": f"{data_loading_time/(step+1):.3f}s",
                "train_time": f"{training_time/(step+1):.3f}s",
                "batch_time": f"{np.mean(batch_times[-10:]):.3f}s",
            }
        )

    # Handle remaining gradients if any
    if len(loader) % grad_accum_steps != 0:
        if device == "cuda" and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Use enhanced metrics calculation
    metrics = calculate_class_metrics(
        all_targets, all_outputs, labels=species_ids, thresholds=None
    )

    # Calculate class distribution for analysis
    class_distribution = {
        label: np.sum(all_targets[:, i]) for i, label in enumerate(species_ids)
    }
    analysis = analyze_class_performance(metrics, species_ids, class_distribution)

    avg_loss = np.mean(losses)

    return avg_loss, metrics, analysis


def validate(model, loader, criterion, device, cfg, species_ids=None):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    enumerate_loader = enumerate(loader)
    pbar = tqdm(enumerate_loader, desc="Validation", unit="batch", total=len(loader))
    with torch.no_grad():
        for step, batch in pbar:
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

    # Use enhanced metrics calculation
    metrics = calculate_class_metrics(
        all_targets, all_outputs, labels=species_ids, thresholds=None
    )

    # Calculate class distribution for analysis
    class_distribution = {
        label: np.sum(all_targets[:, i]) for i, label in enumerate(species_ids)
    }
    analysis = analyze_class_performance(metrics, species_ids, class_distribution)

    avg_loss = np.mean(losses)

    return avg_loss, metrics, analysis


def get_folds(df, cfg):
    # Groups are audio files (recordings)
    groups = df["filename"]  # Each recording is a group
    # Labels for stratification
    labels = df["primary_label"]

    # Initialize the splitter
    skf = StratifiedGroupKFold(
        n_splits=cfg.training.N_FOLD, shuffle=True, random_state=cfg.seed
    )

    # Get fold indices
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels, groups)):
        # Verify no data leakage
        train_files = set(df.iloc[train_idx]["filename"])
        val_files = set(df.iloc[val_idx]["filename"])
        assert (
            len(train_files & val_files) == 0
        ), f"Data leakage detected in fold {fold}"

        # Verify class distribution
        train_dist = df.iloc[train_idx]["primary_label"].value_counts(normalize=True)
        val_dist = df.iloc[val_idx]["primary_label"].value_counts(normalize=True)

        logger.info(f"\nFold {fold}:")
        logger.info(f"Train: {len(train_files)} files")
        logger.info(f"Val: {len(val_files)} files")

        folds.append((train_idx, val_idx))

    return folds


def run_training(cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"training_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting training run in {run_dir}")

    # Initialize wandb group for all folds
    wandb_group = f"train_{'DEBUG' if cfg.training.DEBUG else ''}_{cfg.device.upper()}_{timestamp}"

    # Load training data
    logger.info("Loading training data...")
    df = pd.read_csv(cfg.dirs.train_csv)
    df_cache = load_segments_info(cfg)

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

    df_cache = df_cache[df_cache["primary_label"].isin(df["primary_label"].unique())]
    df = df[df["filename"].isin(set(df_cache["audio_file"]))]
    df = df.reset_index(drop=True)

    species_ids = df["primary_label"].unique().tolist()
    cfg.num_classes = len(species_ids)

    if "filepath" not in df.columns:
        df["filepath"] = cfg.dirs.train_datadir + "/" + df.filename
    if "samplename" not in df.columns:
        df["samplename"] = df.filename.map(
            lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
        )

    folds = get_folds(df, cfg)
    best_scores = []

    # Initialize gradient scaler for mixed precision training if using GPU
    scaler = torch.amp.GradScaler() if cfg.device == "cuda" else None

    torch.backends.cudnn.benchmark = True

    for fold, (train_idx, val_idx) in enumerate(folds):
        if fold not in cfg.training.SELECTED_FOLDS:
            continue

        logger.info(f"\n{'='*30} Fold {fold} {'='*30}")

        wandb_logger = WandbLogger(
            f"fold_{fold}",
            run_dir,
            group=wandb_group,
            tags=[
                f"fold_{fold}",
                f"n_classes_{cfg.num_classes}",
                f"{'DEBUG' if cfg.training.DEBUG else 'PROD'}",
                f"{cfg.device.upper()}",
            ],
            config={
                "batch_size": cfg.training.BATCH_SIZE,
                "learning_rate": cfg.training.LR,
                "epochs": cfg.training.EPOCHS,
                "model": cfg.model.model_name,
                "device": cfg.device,
                "seed": cfg.seed,
                "n_classes": cfg.num_classes,
                "optimizer": cfg.training.OPTIMIZER,
                "scheduler": cfg.training.SCHEDULER,
                "criterion": cfg.training.CRITERION,
                "early_stopping_metric": cfg.training.EARLY_STOPPING_METRIC,
                "early_stopping_patience": cfg.training.EARLY_STOPPING_PATIENCE,
                "samples_per_epoch": cfg.training.SAMPLES_PER_EPOCH,
            },
        )

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        logger.info(f"Training set: {len(train_df)} audio files")
        logger.info(f"Validation set: {len(val_df)} audio files")

        train_dataset = BirdCLEFDataset(train_df, cfg, species_ids, mode="train")
        val_dataset = BirdCLEFDataset(val_df, cfg, species_ids, mode="valid")

        # Create DataLoaders with proper worker configuration
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=False,  # We handle shuffling in the dataset
            num_workers=cfg.training.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                True if cfg.training.NUM_WORKERS > 0 else False
            ),  # Change to True
            prefetch_factor=cfg.training.PREFETCH_FACTOR,
            collate_fn=collate_fn,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=False,  # We handle shuffling in the dataset
            num_workers=cfg.training.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(
                True if cfg.training.NUM_WORKERS > 0 else False
            ),  # Change to True
            prefetch_factor=cfg.training.PREFETCH_FACTOR,
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

            # Set epoch for both train and validation datasets
            train_dataset.set_epoch(epoch)
            val_dataset.set_epoch(epoch)

            # Log memory usage at start of each epoch
            log_memory_usage()

            train_loss, train_metrics, train_analysis = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
                scaler,
                cfg,
                species_ids,
            )

            val_loss, val_metrics, val_analysis = validate(
                model, val_loader, criterion, cfg.device, cfg, species_ids
            )

            if scheduler is not None and not isinstance(
                scheduler, lr_scheduler.OneCycleLR
            ):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Create visualizations
            train_metrics_plots = plot_class_metrics(train_metrics, species_ids)
            val_metrics_plots = plot_class_metrics(val_metrics, species_ids)
            train_cm_plot = plot_confusion_matrix(
                train_metrics["confusion_matrix"], species_ids
            )
            val_cm_plot = plot_confusion_matrix(
                val_metrics["confusion_matrix"], species_ids
            )

            # Log metrics to wandb with fold grouping
            wandb_logger.log(
                {
                    "epoch": epoch + 1,
                    "fold": fold,
                    # Loss metrics
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    # Macro metrics
                    "train_auc": train_metrics["macro_metrics"]["auc"],
                    "train_f1": train_metrics["macro_metrics"]["f1"],
                    "train_precision": train_metrics["macro_metrics"]["precision"],
                    "train_recall": train_metrics["macro_metrics"]["recall"],
                    "val_auc": val_metrics["macro_metrics"]["auc"],
                    "val_f1": val_metrics["macro_metrics"]["f1"],
                    "val_precision": val_metrics["macro_metrics"]["precision"],
                    "val_recall": val_metrics["macro_metrics"]["recall"],
                    # Top-k accuracy
                    "train_top_k_accuracy": train_metrics["top_k_accuracy"],
                    "val_top_k_accuracy": val_metrics["top_k_accuracy"],
                    # Class size correlations
                    "train_class_size_f1_correlation": train_analysis[
                        "class_size_correlation"
                    ]["f1"],
                    "train_class_size_auc_correlation": train_analysis[
                        "class_size_correlation"
                    ]["auc"],
                    "val_class_size_f1_correlation": val_analysis[
                        "class_size_correlation"
                    ]["f1"],
                    "val_class_size_auc_correlation": val_analysis[
                        "class_size_correlation"
                    ]["auc"],
                    # Learning rate
                    "learning_rate": (
                        scheduler.get_last_lr()[0] if scheduler else cfg.training.LR
                    ),
                    # Visualizations
                    "train_precision_plot": train_metrics_plots[0],
                    "train_recall_plot": train_metrics_plots[1],
                    "train_f1_plot": train_metrics_plots[2],
                    "train_auc_plot": train_metrics_plots[3],
                    "val_precision_plot": val_metrics_plots[0],
                    "val_recall_plot": val_metrics_plots[1],
                    "val_f1_plot": val_metrics_plots[2],
                    "val_auc_plot": val_metrics_plots[3],
                    "train_confusion_matrix": train_cm_plot,
                    "val_confusion_matrix": val_cm_plot,
                }
            )

            # Log class-specific metrics
            for label in species_ids:
                wandb_logger.log(
                    {
                        f"train_{label}_precision": train_metrics["per_class"][label][
                            "precision"
                        ],
                        f"train_{label}_recall": train_metrics["per_class"][label][
                            "recall"
                        ],
                        f"train_{label}_f1": train_metrics["per_class"][label]["f1"],
                        f"train_{label}_auc": train_metrics["per_class"][label]["auc"],
                        f"train_{label}_support": train_metrics["per_class"][label][
                            "support"
                        ],
                        f"val_{label}_precision": val_metrics["per_class"][label][
                            "precision"
                        ],
                        f"val_{label}_recall": val_metrics["per_class"][label][
                            "recall"
                        ],
                        f"val_{label}_f1": val_metrics["per_class"][label]["f1"],
                        f"val_{label}_auc": val_metrics["per_class"][label]["auc"],
                        f"val_{label}_support": val_metrics["per_class"][label][
                            "support"
                        ],
                    }
                )

            # Log hard and easy classes
            wandb_logger.log(
                {
                    "train_hard_classes": train_analysis["hard_classes"],
                    "train_easy_classes": train_analysis["easy_classes"],
                    "val_hard_classes": val_analysis["hard_classes"],
                    "val_easy_classes": val_analysis["easy_classes"],
                }
            )

            # Log dataset usage statistics
            logger.info(f"\nDataset Usage Statistics for Epoch {epoch + 1}")
            train_dataset.log_usage_stats(epoch=epoch + 1)

            # Check for early stopping
            current_metric = val_metrics["macro_metrics"][
                cfg.training.EARLY_STOPPING_METRIC
            ]
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
                        "val_auc": val_metrics["macro_metrics"]["auc"],
                        "val_f1": val_metrics["macro_metrics"]["f1"],
                        "train_auc": train_metrics["macro_metrics"]["auc"],
                        "train_f1": train_metrics["macro_metrics"]["f1"],
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

                # Load best model state if it exists
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    if best_optimizer_state is not None:
                        optimizer.load_state_dict(best_optimizer_state)
                    if scheduler and best_scheduler_state is not None:
                        scheduler.load_state_dict(best_scheduler_state)
                else:
                    logger.warning("No best model state found to load")

                break

        best_scores.append(
            {
                "auc": val_metrics["macro_metrics"]["auc"],
                "f1": val_metrics["macro_metrics"]["f1"],
                "epoch": best_epoch,
            }
        )
        logger.info(
            f"Best metrics for fold {fold}: AUC: {val_metrics['macro_metrics']['auc']:.4f}, F1: {val_metrics['macro_metrics']['f1']:.4f} at epoch {best_epoch}"
        )

        # Proper cleanup at the end of each fold
        del train_loader
        del val_loader
        gc.collect()
        torch.cuda.empty_cache()

        # Log memory usage after cleanup
        log_memory_usage()

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
    cfg.update_machine_settings(machine="local")
    set_seed(cfg.seed)

    run_training(cfg)

    logger.info("Training complete!")
