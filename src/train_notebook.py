# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25


import gc
import json
import multiprocessing as mp
import os
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
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import CFG, set_seed
from src.data.dataset import BirdCLEFDataset, collate_fn
from src.models.birdclef_model import BirdCLEFModel
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


def train_one_epoch(
    model, loader, optimizer, criterion, device, scheduler=None, scaler=None, cfg=None
):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    log_memory_every_n_batches = cfg.training.LOG_MEMORY_USAGE_EVERY_N_BATCHES

    # Use gradient accumulation steps from config
    grad_accum_steps = cfg.training.GRAD_ACCUM_STEPS

    enumerate_loader = enumerate(loader)
    pbar = tqdm(enumerate_loader, total=len(loader), desc="Training", unit="batch")

    optimizer.zero_grad()  # Zero gradients at the start of epoch

    # Log initial memory usage
    log_memory_usage()

    for step, batch in pbar:
        inputs = batch["melspec"].to(device, non_blocking=True)
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
            optimizer.zero_grad(set_to_none=True)

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss.item() * grad_accum_steps)  # Scale back the loss for logging

        # Log memory usage and dataset stats every 10 batches
        if (
            log_memory_every_n_batches is not None
            and step % min(len(loader), log_memory_every_n_batches) == 0
        ):
            # Log GPU memory if available
            log_memory_usage()
            # Log dataset memory usage and segment stats
            dataset = loader.dataset
            logger.info(
                f"Memory usage: {dataset.current_memory_usage.value/1024**3:.2f}GB / {dataset.max_memory_gb}GB, "
                f"Files in memory: {len(dataset.files_in_memory)}, "
                f"Files loaded this epoch: {len(dataset.segments_loaded_this_epoch)}"
            )

            # Clear memory more frequently
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        pbar.set_postfix(
            {
                "train_loss": np.mean(losses[-10:]) if losses else 0,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum": f"{step % grad_accum_steps + 1}/{grad_accum_steps}",
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
    metrics = calculate_metrics(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    # Clean up dataset memory after epoch
    loader.dataset.cleanup_after_epoch()

    # Log final memory usage for the epoch
    log_memory_usage()

    return avg_loss, metrics


def validate(model, loader, criterion, device, cfg):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    log_memory_every_n_batches = cfg.training.LOG_MEMORY_USAGE_EVERY_N_BATCHES

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

            # Log memory usage and dataset stats every 10 batches
            if (
                log_memory_every_n_batches is not None
                and step % min(len(loader), log_memory_every_n_batches) == 0
            ):
                log_memory_usage()
                dataset = loader.dataset
                logger.info(
                    f"Memory usage: {dataset.current_memory_usage.value/1024**3:.2f}GB / {dataset.max_memory_gb}GB, "
                    f"Files in memory: {len(dataset.files_in_memory)}, "
                    f"Files loaded this epoch: {len(dataset.segments_loaded_this_epoch)}"
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

    # Log initial memory usage
    log_memory_usage()

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
    scaler = torch.amp.GradScaler() if cfg.device == "cuda" else None

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["primary_label"])):
        if fold not in cfg.training.SELECTED_FOLDS:
            continue

        logger.info(f"\n{'='*30} Fold {fold} {'='*30}")

        # Log memory usage at start of each fold
        log_memory_usage()

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
                "device": cfg.device,
                "seed": cfg.seed,
                "n_classes": cfg.num_classes,
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

        # full_dataset = BirdCLEFDataset(df, cfg, species_ids, mode="full")
        train_dataset = BirdCLEFDataset(train_df, cfg, species_ids, mode="train")
        val_dataset = BirdCLEFDataset(val_df, cfg, species_ids, mode="valid")
        # raise ValueError("Stop here")

        # Create DataLoaders with proper worker configuration
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.training.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=cfg.training.PREFETCH_FACTOR,
            collate_fn=collate_fn,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.training.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,  # Disable persistent workers
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

            # Log memory usage at start of each epoch
            log_memory_usage()

            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
                scaler,
                cfg,
            )

            val_loss, val_metrics = validate(
                model, val_loader, criterion, cfg.device, cfg
            )

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

        # Log memory usage at end of each fold
        log_memory_usage()

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
