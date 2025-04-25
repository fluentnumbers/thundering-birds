# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25


import gc
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import CFG, set_seed
from src.data.dataset import BirdCLEFDataset, collate_fn, load_cache_metadata
from src.data.processing import align_df_and_metadata
from src.models.birdclef_model import BirdCLEFModel
from src.training.criterion import get_criterion
from src.training.losses import HierarchicalBCELoss
from src.training.metrics import (
    analyze_class_performance,
    calculate_class_metrics,
    plot_class_metrics,
    plot_confusion_matrix,
)
from src.training.scheduler import get_scheduler
from src.training.optimizer import get_optimizer
from src.utils.logger import WandbLogger, setup_logger
from src.training.folds import get_folds
from src.utils.utils import log_memory_usage
# Add at the beginning of your script, before any PyTorch imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

warnings.filterwarnings("ignore")
LOGS_DIR = Path("logs")
# LOGS_DIR = Path("/dbfs/RAW/W00001_Data_Unrestricted/Andrejs/birdclef-2025/logs/")

# Create global logger
logger = setup_logger(__name__)



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
    train_weighted_losses = []  # Track weighted losses separately

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
        loss_correction = batch["signal_power_weight"].to(device, non_blocking=True)
        data_loading_time += time.time() - batch_start

        # Training step
        train_start = time.time()

        # Forward pass with mixed precision if using GPU
        if device == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # Calculate base loss based on criterion type
                if isinstance(criterion, HierarchicalBCELoss):
                    primary_targets = batch["primary_target"].to(
                        device, non_blocking=True
                    )
                    secondary_targets = batch["secondary_target"].to(
                        device, non_blocking=True
                    )
                    base_loss = criterion(outputs, primary_targets, secondary_targets)
                else:
                    base_loss = criterion(outputs, targets)

                if len(base_loss.shape) == 2:
                    # If loss is per-class, take mean over classes first
                    base_loss = base_loss.mean(dim=1)
                # Apply signal power weights to each sample's loss if enabled
                loss = (
                    (base_loss * loss_correction).mean()
                    if cfg.training.CORRECT_LOSS
                    else base_loss.mean()
                )
                loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        else:
            outputs = model(inputs)
            # Calculate base loss based on criterion type
            if isinstance(criterion, HierarchicalBCELoss):
                primary_targets = batch["primary_target"].to(device, non_blocking=True)
                secondary_targets = batch["secondary_target"].to(
                    device, non_blocking=True
                )
                base_loss = criterion(outputs, primary_targets, secondary_targets)
            else:
                base_loss = criterion(outputs, targets)

            if len(base_loss.shape) == 2:
                # If loss is per-class, take mean over classes first
                base_loss = base_loss.mean(dim=1)
            # Apply signal power weights to each sample's loss if enabled
            loss = (
                (base_loss * loss_correction).mean()
                if cfg.training.CORRECT_LOSS
                else base_loss.mean()
            )
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation

        # Backward pass with mixed precision if using GPU
        if device == "cuda" and scaler is not None:
            scaler.scale(loss).backward()

            # Unscale before gradient clipping
            if (
                cfg.training.MAX_GRAD_NORM is not None
                and (step + 1) % grad_accum_steps == 0
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.MAX_GRAD_NORM
                )

            # Step optimizer only after accumulating gradients
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()

            # Apply gradient clipping
            if (
                cfg.training.MAX_GRAD_NORM is not None
                and (step + 1) % grad_accum_steps == 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.MAX_GRAD_NORM
                )

            # Step optimizer only after accumulating gradients
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        training_time += time.time() - train_start
        batch_times.append(time.time() - batch_start)

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()  # Use primary targets for metrics

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss.item() * grad_accum_steps)  # Scale back the loss for logging
        train_weighted_losses.append(
            loss.item() * grad_accum_steps
        )  # Track weighted losses

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Update progress bar with timing info and both losses
        pbar.set_postfix(
            {
                "train_loss": np.mean(losses[-10:]) if losses else 0,
                "train_weighted_loss": (
                    np.mean(train_weighted_losses[-10:]) if train_weighted_losses else 0
                ),
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
    avg_train_weighted_loss = np.mean(train_weighted_losses)

    # Add weighted loss to metrics
    metrics["train_weighted_loss"] = avg_train_weighted_loss

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

            # Calculate loss based on criterion type
            if isinstance(criterion, HierarchicalBCELoss):
                primary_targets = batch["primary_target"].to(device)
                secondary_targets = batch["secondary_target"].to(device)
                loss = criterion(outputs, primary_targets, secondary_targets)
            else:
                loss = criterion(outputs, targets)

            if isinstance(loss, torch.Tensor) and len(loss.shape) > 0:
                loss = loss.mean()

            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss.item())

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




def run_training(cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"training_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting training run in {run_dir}")



    # Load training data
    logger.info("Loading training data...")

    df = pd.read_csv(cfg.dirs.train_csv)
    if cfg.training.DEBUG:
        cfg.update_debug_settings()
        # Filter the dataframe to keep only the top 3 classes
        class_counts = df["primary_label"].value_counts().sort_index()
        # Get half of the classes with most files and half with least files
        half_n = cfg.training.DEBUG_N_CLASSES // 2
        most_common_classes = class_counts.nlargest(half_n).index.tolist()
        least_common_classes = (
            class_counts[class_counts >= 4].nsmallest(half_n).index.tolist()
        )
        top_n_classes = least_common_classes + most_common_classes

        df = df[df["primary_label"].isin(top_n_classes)]
        logger.info(
            f"For DEBUG run, training data was filtered to {len(df)} audio files from {cfg.training.DEBUG_N_CLASSES} classes"
        )
    df, df_cache = align_df_and_metadata(df, load_cache_metadata(cfg))
    species_ids = df["primary_label"].unique().tolist()
    cfg.num_classes = len(species_ids)

    folds = get_folds(df, cfg)
    best_scores = []

    # Initialize gradient scaler for mixed precision training if using GPU
    scaler = torch.amp.GradScaler() if cfg.device == "cuda" else None

    torch.backends.cudnn.benchmark = True

    # Initialize wandb group for all folds
    wandb_group = f"train_{'DEBUG' if cfg.training.DEBUG else 'PROD'}_{cfg.num_classes}classes_{cfg.device.upper()}_{timestamp}"

    for fold, (train_file_idx, val_file_idx) in enumerate(folds):
        if fold not in cfg.training.SELECTED_FOLDS:
            continue
        train_df = df.iloc[train_file_idx].reset_index(drop=False)
        val_df = df.iloc[val_file_idx].reset_index(drop=False)

        train_dataset = BirdCLEFDataset(train_df, cfg, species_ids, mode="train")
        val_dataset = BirdCLEFDataset(val_df, cfg, species_ids, mode="valid")

        logger.info(f"\n{'='*30} Fold {fold} {'='*30}")

        wandb_logger = WandbLogger(
            f"fold{fold}_{cfg.num_classes}classes",
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
                "learning_rate": cfg.training.BASE_LR,
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
                "progressive_unfreezing": cfg.training.PROGRESSIVE_UNFREEZING,
                "sampling_classes_weights": cfg.training.SAMPLING_CLASSES_WEIGHTS,
                "samples_per_epoch": cfg.training.SAMPLES_PER_EPOCH,
                "lr_scaling": cfg.training.LR_SCALING,
                "lr_scale_factor": cfg.training.LR_SCALE_FACTOR,
                "warmup_epochs": cfg.training.WARMUP_EPOCHS,
                "warmup_factor": cfg.training.WARMUP_FACTOR,
                "max_grad_norm": cfg.training.MAX_GRAD_NORM,
                "lr_layer_decay": cfg.training.LR_LAYER_DECAY,
                "train_files": len(train_dataset.metadata_df["filename"].unique()),
                "train_segments": len(train_dataset.metadata_df),
                "train_classes": len(train_dataset.class_ids),
                "val_files": len(val_dataset.metadata_df["filename"].unique()),
                "val_segments": len(val_dataset.metadata_df),
                "val_classes": len(val_dataset.class_ids),
            },
        )
        # After initializing wandb_logger
        config_dict = cfg.to_dict()
        wandb_logger.store_config_artifact(config_dict, artifact_name="training_config")

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

        # Initially freeze the backbone
        if cfg.training.PROGRESSIVE_UNFREEZING:
            model.model.freeze_backbone(num_layers=-1)
            logger.info("Initial trainable parameters:")
            param_counts = model.model.get_trainable_params()
            for component, count in param_counts.items():
                logger.info(f"{component}: {count:,} parameters")

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

            # Progressive unfreezing
            if cfg.training.PROGRESSIVE_UNFREEZING:
                model.model.progressive_unfreeze(epoch, cfg.training.EPOCHS)

                # Log current trainable parameters
                if epoch % 5 == 0:  # Log every 5 epochs
                    param_counts = model.model.get_trainable_params()
                    logger.info(f"\nTrainable parameters at epoch {epoch+1}:")
                    for component, count in param_counts.items():
                        logger.info(f"{component}: {count:,} parameters")

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

            # Log all metrics to wandb with fold grouping
            wandb_logger.log(
                {
                    "epoch": epoch + 1,
                    "fold": fold,
                    "TOP/val_loss": val_loss,
                    "TOP/train_loss": train_loss,
                    "TOP/val_f1": val_metrics["macro_metrics"]["f1"],
                    "TOP/val_confusion_matrix": val_cm_plot,
                    # Loss metrics
                    "train/loss": train_loss,
                    "train/weighted_loss": train_metrics["train_weighted_loss"],
                    "val/loss": val_loss,
                    # Macro metrics
                    "train/auc": train_metrics["macro_metrics"]["auc"],
                    "train/f1": train_metrics["macro_metrics"]["f1"],
                    "train/precision": train_metrics["macro_metrics"]["precision"],
                    "train/recall": train_metrics["macro_metrics"]["recall"],
                    "val/auc": val_metrics["macro_metrics"]["auc"],
                    "val/f1": val_metrics["macro_metrics"]["f1"],
                    "val/precision": val_metrics["macro_metrics"]["precision"],
                    "val/recall": val_metrics["macro_metrics"]["recall"],
                    # Top-k accuracy
                    "train/top_k_accuracy": train_metrics["top_k_accuracy"],
                    "val/top_k_accuracy": val_metrics["top_k_accuracy"],
                    # Class size correlations
                    "train/class_size_f1_correlation": train_analysis[
                        "class_size_correlation"
                    ]["f1"],
                    "train/class_size_auc_correlation": train_analysis[
                        "class_size_correlation"
                    ]["auc"],
                    "val/class_size_f1_correlation": val_analysis[
                        "class_size_correlation"
                    ]["f1"],
                    "val/class_size_auc_correlation": val_analysis[
                        "class_size_correlation"
                    ]["auc"],
                    # Learning rate
                    "learning_rate": (
                        scheduler.get_last_lr()[0] if scheduler else cfg.training.LR
                    ),
                    "train/hard_classes": train_analysis["hard_classes"],
                    "train/easy_classes": train_analysis["easy_classes"],
                    "val/hard_classes": val_analysis["hard_classes"],
                    "val/easy_classes": val_analysis["easy_classes"],
                    # Visualizations
                    "train/precision_plot": train_metrics_plots[0],
                    "train/recall_plot": train_metrics_plots[1],
                    "train/f1_plot": train_metrics_plots[2],
                    "train/auc_plot": train_metrics_plots[3],
                    "val/precision_plot": val_metrics_plots[0],
                    "val/recall_plot": val_metrics_plots[1],
                    "val/f1_plot": val_metrics_plots[2],
                    "val/auc_plot": val_metrics_plots[3],
                    "train/confusion_matrix": train_cm_plot,
                    "val/confusion_matrix": val_cm_plot,
                },
            )

            # Log class-specific metrics
            for label in species_ids:
                wandb_logger.log(
                    {
                        f"per_class/train_{label}_precision": train_metrics[
                            "per_class"
                        ][label]["precision"],
                        f"per_class/train_{label}_recall": train_metrics["per_class"][
                            label
                        ]["recall"],
                        f"per_class/train_{label}_f1": train_metrics["per_class"][
                            label
                        ]["f1"],
                        f"per_class/train_{label}_auc": train_metrics["per_class"][
                            label
                        ]["auc"],
                        f"per_class/train_{label}_support": train_metrics["per_class"][
                            label
                        ]["support"],
                        f"per_class/val_{label}_precision": val_metrics["per_class"][
                            label
                        ]["precision"],
                        f"per_class/val_{label}_recall": val_metrics["per_class"][
                            label
                        ]["recall"],
                        f"per_class/val_{label}_f1": val_metrics["per_class"][label][
                            "f1"
                        ],
                        f"per_class/val_{label}_auc": val_metrics["per_class"][label][
                            "auc"
                        ],
                        f"per_class/val_{label}_support": val_metrics["per_class"][
                            label
                        ]["support"],
                    }
                )

            # Use raw validation loss or other metrics for early stopping
            if cfg.training.EARLY_STOPPING_METRIC == "loss":
                current_metric = -val_loss  # Negative because we want to minimize loss
            else:
                current_metric = val_metrics["macro_metrics"][
                    cfg.training.EARLY_STOPPING_METRIC
                ]

            logger.info(
                f"Current {cfg.training.EARLY_STOPPING_METRIC}: {current_metric:.3f}, best: {best_metric:.3f}, delta: {current_metric - best_metric:.3f}"
            )
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
                    f"New best {cfg.training.EARLY_STOPPING_METRIC}: {best_metric:.3f} at epoch {best_epoch} \n"
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
                    f"No improvement in {cfg.training.EARLY_STOPPING_METRIC} for {no_improvement_epochs} epochs (best value: {best_metric:.3f})"
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
