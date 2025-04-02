import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.config import LOGS_DIR
from src.data.dataset import BirdSoundDataset, collate_fn, get_transforms
from src.data.preprocessing import load_metadata, preprocess_and_save_dataset
from src.models.model_factory import ModelFactory
from src.utils.logger import WandbLogger, setup_logger
from src.utils.visualization import save_attention_outputs

logger = setup_logger(__name__)
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Enable MKL-DNN acceleration if available
torch.backends.mkldnn.enabled = True


def step_scheduler(
    scheduler: optim.lr_scheduler._LRScheduler, val_loss: float = None
) -> None:
    """Step the learning rate scheduler based on its type.

    Args:
        scheduler: The learning rate scheduler to step
        val_loss: Optional validation loss for ReduceLROnPlateau scheduler
    """
    if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
        # OneCycleLR should be stepped per batch
        scheduler.step()
    elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        # ReduceLROnPlateau needs the validation loss
        if val_loss is None:
            raise ValueError(
                "val_loss must be provided for ReduceLROnPlateau scheduler"
            )
        scheduler.step(val_loss)
    else:
        # Other schedulers (StepLR, etc.) just need step()
        scheduler.step()


def split_train_validation_data(metadata_df, config, logger, wandb_logger=None):
    """Split the dataset into training and validation sets.

    Args:
        metadata_df: DataFrame containing metadata
        config: Configuration object
        logger: Logger instance
        wandb_logger: Optional WandB logger instance

    Returns:
        Tuple of (train_df, valid_df)
    """
    if config.DEV_MODE:
        # For DEV_MODE, use a pre-determined split with config.DEV_MODE_N_CLASSES specific classes
        class_counts = metadata_df["primary_label"].value_counts()
        min_n_samples = 4
        unique_labels = sorted(
            class_counts[class_counts >= min_n_samples].index.tolist()
        )[: config.DEV_MODE_N_CLASSES]
        metadata_df = metadata_df[metadata_df["primary_label"].isin(unique_labels)]
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        metadata_df["target"] = metadata_df["primary_label"].map(label2id)

        logger.info(
            f"DEV_MODE: Filtered out classes with less than {min_n_samples} samples. Training on {len(unique_labels)} classes: {unique_labels}"
        )

        if wandb_logger:
            wandb_logger.log({"unique_labels": unique_labels})

        # Create a simple train/test split for these classes
        train_df, valid_df = train_test_split(
            metadata_df,
            test_size=0.2,
            random_state=config.SEED,
            stratify=metadata_df["target"],
        )
        config.N_CLASSES = len(unique_labels)

    else:
        # Create label mapping for full dataset
        unique_labels = sorted(metadata_df["primary_label"].unique())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        metadata_df["target"] = metadata_df["primary_label"].map(label2id)

        train_df, valid_df = train_test_split(
            metadata_df,
            test_size=0.2,
            random_state=config.SEED,
            stratify=metadata_df["target"],
        )
        config.N_CLASSES = len(unique_labels)

    return train_df, valid_df


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config,
    epoch_idx: int,
    run_dir: Path,
    wandb_logger: WandbLogger,
    scaler: GradScaler = None,
) -> float:
    """Train one epoch with distributed training support."""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)

    # Create directories for spectrograms and attention outputs if they don't exist
    if config.LOCAL_RANK <= 0:  # Only create directories on main process
        spectrograms_dir = run_dir / "spectrograms"
        attention_dir = run_dir / "attention_outputs"
        spectrograms_dir.mkdir(exist_ok=True)
        attention_dir.mkdir(exist_ok=True)

    # Pre-allocate tensors for batch processing with rank-specific description
    rank_desc = f"Rank {config.LOCAL_RANK}" if config.DISTRIBUTED_TRAINING else "CPU"
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch_idx+1}/{config.EPOCHS} [{rank_desc}]",
        total=total_batches,
        unit="batch",
        position=(
            config.LOCAL_RANK if config.DISTRIBUTED_TRAINING else 0
        ),  # Stack progress bars vertically
        leave=True,  # Keep the progress bar after completion
    )

    # Initialize gradients at the start of the epoch
    # This is needed because we're using gradient accumulation:
    # 1. First zero_grad() ensures we start with clean gradients
    # 2. Subsequent zero_grad() calls after optimizer.step() clear gradients for the next accumulation cycle
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Move data to device in a single operation
        inputs = inputs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        # Forward pass with mixed precision if enabled
        if config.MIXED_PRECISION and scaler is not None:
            with autocast(device_type="cuda" if config.DEVICE == "cuda" else "cpu"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass with gradient scaling if mixed precision is enabled
        if config.MIXED_PRECISION and scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Update metrics with proper synchronization
        loss_value = loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        if config.DISTRIBUTED_TRAINING:
            # Synchronize loss across all processes
            loss_tensor = torch.tensor(loss_value, device=config.DEVICE)
            dist.all_reduce(
                loss_tensor, op=dist.ReduceOp.SUM, group=config.process_group
            )
            loss_value = loss_tensor.item() / config.WORLD_SIZE

        total_loss += loss_value

        # Save attention outputs for batches 0 to 5
        if batch_idx == 0 and config.SAVE_SPECTROGRAMS:
            # Get metadata for the current batch using the dataset's method
            batch_metadata = train_loader.dataset.get_batch_metadata(
                batch_idx * config.BATCH_SIZE, config.BATCH_SIZE
            )

            # Save attention outputs for all samples in the batch
            for idx in range(min(config.SAVE_SPECTROGRAMS_N_SAMPLES, len(inputs))):
                try:
                    attention_outputs = model.get_attention_outputs()[idx]
                    label_id = labels[idx].item()

                    # Get original filename and class name from metadata
                    sample_metadata = batch_metadata.iloc[idx]
                    original_filename = Path(sample_metadata["filename"]).stem
                    label = sample_metadata["primary_label"]

                    # Include filename, class label and start sec in the filename
                    filename = f"{label}_{original_filename}_epoch_{epoch_idx}_batch_{batch_idx}_sample_{idx}"

                    save_attention_outputs(
                        attention_outputs,
                        label=label,
                        class_id=label_id,
                        filename=filename,
                        batch_id=batch_idx,
                        sample_id=idx,
                        save_dir=attention_dir,
                        epoch_id=epoch_idx,
                        wandb_logger=wandb_logger,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to save attention outputs for sample {idx}: {e}"
                    )
                    continue

            # Clear attention outputs after saving to prevent memory leaks
            model.clear_attention_outputs()

        # Log batch metrics with reduced frequency and only on main process
        if batch_idx % 100 == 0 and config.LOCAL_RANK <= 0:
            wandb_logger.log(
                {
                    "epoch": epoch_idx + 1,
                    "batch": batch_idx + 1,
                    "batch_loss": loss_value,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        # Update progress bar with current loss
        pbar.set_postfix(loss=total_loss / (batch_idx + 1))

    # Calculate average loss over actual optimizer steps
    actual_steps = max(1, len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS)
    return total_loss / actual_steps


def validate(
    model: nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    config,
    epoch_idx: int,
    wandb_logger: WandbLogger,
) -> Tuple[float, float, float]:
    """Validate the model and compute metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Store predictions and labels for F1 calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = 100 * correct / total

    # Calculate F1 score (macro average across all classes)
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Calculate average loss
    avg_loss = total_loss / len(valid_loader)

    # Log validation metrics
    wandb_logger.log(
        {
            "epoch": epoch_idx,
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_f1": f1 * 100,  # Convert to percentage for consistency
        }
    )

    return avg_loss, accuracy, f1


def setup_distributed(config):
    """Initialize distributed training for Databricks cluster."""
    if config.DISTRIBUTED_TRAINING:
        # Get rank from environment
        if "RANK" in os.environ:
            config.LOCAL_RANK = int(os.environ["RANK"])
        else:
            # If not set, assume we're the driver (rank 0)
            config.LOCAL_RANK = 0

        # Set up the device
        torch.cuda.set_device(config.LOCAL_RANK)

        # Initialize the distributed process group
        dist.init_process_group(
            backend=config.DIST_BACKEND,
            init_method=f"tcp://{config.MASTER_ADDR}:{config.MASTER_PORT}",
            world_size=config.WORLD_SIZE,
            rank=config.LOCAL_RANK,
        )

        logger.info(
            f"Initialized distributed training on rank {config.LOCAL_RANK} "
            f"with {config.WORLD_SIZE} processes"
        )

        # Set up process group for all_reduce operations
        if not hasattr(config, "process_group"):
            config.process_group = dist.new_group()


def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_final_model(model, config, run_dir: Path, metadata_df, wandb_logger):
    """Save the final model with metadata for inference."""
    # Create timestamp for the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = run_dir / f"final_model_{timestamp}.pt"
    onnx_path = run_dir / f"final_model_{timestamp}.onnx"  # Initialize onnx_path here

    # Get the label mapping from metadata_df
    unique_labels = sorted(metadata_df["primary_label"].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    # Save model with essential data for inference
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": config.N_CLASSES,
            "class_mapping": {idx: label for label, idx in label2id.items()},
        },
        final_model_path,
    )
    logger.info(f"Saved final model to {final_model_path}")

    # Export to ONNX format for faster inference (optional)
    try:
        # Set model to eval mode and ensure it's on the correct device
        model.eval()
        model = model.to(config.DEVICE)

        # Create dummy input with correct shape and device
        dummy_input = torch.randn(1, 3, 224, 224, device=config.DEVICE)

        # Ensure model is in inference mode and all parameters are properly initialized
        with torch.no_grad():
            # Forward pass to ensure all BatchNorm layers are properly initialized
            _ = model(dummy_input)

            # Ensure all BatchNorm layers are properly initialized
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.eval()

            # Forward pass again after resetting BatchNorm stats
            _ = model(dummy_input)

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                keep_initializers_as_inputs=True,
            )
            logger.info(f"Successfully exported ONNX model to {onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to export ONNX model: {e}")

    # Save model to wandb if enabled
    if wandb_logger and wandb_logger.enabled:
        try:
            artifact = wandb_logger.wandb.Artifact(
                name=f"model-{run_dir.name}",
                type="model",
                description="Trained bird sound classification model",
            )
            # Add the PyTorch model file
            artifact.add_file(str(final_model_path))
            # Add the ONNX model if it was successfully exported
            if os.path.exists(onnx_path):
                artifact.add_file(str(onnx_path))
            # Log the artifact to wandb
            wandb_logger.wandb.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"Failed to save model to wandb: {e}")


def train(config, run_dir: Path):
    """Main training pipeline with distributed support."""
    # Initialize wandb logger only on main process
    wandb_logger = None
    if config.LOCAL_RANK <= 0:
        wandb_logger = WandbLogger(run_dir.name, run_dir)

    try:
        # Set random seeds
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        # Create directories for processed data
        processed_data_dir = config.PROCESSED_DATA_DIR
        # Check if processed data directory already exists and contains files
        if (
            processed_data_dir.exists()
            and (processed_data_dir / "train_metadata.csv").exists()
            and (processed_data_dir / "valid_metadata.csv").exists()
        ):
            logger.info(
                f"Using existing processed data directory: {processed_data_dir}"
            )
            run_specific_dir = processed_data_dir
            train_data_dir = run_specific_dir / "train"
            valid_data_dir = run_specific_dir / "valid"
        else:
            logger.info(f"Creating new processed data directory: {processed_data_dir}")
            # Create run-specific subdirectories for train and validation data
            run_specific_dir = processed_data_dir / run_dir.name
            run_specific_dir.mkdir(exist_ok=True, parents=True)

            train_data_dir = run_specific_dir / "train"
            valid_data_dir = run_specific_dir / "valid"
            train_data_dir.mkdir(exist_ok=True, parents=True)
            valid_data_dir.mkdir(exist_ok=True, parents=True)

        # Save metadata info file
        metadata_info_path = run_specific_dir / "dataset_metadata.json"

        # Check if we have existing processed data
        if metadata_info_path.exists():
            logger.info("Found existing processed data, loading metadata...")
            with open(metadata_info_path, "r") as f:
                metadata_info = json.load(f)
            train_processed_df = pd.read_csv(run_specific_dir / "train_metadata.csv")
            valid_processed_df = pd.read_csv(run_specific_dir / "valid_metadata.csv")
            config.N_CLASSES = metadata_info["n_classes"]
            logger.info(
                f"Loaded existing processed data with {config.N_CLASSES} classes"
            )
        else:
            # Load and process data as before
            metadata_df = load_metadata(config)

            # Split data
            train_df, valid_df = split_train_validation_data(
                metadata_df, config, logger, wandb_logger
            )

            # Preprocess and save datasets
            train_data_dir, train_processed_df = preprocess_and_save_dataset(
                train_df,
                config,
                train_data_dir,
                batch_size=config.BATCH_SIZE,
                n_workers=config.NUM_WORKERS,
            )
            valid_data_dir, valid_processed_df = preprocess_and_save_dataset(
                valid_df,
                config,
                valid_data_dir,
                batch_size=config.BATCH_SIZE,
                n_workers=config.NUM_WORKERS,
            )

            # Save metadata info and processed DataFrames
            metadata_info = {
                "n_classes": config.N_CLASSES,
                "train_size": len(train_processed_df),
                "valid_size": len(valid_processed_df),
                "batch_size": config.BATCH_SIZE,
                "seed": config.SEED,
                "dev_mode": config.DEV_MODE,
                "processing_date": datetime.now().isoformat(),
            }

            with open(metadata_info_path, "w") as f:
                json.dump(metadata_info, f, indent=4)

            # Save DataFrames with processed data info
            train_processed_df.to_csv(
                run_specific_dir / "train_metadata.csv", index=False
            )
            valid_processed_df.to_csv(
                run_specific_dir / "valid_metadata.csv", index=False
            )

            logger.info(f"Saved dataset metadata to {metadata_info_path}")
            logger.info(
                f"Saved train metadata to {run_specific_dir / 'train_metadata.csv'}"
            )
            logger.info(
                f"Saved valid metadata to {run_specific_dir / 'valid_metadata.csv'}"
            )

        # Initialize distributed training after preprocessing
        setup_distributed(config)

        # Create datasets with processed data
        train_dataset = BirdSoundDataset(
            train_processed_df,
            augmentation=get_transforms("train"),
            mode="train",
        )
        valid_dataset = BirdSoundDataset(
            valid_processed_df,
            augmentation=get_transforms("valid"),
            mode="valid",
        )

        # Create distributed samplers
        train_sampler = (
            DistributedSampler(train_dataset) if config.DISTRIBUTED_TRAINING else None
        )
        valid_sampler = (
            DistributedSampler(valid_dataset, shuffle=False)
            if config.DISTRIBUTED_TRAINING
            else None
        )

        # Create dataloaders with distributed settings
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        # Initialize model using the factory
        model = ModelFactory.create_model(
            model_config=config.model_config,
            num_classes=config.N_CLASSES,
        )

        # Move model to device and wrap with DDP
        model = model.to(config.DEVICE)
        if config.DEVICE == "cuda":
            model = model.to(memory_format=torch.channels_last)
            if config.DISTRIBUTED_TRAINING:
                model = DDP(
                    model, device_ids=[config.LOCAL_RANK], find_unused_parameters=True
                )

        # Initialize gradient scaler for mixed precision training
        scaler = (
            GradScaler() if config.MIXED_PRECISION and config.DEVICE == "cuda" else None
        )

        # Initialize training components with optimized settings
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = optim.AdamW(  # Changed to AdamW for better weight decay
            model.parameters(),
            lr=config.LR_MAX,
            weight_decay=0.01,  # Add weight decay for regularization
            eps=1e-8,
        )

        # Calculate total steps and setup cosine annealing scheduler
        num_steps_per_epoch = len(train_loader)
        T_max = config.EPOCHS * num_steps_per_epoch  # Total number of steps

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,  # Total number of steps
            eta_min=config.LR_MAX * 1e-4,  # Minimum learning rate
            verbose=False,
        )

        # Training loop
        best_val_f1 = 0.0
        best_model_path = None
        patience_counter = 0

        for epoch_idx in range(config.EPOCHS):
            if config.DISTRIBUTED_TRAINING:
                train_sampler.set_epoch(epoch_idx)

            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                config,
                epoch_idx,
                run_dir,
                wandb_logger,
                scaler,
            )

            # Validation phase
            val_loss, val_accuracy, val_f1 = validate(
                model,
                valid_loader,
                criterion,
                config,
                epoch_idx,
                wandb_logger,
            )

            # Log metrics only on main process
            if config.LOCAL_RANK <= 0:
                wandb_logger.log(
                    {
                        "epoch": epoch_idx,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "val_f1": val_f1 * 100,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )

                # Save checkpoints only on main process
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_path = run_dir / f"best_model_epoch_{epoch_idx+1}.pt"
                    torch.save(
                        {
                            "epoch": epoch_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": (
                                scaler.state_dict() if scaler is not None else None
                            ),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                            "val_f1": val_f1,
                        },
                        best_model_path,
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Regular checkpoint saving with reduced frequency
                if (epoch_idx + 1) % 10 == 0:  # Save every 10 epochs instead of 5
                    checkpoint_path = run_dir / f"model_epoch_{epoch_idx+1}.pt"
                    torch.save(
                        {
                            "epoch": int(epoch_idx),
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": (
                                scaler.state_dict() if scaler is not None else None
                            ),
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss),
                            "val_accuracy": float(val_accuracy),
                            "val_f1": float(val_f1) if "val_f1" in locals() else None,
                        },
                        checkpoint_path,
                    )

                # Early stopping
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f"Early stopping triggered after {epoch_idx + 1} epochs"
                    )
                    break

        # Final model saving only on main process
        if config.LOCAL_RANK <= 0:
            save_final_model(model, config, run_dir, metadata_df, wandb_logger)

    finally:
        # Cleanup distributed training
        cleanup_distributed()
        if wandb_logger:
            wandb_logger.finish()
