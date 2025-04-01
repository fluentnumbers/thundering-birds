import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import LOGS_DIR
from src.data.dataset import BirdSoundDataset, collate_fn, get_transforms
from src.data.preprocessing import load_metadata, preprocess_and_save_dataset
from src.models.efficientnet import create_model
from src.models.model_factory import ModelFactory
from src.utils.logger import WandbLogger, setup_logger
from src.utils.visualization import save_attention_outputs, save_melspectrogram

logger = setup_logger(__name__)


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
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)

    # Create directories for spectrograms and attention outputs if they don't exist
    spectrograms_dir = run_dir / "spectrograms"
    attention_dir = run_dir / "attention_outputs"
    spectrograms_dir.mkdir(exist_ok=True)
    attention_dir.mkdir(exist_ok=True)

    # Pre-allocate tensors for batch processing
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch_idx+1}/{config.EPOCHS}",
        total=total_batches,
        unit="batch",
    )
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Move data to device in a single operation
        inputs = inputs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Save attention outputs for the same 3 samples every epoch
        if batch_idx == 0 and config.SAVE_SPECTROGRAMS:
            # Only save the first 3 samples from the first batch
            for idx in range(min(config.SAVE_SPECTROGRAMS_N_SAMPLES, len(inputs))):
                try:
                    attention_outputs = model.get_attention_outputs()[idx]
                    label_id = labels[idx].item()

                    # Get label from the dataset's DataFrame
                    sample_df = train_loader.dataset.metadata_df.iloc[idx]
                    label = sample_df["label"]
                    filename = sample_df["filename"]
                    original_filename = Path(filename).stem

                    # Include filename, class label and start sec in the filename
                    filename = f"{original_filename}_{label}_epoch_{epoch_idx}_batch_{batch_idx}_sample_{idx}"

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

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()

        # Log batch metrics with reduced frequency
        if batch_idx % 100 == 0:  # Log every 100 batches instead of 50
            wandb_logger.log(
                {
                    "epoch": epoch_idx + 1,
                    "batch": batch_idx + 1,
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        pbar.set_postfix(loss=total_loss / (batch_idx + 1))

    return total_loss / len(train_loader)


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


def train(config, run_dir: Path):
    """Main training pipeline."""
    # Initialize wandb logger with the existing run_dir
    wandb_logger = WandbLogger(run_dir.name, run_dir)

    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Create directories for processed data
    processed_data_dir = config.PROCESSED_DATA_DIR
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Create run-specific subdirectories for train and validation data
    run_specific_dir = processed_data_dir / run_dir.name
    run_specific_dir.mkdir(exist_ok=True)

    train_data_dir = run_specific_dir / "train"
    valid_data_dir = run_specific_dir / "valid"
    train_data_dir.mkdir(exist_ok=True)
    valid_data_dir.mkdir(exist_ok=True)

    # Load data
    metadata_df = load_metadata(config)

    # Split data
    if config.DEV_MODE:
        # For DEV_MODE, use a pre-determined split with 5 specific classes
        # Define a fixed list of 5 classes to use in development mode
        class_counts = metadata_df["primary_label"].value_counts()
        min_n_samples = 4
        classes_to_keep = class_counts[class_counts >= min_n_samples].index.tolist()
        metadata_df = metadata_df[metadata_df["primary_label"].isin(classes_to_keep)]
        unique_labels = sorted(metadata_df["primary_label"].unique())[
            : config.DEV_MODE_N_CLASSES
        ]
        metadata_df = metadata_df[metadata_df["primary_label"].isin(unique_labels)]

        # Create a new label mapping for the filtered classes
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        metadata_df["target"] = metadata_df["primary_label"].map(label2id)

        logger.info(
            f"Filtered out classes with less than {min_n_samples} samples. Remaining classes: {len(unique_labels)}"
        )

        # Log the selected classes for reproducibility
        logger.info(
            f"DEV_MODE: Using {len(unique_labels)} classes for development: {unique_labels}"
        )
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

    # Preprocess and save datasets
    train_data_dir, train_processed_df = preprocess_and_save_dataset(
        train_df, config, train_data_dir, batch_size=config.BATCH_SIZE
    )
    valid_data_dir, valid_processed_df = preprocess_and_save_dataset(
        valid_df, config, valid_data_dir, batch_size=config.BATCH_SIZE
    )

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

    # Create dataloaders with optimized settings for CPU
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,  # Enable pin_memory for faster data transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
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
    model = model.to(config.DEVICE)

    # Enable torch.backends.cudnn benchmarking for faster training
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    # Initialize training components with optimized settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR_MAX, eps=1e-8)

    # Calculate total steps for scheduler
    total_steps = config.EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR_MAX,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=1e3,
        final_div_factor=1e4,
    )

    # Training loop
    best_val_f1 = 0.0
    best_model_path = None
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch_idx in range(config.EPOCHS):
        # Training phase
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
        )

        # Validation phase with F1 score
        val_loss, val_accuracy, val_f1 = validate(
            model,
            valid_loader,
            criterion,
            config,
            epoch_idx,
            wandb_logger,
        )

        # Log epoch metrics
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

        # Save best model (now considering F1 score)
        if val_f1 > best_val_f1:  # Change criterion to F1 score
            best_val_f1 = val_f1
            best_model_path = run_dir / f"best_model_epoch_{epoch_idx+1}.pt"
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
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
                    "epoch": epoch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1": val_f1,
                },
                checkpoint_path,
            )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch_idx + 1} epochs")
            break

    # Save final model for Kaggle submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = run_dir / f"final_model_{timestamp}.pt"

    # Use the best model for final save
    if best_model_path is not None and best_model_path.exists():
        best_checkpoint = torch.load(best_model_path)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        logger.info(
            f"Using best model from epoch {best_checkpoint['epoch'] + 1} with validation accuracy {best_checkpoint['val_accuracy']:.2f}%"
        )

    # Get the label mapping from metadata_df
    unique_labels = sorted(metadata_df["primary_label"].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    # Save only the essential data in a format that can be loaded without source code
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": config.N_CLASSES,  # Save number of classes directly
            "class_mapping": {
                idx: label for label, idx in label2id.items()
            },  # Reverse mapping for inference
        },
        final_model_path,
    )

    # Export to ONNX format for faster inference (optional)
    try:
        dummy_input = torch.randn(1, 3, 224, 224, device=config.DEVICE)
        onnx_path = run_dir / "model.onnx"
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
        )
    except Exception as e:
        logging.warning(f"Failed to export ONNX model: {e}")

    # Save model to wandb
    if wandb_logger.enabled:
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
            logging.warning(f"Failed to save model to wandb: {e}")

    wandb_logger.finish()
