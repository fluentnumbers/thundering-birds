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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.dataset import BirdSoundDataset, collate_fn, get_transforms
from src.data.preprocessing import load_metadata, preprocess_dataset
from src.models.efficientnet import create_model
from src.models.model_factory import ModelFactory
from src.utils.logger import WandbLogger, setup_logger
from src.utils.visualization import save_attention_outputs, save_melspectrogram


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config,
    epoch: int,
    run_dir: Path,
    wandb_logger: WandbLogger,
    metadata_df: pd.DataFrame,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)

    # Create directory for spectrograms and attention outputs if they don't exist
    spectrograms_dir = run_dir / "spectrograms"
    attention_dir = run_dir / "attention_outputs"
    spectrograms_dir.mkdir(exist_ok=True)
    attention_dir.mkdir(exist_ok=True)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Save attention outputs for batches 0 to 5
        if step <= 5 and config.SAVE_SPECTROGRAMS:
            # Save attention outputs for all samples in the batch
            for idx in range(len(inputs)):
                attention_outputs = model.get_attention_outputs()[idx]
                label_id = labels[idx].item()
                label = metadata_df[metadata_df["target"] == label_id][
                    "primary_label"
                ].iloc[0]

                # Get the original filename from metadata_df
                # Use the label_id to find the corresponding row in metadata_df
                row = metadata_df[metadata_df["target"] == label_id].iloc[0]
                original_filename = row.get("filename", "unknown")

                # Include filename, class label and start sec in the filename
                filename = f"{original_filename}_{label}_batch_{step}_sample_{idx}"

                save_attention_outputs(
                    attention_outputs,
                    label=label,
                    class_id=label_id,
                    filename=filename,
                    chunk_id=idx,
                    batch_id=step,
                    sample_id=idx,
                    save_dir=attention_dir,
                    epoch=epoch,
                    step=step,
                    wandb_logger=wandb_logger,
                )

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()

        # Log batch metrics with reduced frequency
        if step % 50 == 0:  # Log every 50 steps
            wandb_logger.log(
                {
                    "batch": step + 1,
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        pbar.set_postfix(loss=total_loss / (step + 1))

    return total_loss / len(train_loader)


def train(config, run_dir: Path):
    """Main training pipeline."""
    # Initialize wandb logger
    wandb_logger = WandbLogger(run_dir.name, run_dir)

    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Load data
    metadata_df = load_metadata(config)

    # Split data
    if config.DEV_MODE:
        train_df, valid_df = train_test_split(
            metadata_df, test_size=0.2, random_state=config.SEED
        )
    else:
        train_df, valid_df = train_test_split(
            metadata_df,
            test_size=0.2,
            random_state=config.SEED,
            stratify=metadata_df["primary_label"],
        )

    # Precompute datasets
    train_spectrograms, train_labels = preprocess_dataset(train_df, config)
    valid_spectrograms, valid_labels = preprocess_dataset(valid_df, config)

    # Create datasets with precomputed data
    train_dataset = BirdSoundDataset(
        train_spectrograms,
        train_labels,
        augmentation=get_transforms("train"),
        mode="train",
    )
    valid_dataset = BirdSoundDataset(
        valid_spectrograms,
        valid_labels,
        augmentation=get_transforms("valid"),
        mode="valid",
    )

    # Create dataloaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Initialize model using the factory
    model = ModelFactory.create_model(
        model_config=config.model_config,
        num_classes=config.N_CLASSES,
    )
    model = model.to(config.DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR_MAX)

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
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            epoch,
            run_dir,
            wandb_logger,
            metadata_df,
        )

        # Log epoch metrics
        wandb_logger.log({"epoch": epoch, "train_loss": train_loss})

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = run_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                },
                checkpoint_path,
            )

    # Save final model for Kaggle submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = run_dir / f"final_model_{timestamp}.pt"

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
