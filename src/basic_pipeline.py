import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import albumentations as albu
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Setup logging with both file and console handlers
log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = log_dir / log_filename

# Create formatters and handlers
file_handler = logging.FileHandler(log_filepath)
console_handler = logging.StreamHandler()

# Define format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file_formatter = logging.Formatter(log_format)
console_formatter = logging.Formatter(log_format)

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Setup root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Logging to {log_filepath}")


class Config:
    """Configuration class for the pipeline."""

    def __init__(self):
        self.SEED = 2024
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MIXED_PRECISION = False

        # Data config
        self.DATA_ROOT = Path("data/birdclef-2025")
        self.FS = 32000  # Sample rate
        self.N_FFT = 1095  # FFT size
        self.WIN_SIZE = 412  # Window size
        self.WIN_LAP = 100  # Window overlap
        self.MIN_FREQ = 40
        self.MAX_FREQ = 15000
        self.USE_XYMASKING = True

        # Training config
        self.BATCH_SIZE = 32
        self.NUM_WORKERS = 1
        self.LR_MAX = 3e-4
        self.EPOCHS = 2

        # Model config
        self.N_CLASSES = None  # Will be set after data loading

        # Audio config
        self.nsamples = 160000  # 5 seconds * 32000 Hz
        self.padmode = "constant"
        self.ufoldoverlap = 80000  # 2.5 seconds overlap


def load_metadata(config: Config) -> pd.DataFrame:
    """Load and prepare metadata."""
    logger.info("Loading metadata...")
    metadata_df = pd.read_csv(f"{config.DATA_ROOT}/train.csv")

    # Add full filepath
    metadata_df["filepath"] = metadata_df["filename"].apply(
        lambda x: os.path.join(config.DATA_ROOT, "train_audio", x)
    )

    # Get unique labels and create label mapping
    labels = sorted(metadata_df["primary_label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    metadata_df["target"] = metadata_df["primary_label"].map(label2id)

    config.N_CLASSES = len(labels)
    logger.info(f"Found {config.N_CLASSES} unique classes")

    return metadata_df


class MelSpectrogramTransform:
    """Computes the Mel Spectogram of an audio sample."""

    def __init__(self, config: Config):
        self.to_melspectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.FS,
            n_fft=config.N_FFT,
            hop_length=config.WIN_LAP,
            f_max=config.MAX_FREQ,
            f_min=config.MIN_FREQ,
            n_mels=128,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.etol = 1e-8

    def __call__(self, audio_sample: torch.Tensor) -> torch.Tensor:
        if torch.isnan(audio_sample).any():
            mean_value = torch.nanmean(audio_sample)
            audio_sample = torch.nan_to_num(audio_sample, nan=mean_value)

        output = self.to_melspectogram(audio_sample)
        output = librosa.power_to_db(output, ref=np.max)
        output = (output - output.min()) / (output.max() - output.min() + self.etol)

        return torch.tensor(output)


class BirdSoundDataset(Dataset):
    """Dataset class for bird sound spectrograms."""

    def __init__(
        self, metadata: pd.DataFrame, config: Config, augmentation=None, mode="train"
    ):
        self.metadata = metadata
        self.config = config
        self.augmentation = augmentation
        self.mode = mode
        self.total_samples = len(metadata)
        self.sample_count = 0  # Track actual processing order
        self.mel_transform = MelSpectrogramTransform(config)
        logger.info(f"Created {mode} dataset with {self.total_samples} audio files")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        self.sample_count += 1

        # Load audio
        audio_data, _ = librosa.load(row.filepath, sr=self.config.FS)
        logger.debug(
            f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
            f"Raw audio shape: {audio_data.shape}, "
            f"duration: {len(audio_data)/self.config.FS:.2f}s"
        )

        # Convert to tensor and pad if necessary
        audio_tensor = torch.tensor(audio_data)
        nsamples = audio_tensor.shape[-1]
        rsamples = nsamples % self.config.nsamples

        # Pad the audio
        audio_tensor = torch.nn.functional.pad(
            audio_tensor, (0, self.config.nsamples - rsamples), mode=self.config.padmode
        )
        logger.debug(
            f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
            f"Audio after padding: {audio_tensor.shape}, "
            f"duration: {len(audio_tensor)/self.config.FS:.2f}s"
        )

        # Unfold into 5-second segments with overlap
        audio_segments = audio_tensor.unfold(
            dimension=-1, size=self.config.nsamples, step=self.config.ufoldoverlap
        )
        num_segments = audio_segments.shape[0]
        logger.debug(
            f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
            f"Audio segments shape: {audio_segments.shape}, "
            f"number of segments: {num_segments}"
        )

        # Randomly select one segment for training
        segment_idx = torch.randint(0, num_segments, (1,)).item()
        audio_segment = audio_segments[segment_idx]
        logger.debug(
            f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
            f"Selected segment {segment_idx}/{num_segments} shape: {audio_segment.shape}"
        )

        # Convert segment to mel spectrogram
        mel_spec = self.mel_transform(audio_segment)
        logger.debug(
            f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
            f"Mel spectrogram shape: {mel_spec.shape}"
        )

        # Resize spectrogram to 224x224 for EfficientNet
        mel_spec = torch.tensor(cv2.resize(mel_spec.numpy(), (224, 224)))
        logger.debug(
            f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
            f"Resized spectrogram shape: {mel_spec.shape}"
        )

        # Apply augmentations if any
        if self.augmentation:
            mel_spec = torch.tensor(self.augmentation(image=mel_spec.numpy())["image"])
            logger.debug(
                f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
                f"Augmented spectrogram shape: {mel_spec.shape}"
            )

        # Add channel dimension and repeat to 3 channels
        mel_spec = mel_spec.unsqueeze(0).repeat(3, 1, 1)
        logger.debug(
            f"[{self.mode}] Sample ({self.sample_count}/{self.total_samples}, idx={index}) - "
            f"Final tensor shape: {mel_spec.shape}"
        )

        return mel_spec, torch.tensor(row.target, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function to handle batching of spectrograms."""
    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Stack inputs and labels
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)

    return inputs, labels


def get_transforms(mode: str) -> albu.Compose:
    """Get augmentation transforms based on mode."""
    return None
    if mode == "train":
        return albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                (
                    albu.XYMasking(
                        p=0.3,
                        num_masks_x=(1, 3),
                        num_masks_y=(1, 3),
                        mask_x_length=(1, 10),
                        mask_y_length=(1, 20),
                    )
                    if Config().USE_XYMASKING
                    else albu.NoOp()
                ),
            ]
        )
    return albu.Compose([])


class EfficientNetModel(nn.Module):
    """EfficientNet model for bird sound classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.efficientnet = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            pretrained=True,
        )
        self.efficientnet.classifier.fc = nn.Linear(
            self.efficientnet.classifier.fc.in_features, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: Config,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)

    # Log dataloader info at the start of epoch
    logger.info(
        f"Starting epoch {epoch+1}/{config.EPOCHS} with {total_batches} batches "
        f"of size {config.BATCH_SIZE}"
    )

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
    for step, (inputs, labels) in enumerate(pbar):
        # Log batch dimensions
        logger.debug(
            f"Batch (step {step + 1}/{total_batches}) - "
            f"Inputs: {inputs.shape} ({inputs.dtype}), "
            f"Labels: {labels.shape} ({labels.dtype})"
        )

        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        logger.info(
            f"Batch 1/{total_batches} - " f"Model output dimensions: {outputs.shape}"
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
            logger.info(
                f"Epoch {epoch+1}/{config.EPOCHS}, "
                f"Batch {step + 1}/{total_batches}, "
                f"Loss: {loss.item():.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            wandb.log(
                {
                    "batch": step + 1,
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        pbar.set_postfix(loss=total_loss / (step + 1))

    return total_loss / len(train_loader)


def main():
    """Main training pipeline."""
    # Initialize wandb
    wandb.init(project="bird-sound-classification")

    # Initialize config
    config = Config()
    logger.info(f"Using device: {config.DEVICE}")

    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Load data
    metadata_df = load_metadata(config)
    logger.info(f"Loaded metadata with {len(metadata_df)} samples")

    # Split data
    train_df, valid_df = train_test_split(
        metadata_df,
        test_size=0.2,
        random_state=config.SEED,
        stratify=metadata_df["primary_label"],
    )
    logger.info(
        f"Split data into {len(train_df)} train and {len(valid_df)} validation samples"
    )

    # Create datasets
    train_dataset = BirdSoundDataset(
        train_df, config, augmentation=get_transforms("train"), mode="train"
    )
    valid_dataset = BirdSoundDataset(
        valid_df, config, augmentation=get_transforms("valid"), mode="valid"
    )

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    logger.info(
        f"Created dataloaders - Train: {len(train_loader)} batches, "
        f"Valid: {len(valid_loader)} batches"
    )

    # Initialize model
    model = EfficientNetModel(config.N_CLASSES)
    model = model.to(config.DEVICE)
    logger.info(f"Initialized model with {config.N_CLASSES} output classes")

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
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
    logger.info("Starting training...")
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, config, epoch
        )

        # Log epoch metrics
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} - Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                },
                f"checkpoints/model_epoch_{epoch+1}.pt",
            )

    logger.info("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()
