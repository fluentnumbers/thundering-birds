import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import albumentations as albu
import cv2
import librosa
import matplotlib.pyplot as plt
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

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create timestamped run directory
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"basic_pipeline_{run_timestamp}"
run_dir = log_dir / run_name
run_dir.mkdir(exist_ok=True)

# Setup logging with both file and console handlers
log_filename = "training.log"
log_filepath = run_dir / log_filename

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create formatters and handlers
file_handler = logging.FileHandler(log_filepath)
console_handler = logging.StreamHandler()

# Define format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file_formatter = logging.Formatter(log_format)
console_formatter = logging.Formatter(log_format)

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add new handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Silence some logs
logging.getLogger("numba.core").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger.info(f"Logging to {log_filepath}")


class Config:
    """Configuration class for the pipeline."""

    def __init__(self):
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEV_MODE = False  # Take only fraction of the dataset, for dev
        self.DEV_MODE_N_SAMPLES = 300  # Number of samples to use in development mode

        # Data config
        self.DATA_ROOT = Path("data/birdclef-2025")
        self.FS = 32000  # Sample rate
        self.N_FFT = 1024  # FFT size
        self.WIN_LAP = 512  # Window overlap
        self.MIN_FREQ = 50
        self.MAX_FREQ = 14000
        self.N_MELS = 128

        self.segment_duration = 5  # seconds
        self.nsamples = self.segment_duration * self.FS
        self.padmode = "constant"
        self.ufoldoverlap = self.nsamples // 2  # 2.5 seconds overlap

        # Training config
        self.BATCH_SIZE = 32
        self.NUM_WORKERS = 1
        self.LR_MAX = 3e-4
        self.EPOCHS = 20

        # Model config
        self.N_CLASSES = None  # Will be set after data loading


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

    # For development: limit to a fractions of  samples while maintaining class distribution
    if config.DEV_MODE:
        metadata_df = metadata_df.groupby("primary_label", group_keys=False).apply(
            lambda x: x.sample(
                n=min(config.DEV_MODE_N_SAMPLES // config.N_CLASSES, len(x))
            )
        )
        logger.info(f"Development mode: Limited dataset to {len(metadata_df)} samples")

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
            n_mels=config.N_MELS,
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


def preprocess_dataset(
    metadata_df: pd.DataFrame, config: Config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute all segments and mel spectrograms for the dataset.

    Args:
        metadata_df: DataFrame containing metadata
        config: Configuration object

    Returns:
        Tuple of (precomputed_spectrograms, labels) where spectrograms has shape [n_total_segments, 3, 224, 224]
        and labels has shape [n_total_segments]
    """
    logger.info("Starting dataset preprocessing...")
    mel_transform = MelSpectrogramTransform(config)

    # Initialize lists to store results
    all_spectrograms = []
    all_labels = []
    total_size_mb = 0

    # Process each audio file
    for idx, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Processing audio files"
    ):
        # Load audio
        audio_data, _ = librosa.load(row.filepath, sr=config.FS)
        audio_tensor = torch.tensor(audio_data)

        # Pad if necessary
        nsamples = audio_tensor.shape[-1]
        rsamples = nsamples % config.nsamples
        audio_tensor = torch.nn.functional.pad(
            audio_tensor, (0, config.nsamples - rsamples), mode=config.padmode
        )

        # Calculate number of segments
        n_segments = (len(audio_tensor) - config.nsamples) // config.ufoldoverlap + 1
        logger.debug(f"File {row.filename}: {n_segments} segments")

        # Process each segment
        for segment_idx in range(n_segments):
            start_idx = segment_idx * config.ufoldoverlap
            audio_segment = audio_tensor[start_idx : start_idx + config.nsamples]

            # Convert to mel spectrogram
            mel_spec = mel_transform(audio_segment)

            # Resize to 224x224
            mel_spec = torch.tensor(cv2.resize(mel_spec.numpy(), (224, 224)))

            # Add channel dimension and repeat to 3 channels
            mel_spec = mel_spec.unsqueeze(0).repeat(3, 1, 1)

            # Append to lists
            all_spectrograms.append(mel_spec)
            all_labels.append(row.target)

            # Log memory usage every 1000 segments
            if len(all_spectrograms) % 1000 == 0:
                current_size = sum(
                    spec.element_size() * spec.nelement() for spec in all_spectrograms
                )
                current_size_mb = current_size / (1024 * 1024)
                total_size_mb = current_size_mb
                logger.info(
                    f"Processed {len(all_spectrograms)} segments. Current memory usage: {current_size_mb:.2f} MB"
                )

    # Convert lists to tensors
    logger.info("Converting lists to tensors...")
    spectrograms = torch.stack(all_spectrograms)
    labels = torch.tensor(all_labels)

    # Log final memory usage
    final_size = spectrograms.element_size() * spectrograms.nelement()
    final_size_mb = final_size / (1024 * 1024)
    logger.info(f"Preprocessing complete. Final dataset size: {final_size_mb:.2f} MB")
    logger.info(
        f"Dataset shape: Spectrograms {spectrograms.shape}, Labels {labels.shape}"
    )

    return spectrograms, labels


class BirdSoundDataset(Dataset):
    """Dataset class for bird sound spectrograms using precomputed data."""

    def __init__(
        self,
        spectrograms: torch.Tensor,
        labels: torch.Tensor,
        augmentation=None,
        mode="train",
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augmentation = augmentation
        self.mode = mode
        self.total_samples = len(spectrograms)
        logger.info(f"Created {mode} dataset with {self.total_samples} samples")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = self.spectrograms[index]
        label = self.labels[index]

        # Apply augmentations if any
        if self.augmentation and self.mode == "train":
            spec = torch.tensor(self.augmentation(image=spec.numpy())["image"])

        return spec, label


def collate_fn(batch):
    """Custom collate function to handle batching of spectrograms.

    Args:
        batch: List of tuples (mel_spec, label) where mel_spec has shape [3, H, W]
              and label is a scalar

    Returns:
        Tuple of (inputs, labels) where inputs has shape [batch_size, 3, H, W]
        and labels has shape [batch_size]
    """
    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Stack inputs and labels
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)

    return inputs, labels


def get_transforms(mode: str) -> albu.Compose:
    """Get augmentation transforms based on mode."""
    return None


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


class WandbLogger:
    """Wrapper for wandb logging functionality."""

    def __init__(self, run_name: str, run_dir: Path):
        self.enabled = False
        try:
            import wandb

            self.wandb = wandb
            self.wandb.init(
                project="bird-sound-classification",
                name=run_name,
                dir=str(run_dir),
            )
            self.enabled = True
            logger.info("Initialized wandb logging")
        except ImportError:
            logger.info("wandb not available, continuing without wandb logging")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    def log(self, data: dict) -> None:
        """Log data to wandb if available."""
        if self.enabled:
            try:
                self.wandb.log(data)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

    def log_image(self, image: np.ndarray, caption: str, **kwargs) -> None:
        """Log image to wandb if available."""
        if self.enabled:
            try:
                self.wandb.log(
                    {"images": self.wandb.Image(image, caption=caption), **kwargs}
                )
            except Exception as e:
                logger.warning(f"Failed to log image to wandb: {e}")

    def finish(self) -> None:
        """Finish wandb run if available."""
        if self.enabled:
            try:
                self.wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb: {e}")


def save_melspectrogram(
    spec: np.ndarray,
    label: str,
    class_id: int,
    filename: str,
    chunk_id: int,
    batch_id: int,
    sample_id: int,
    save_dir: Path,
    epoch: int,
    step: int,
    wandb_logger: WandbLogger,
) -> None:
    """Save mel spectrogram as an image and optionally log to wandb.

    Args:
        spec: Mel spectrogram array
        label: Class label (species name)
        class_id: Numeric class ID
        filename: Original audio filename
        chunk_id: ID of the chunk in the audio file
        batch_id: ID of the batch
        sample_id: ID of the sample in the batch
        save_dir: Directory to save the image
        epoch: Current epoch number
        step: Current step number
        wandb_logger: WandbLogger instance
    """
    # Create figure
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar()
    plt.title(f"Label: {label} (ID: {class_id})\nFile: {filename}\nChunk: {chunk_id}")
    plt.tight_layout()

    # Create filename with all components
    img_filename = (
        f"epoch{epoch:02d}_batch{batch_id:03d}_sample{sample_id:03d}_"
        f"class{class_id:03d}_{label}_chunk{chunk_id:03d}.png"
    )

    # Save to local filesystem
    plt.savefig(save_dir / img_filename)
    plt.close()
    logger.debug(f"Saved spectrogram to {img_filename}")

    # Log to wandb if available
    wandb_logger.log_image(
        spec,
        f"Label: {label} (ID: {class_id}), File: {filename}, Chunk: {chunk_id}",
        epoch=epoch,
        step=step,
        label=label,
        class_id=class_id,
        filename=filename,
        chunk_id=chunk_id,
        batch_id=batch_id,
        sample_id=sample_id,
    )


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: Config,
    epoch: int,
    run_dir: Path,
    wandb_logger: WandbLogger,
    metadata_df: pd.DataFrame,  # Add metadata_df parameter
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)

    # Create directory for spectrograms if it doesn't exist
    spectrograms_dir = run_dir / "spectrograms"
    spectrograms_dir.mkdir(exist_ok=True)

    # Log epoch separator and info
    logger.info("-" * 80)
    logger.info(
        f"Starting epoch {epoch+1}/{config.EPOCHS} with {total_batches} batches "
        f"of size {config.BATCH_SIZE}"
    )
    logger.info("-" * 80)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
    for step, (inputs, labels) in enumerate(pbar):
        # Log batch separator and dimensions
        logger.debug("=" * 40)
        logger.debug(
            f"Batch (step {step + 1}/{total_batches}) - "
            f"Inputs: {inputs.shape} ({inputs.dtype}), "
            f"Labels: {labels.shape} ({labels.dtype})"
        )

        # Save random spectrograms from the first batch of each epoch
        if step == 0:
            # Get 3 random indices from the current batch
            random_indices = torch.randperm(len(inputs))[:3]

            # Save spectrograms
            for idx, random_idx in enumerate(random_indices):
                spec = inputs[random_idx]
                label_id = labels[random_idx].item()

                # Get the original label from metadata
                label = metadata_df[metadata_df["target"] == label_id][
                    "primary_label"
                ].iloc[0]

                # Convert to numpy and save
                spec_np = spec[0].numpy()  # Take first channel since they're identical
                save_melspectrogram(
                    spec_np,
                    label=label,
                    class_id=label_id,
                    filename=f"batch_{step}_sample_{random_idx.item()}",
                    chunk_id=idx,
                    batch_id=step,
                    sample_id=random_idx.item(),
                    save_dir=spectrograms_dir,
                    epoch=epoch,
                    step=step,
                    wandb_logger=wandb_logger,
                )

        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        logger.info(
            f"Batch {step + 1}/{total_batches} - "
            f"Model output dimensions: {outputs.shape}"
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
            wandb_logger.log(
                {
                    "batch": step + 1,
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        pbar.set_postfix(loss=total_loss / (step + 1))
        logger.debug("=" * 40)

    # Log end of epoch separator
    logger.info("-" * 80)
    logger.info(f"Completed epoch {epoch+1}/{config.EPOCHS}")
    logger.info("-" * 80)

    return total_loss / len(train_loader)


def main():
    """Main training pipeline."""
    # Initialize wandb logger
    wandb_logger = WandbLogger(run_name, run_dir)

    # Initialize config
    config = Config()
    logger.info(f"Using device: {config.DEVICE}")

    # Save config to run directory
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    # Convert Path objects to strings
    config_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items()
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    logger.info(f"Saved config to {run_dir / 'config.json'}")

    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Load data
    metadata_df = load_metadata(config)
    logger.info(f"Loaded metadata with {len(metadata_df)} samples")

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
    logger.info(
        f"Split data into {len(train_df)} train and {len(valid_df)} validation samples"
    )

    # Precompute datasets
    logger.info("Precomputing training dataset...")
    train_spectrograms, train_labels = preprocess_dataset(train_df, config)

    logger.info("Precomputing validation dataset...")
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
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            epoch,
            run_dir,
            wandb_logger,
            metadata_df,  # Pass metadata_df to train_epoch
        )

        # Log epoch metrics
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} - Loss: {train_loss:.4f}")
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
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training completed!")
    wandb_logger.finish()


if __name__ == "__main__":
    main()
