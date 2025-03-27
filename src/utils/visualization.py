import io
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch

from src.utils.logger import WandbLogger


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
    """Save mel spectrogram as an image and optionally log to wandb."""
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


def save_attention_outputs(
    attention_outputs: torch.Tensor,
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
    """Save attention outputs as images and optionally log to wandb."""
    # Convert to numpy and normalize for visualization
    attention_np = attention_outputs.cpu().numpy()

    # Create figure with 3 subplots for each channel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original spectrogram
    axes[0].imshow(attention_np[0], aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Original Spectrogram")
    axes[0].axis("off")

    # Plot first attention channel
    axes[1].imshow(attention_np[1], aspect="auto", origin="lower", cmap="hot")
    axes[1].set_title("Attention Channel 1")
    axes[1].axis("off")

    # Plot second attention channel
    axes[2].imshow(attention_np[2], aspect="auto", origin="lower", cmap="magma")
    axes[2].set_title("Attention Channel 2")
    axes[2].axis("off")

    plt.suptitle(
        f"Epoch {epoch} - Label: {label} (ID: {class_id})\nFile: {filename}\nChunk: {chunk_id}"
    )
    plt.tight_layout()

    # Create filename with all components
    img_filename = f"attention_{filename}_epoch{epoch:02d}_{label}(id:{class_id})_batch{batch_id:03d}_sample{sample_id:03d}_chunk{chunk_id:03d}.png"

    # Save to local filesystem
    plt.savefig(save_dir / img_filename)
    plt.close()

    # Log each channel separately to wandb with different colormaps and normalization
    channel_configs = [
        {"name": "original", "cmap": "viridis", "normalize": False},
        {"name": "attention1", "cmap": "hot", "normalize": True},
        {"name": "attention2", "cmap": "magma", "normalize": True},
    ]

    for channel_idx, config in enumerate(channel_configs):
        plt.figure(figsize=(10, 4))

        # Get the channel data
        channel_data = attention_np[channel_idx]

        # Normalize if specified
        if config["normalize"]:
            channel_data = (channel_data - channel_data.min()) / (
                channel_data.max() - channel_data.min() + 1e-8
            )

        # Plot with specified colormap
        plt.imshow(channel_data, aspect="auto", origin="lower", cmap=config["cmap"])
        plt.colorbar()
        plt.title(
            f"{config['name'].capitalize()} - Epoch {epoch} - {label} (ID: {class_id})"
        )
        plt.axis("off")
        plt.tight_layout()

        # Save figure to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)

        # Convert buffer to numpy array
        img = PIL.Image.open(buf)
        img_array = np.array(img)

        # Log to wandb with channel-specific name
        wandb_logger.log_image(
            img_array,
            f"{config['name'].capitalize()} - Epoch {epoch} - Label: {label} (ID: {class_id}), File: {filename}, Chunk: {chunk_id}",
            epoch=epoch,
            step=step,
            label=label,
            class_id=class_id,
            filename=filename,
            chunk_id=chunk_id,
            batch_id=batch_id,
            sample_id=sample_id,
            channel=config["name"],
        )
