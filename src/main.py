import json

# Set multiprocessing start method to 'spawn' for cluster environments
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

mp.set_start_method("spawn", force=True)

from src.config import LOGS_DIR, Config
from src.training.trainer import train
from src.utils.logger import setup_logger


def main():
    """Main entry point for the training pipeline."""
    # Initialize config
    config = Config()

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"basic_pipeline_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger(__name__, run_dir)

    # Log device information
    device_info = (
        f"Using device: {config.DEVICE}"
        f"{' with ' + str(torch.cuda.device_count()) + ' GPUs' if torch.cuda.is_available() else ''}"
    )
    logger.info(device_info)

    # Save config to run directory
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    # Convert Path objects to strings
    config_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items()
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    logger.info(f"Saved config to {run_dir / 'config.json'}")

    # Run training
    train(config, run_dir)


if __name__ == "__main__":
    load_dotenv(".env")
    main()
