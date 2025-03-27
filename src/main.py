import json
from datetime import datetime
from pathlib import Path

from src.config.config import Config
from src.training.trainer import train
from src.utils.logger import setup_logger


def main():
    """Main entry point for the training pipeline."""
    # Initialize config
    config = Config()

    # Setup logging
    run_dir = (
        Path("logs") / f"basic_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logger = setup_logger(run_dir)
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

    # Run training
    train(config, run_dir)


if __name__ == "__main__":
    main()
