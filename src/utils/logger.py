import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import LOGS_DIR


def setup_logger(name: str, run_dir: Path = None) -> logging.Logger:
    """Setup logging with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Only add handlers if they haven't been added already
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create file handler if run_dir is provided
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            log_filepath = run_dir / "training.log"
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setLevel(logging.INFO)
            logger.info(f"Logging to {log_filepath}")

        # Define format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        if run_dir is not None:
            file_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(console_handler)
        if run_dir is not None:
            logger.addHandler(file_handler)

        # Silence some logs
        logging.getLogger("numba.core").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

    return logger


class WandbLogger:
    """Wrapper for wandb logging functionality."""

    def __init__(self, run_name: str, run_dir: Path):
        self.enabled = False
        try:
            import wandb

            self.wandb = wandb
            self.run_dir = run_dir  # Use the provided run_dir directly
            self.wandb.init(
                project="bird-sound-classification",
                name=run_name,
                dir=str(self.run_dir),
            )
            self.enabled = True
            logging.info("Initialized wandb logging")
        except ImportError:
            logging.info("wandb not available, continuing without wandb logging")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")

    def log(self, data: dict) -> None:
        """Log data to wandb if available."""
        if self.enabled:
            try:
                self.wandb.log(data)
            except Exception as e:
                logging.warning(f"Failed to log to wandb: {e}")

    def log_image(self, image: np.ndarray, caption: str, **kwargs) -> None:
        """Log image to wandb if available."""
        if self.enabled:
            try:
                self.wandb.log(
                    {"images": self.wandb.Image(image, caption=caption), **kwargs}
                )
            except Exception as e:
                logging.warning(f"Failed to log image to wandb: {e}")

    def finish(self) -> None:
        """Finish wandb run if available."""
        if self.enabled:
            try:
                self.wandb.finish()
            except Exception as e:
                logging.warning(f"Failed to finish wandb: {e}")
