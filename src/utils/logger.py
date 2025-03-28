import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import LOGS_DIR


def setup_logger(run_dir: Path) -> logging.Logger:
    """Setup logging with both file and console handlers."""
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

    return logger


class WandbLogger:
    """Wrapper for wandb logging functionality."""

    def __init__(self, run_name: str, run_dir: Path):
        self.enabled = False
        try:
            import wandb

            self.wandb = wandb
            self.run_dir = LOGS_DIR / run_dir
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
