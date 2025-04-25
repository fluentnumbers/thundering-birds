import logging
from datetime import datetime
from pathlib import Path

import numpy as np


def setup_logger(name: str, run_dir: Path = None) -> logging.Logger:
    """Setup logging with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent message propagation to parent loggers

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
        log_format = "%(levelname)s - %(message)s"
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
        logging.getLogger("torch.hub").setLevel(
            logging.WARNING
        )  # Silence torch.hub messages
        logging.getLogger("torch").setLevel(
            logging.WARNING
        )  # Silence general torch messages
        logging.getLogger("torchaudio").setLevel(
            logging.WARNING
        )  # Silence torchaudio messages

    return logger


class WandbLogger:
    """Wrapper for wandb logging functionality."""

    def __init__(
        self,
        run_name: str,
        run_dir: Path,
        group: str = None,
        tags: list = None,
        config: dict = None,
    ):
        self.enabled = False
        try:
            import wandb

            self.wandb = wandb

            # Validate and clean tags
            if tags is not None:
                # Remove empty strings and ensure tags are between 1-64 chars
                tags = [tag for tag in tags if tag and 1 <= len(str(tag)) <= 64]
                if not tags:  # If all tags were invalid, set to None
                    tags = None

            # Validate group name
            if group is not None and (not group or len(str(group)) > 64):
                group = None

            # Validate run name
            if not run_name or len(str(run_name)) > 64:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.run_dir = run_dir
            self.run = self.wandb.init(
                project="bird-sound-classification",
                name=run_name,
                dir=str(self.run_dir),
                group=group,
                tags=tags,
                config=config,
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
                    {caption: self.wandb.Image(image, caption=caption, **kwargs)}
                )
            except Exception as e:
                logging.warning(f"Failed to log image to wandb: {e}")

    def finish(self) -> None:
        """Finish the wandb run if available."""
        if self.enabled:
            try:
                self.wandb.finish()
            except Exception as e:
                logging.warning(f"Failed to finish wandb run: {e}")

    def store_config_artifact(self, config: dict, artifact_name: str = "config") -> None:
        """Store configuration as a wandb artifact.

        Args:
            config: Configuration dictionary to store
            artifact_name: Name for the artifact (default: "config")
        """
        if self.enabled:
            try:
                # Create a new artifact
                artifact = self.wandb.Artifact(
                    name=artifact_name,
                    type="config",
                    description="Training configuration"
                )

                # Add the config as a JSON file
                import json
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(config, f, indent=4)
                    artifact.add_file(f.name, name="config.json")

                # Log the artifact
                self.run.log_artifact(artifact)

                # Clean up temporary file
                import os
                os.unlink(f.name)

                logging.info(f"Stored configuration in wandb artifact: {artifact_name}")
            except Exception as e:
                logging.warning(f"Failed to store configuration in wandb artifact: {e}")
