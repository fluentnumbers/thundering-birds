import os
import subprocess
import sys
import zipfile
from pathlib import Path

from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


def download_dataset():
    """Download the BirdCLEF 2025 dataset using kaggle CLI and extract it to data/birdclef-2025/ directory."""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path("data") / "birdclef-2025"
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified data directory at {data_dir}")

        # Download the dataset
        logger.info("Starting dataset download...")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "birdclef-2025"], check=True
        )

        # Extract the zip file to data directory
        zip_file = Path("birdclef-2025.zip")
        if zip_file.exists():
            logger.info("Extracting dataset files...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            # Remove the zip file after extraction
            zip_file.unlink()
            logger.info(
                "Dataset downloaded and extracted successfully to data/birdclef-2025/ directory!"
            )
        else:
            logger.error("Downloaded zip file not found")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
