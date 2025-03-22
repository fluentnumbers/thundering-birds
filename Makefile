.PHONY: download_kaggle_dataset clean_data setup install

# Default target
all: setup install download_kaggle_dataset

# Setup Poetry environment
setup:
	poetry env use python3.12
	poetry install

# Install dependencies
install:
	poetry install

# Download and extract Kaggle dataset
download_kaggle_dataset:
	poetry run python download_kaggle_dataset.py

# Clean the data directory
clean_data:
	rm -rf data/*
	rm -f birdclef-2025.zip

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup              - Set up Poetry environment with Python 3.12"
	@echo "  make install           - Install project dependencies"
	@echo "  make download_kaggle_dataset - Download the BirdCLEF 2025 dataset"
	@echo "  make clean_data        - Remove all downloaded and extracted data"
	@echo "  make all              - Run setup, install, and download dataset"
	@echo "  make help             - Show this help message"
