.PHONY: download_kaggle_dataset clean_data

# Download and extract Kaggle dataset
download_kaggle_dataset:
	python download_kaggle_dataset.py

# Clean the data directory
clean_data:
	rm -rf data/*
	rm -f birdclef-2025.zip

# Show help
help:
	@echo "Available commands:"
	@echo "  make download_kaggle_dataset - Download the BirdCLEF 2025 dataset"
	@echo "  make clean_data            - Remove all downloaded and extracted data"
	@echo "  make help                 - Show this help message"
