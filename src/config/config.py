from pathlib import Path

import torch


class Config:
    """Configuration class for the pipeline."""

    def __init__(self):
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEV_MODE = True  # Take only fraction of the dataset, for dev
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
        self.NUM_WORKERS = 4
        self.LR_MAX = 3e-4
        self.EPOCHS = 5

        # Model config
        self.N_CLASSES = None  # Will be set after data loading
