from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class CFG:
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    machine = "local"  # Options: "databricks", "local", "server", "kaggle"
    use_tta = False  # test-time-augmentation placeholder

    dirs = EasyDict()
    dirs.DATA_ROOT = Path("data/birdclef-2025")

    inference = EasyDict()
    inference.DEBUG = False
    inference.USE_SPECIFIC_FOLDS = False

    training = EasyDict()
    training.DEBUG = True if device == "cpu" else False
    training.EPOCHS = 50
    training.NUM_WORKERS = 10 if device == "cuda" else 0
    training.PREFETCH_FACTOR = 6 if training.NUM_WORKERS > 0 else None
    training.AUGMENTATION_PROB = 0.5
    training.N_FOLD = 5
    training.PROGRESSIVE_UNFREEZING = False
    training.SAMPLING_CLASSES_WEIGHTS = "uniform"
    training.SELECTED_FOLDS = [0, 1, 2, 3, 4]
    training.SAMPLES_PER_EPOCH = 15000
    training.CORRECT_LOSS = True
    training.SAVE_INTERMEDIATE_MODEL = True
    training.EARLY_STOPPING_METRIC = "f1"  # f1 auc
    training.EARLY_STOPPING_MIN_DELTA = 0.005
    training.EARLY_STOPPING_PATIENCE = 100
    training.BATCH_SIZE = 256 if device == "cuda" else 32
    training.GRAD_ACCUM_STEPS = 4
    training.OPTIMIZER = "AdamW"
    training.WEIGHT_DECAY = 1e-5

    # Learning rate configuration
    training.SCHEDULER = "CosineAnnealingLR" # WarmupCosineScheduler CosineAnnealingLR
    training.BASE_LR = 3e-3  # Base learning rate before scaling
    training.MIN_LR = 1e-6
    training.T_MAX = training.EPOCHS
    training.LR_SCALING = True  # Whether to use dynamic LR scaling
    training.LR_SCALE_FACTOR = 0.1  # Factor to scale LR for classifier head
    training.WARMUP_EPOCHS = 5  # Number of epochs for warmup
    training.WARMUP_FACTOR = 0.1  # Starting warmup factor
    training.MAX_GRAD_NORM = 5.0  # Maximum gradient norm for clipping, None to disable
    training.LR_LAYER_DECAY = 0.9  # Layer-wise learning rate decay factor


    training.CRITERION = "BCEWithLogitsLoss"  # AsymmetricLossMultiLabel BCEWithLogitsLoss HierarchicalBCELoss CELoss
    # label smoothing
    training.USE_LABEL_SMOOTHING = True
    training.PRIMARY_LABEL_SMOOTHING = 0.1  # primary label weight = 1 - smoothing
    training.SECONDARY_LABEL_WEIGHT = (
        0.2  # Weight for secondary labels in label smoothing
    )
    # HierarchicalBCELoss
    training.PRIMARY_WEIGHT = (
        1.0 if training.CRITERION == "HierarchicalBCELoss" else None
    )
    training.SECONDARY_WEIGHT = (
        0.5 if training.CRITERION == "HierarchicalBCELoss" else None
    )


    def update_debug_settings(self):
        if self.training.DEBUG:
            self.training.DEBUG_N_CLASSES = 10
            self.training.EPOCHS = 50
            self.training.SAMPLES_PER_EPOCH = 10000
            self.training.SELECTED_FOLDS = [0]
            self.training.T_MAX = self.training.EPOCHS

    model = EasyDict()
    model.model_name = "efficientnet-b0"
    model.kernel_size = (3, 3)
    model.cfar_scaling_factors = (1, 2)
    model.mixup_alpha = 0

    preprocessing = EasyDict()
    preprocessing.NUM_WORKERS = 10
    preprocessing.SAMPLE_RATE = 32000
    preprocessing.PADMODE = "constant"
    preprocessing.N_FFT = 1024
    preprocessing.HOP_LENGTH = 512
    preprocessing.N_MELS = 128
    preprocessing.FMIN = 50
    preprocessing.FMAX = 14000
    preprocessing.SEGMENT_DURATION = 5  # seconds
    preprocessing.NSAMPLES = preprocessing.SEGMENT_DURATION * preprocessing.SAMPLE_RATE
    preprocessing.UFOLD_OVERLAP = preprocessing.NSAMPLES // 2  # 2.5 seconds overlap
    preprocessing.MAKE_RGB = False

    def update_machine_settings(self, machine="local", inference_debug=False):
        self.machine = machine
        if self.machine == "databricks":
            self.dirs.DATA_ROOT = Path(
                "/dbfs/RAW/W00001_Data_Unrestricted/Andrejs/birdclef-2025/"
            )

        elif self.machine == "kaggle":
            self.dirs.DATA_ROOT = Path("/kaggle/input/birdclef-2025")
            self.dirs.MODEL_PATH = Path("/kaggle/input/efficientnet_b0")
            self.dirs.test_soundscapes = (
                self.dirs.DATA_ROOT / "test_soundscapes"
            ).as_posix()
            self.dirs.train_soundscapes = (
                self.dirs.DATA_ROOT / "train_soundscapes"
            ).as_posix()

        elif self.machine in ["server"]:
            self.dirs.DATA_ROOT = Path("/data/birdclef-2025")

        self.dirs.train_datadir = (
            self.dirs.DATA_ROOT / "train_audio_no_voice"
        ).as_posix()
        self.dirs.train_csv = (self.dirs.DATA_ROOT / "train.csv").as_posix()
        self.dirs.taxonomy_csv = (self.dirs.DATA_ROOT / "taxonomy.csv").as_posix()
        self.dirs.cache_dir = (
            self.dirs.DATA_ROOT / "train_audio_no_voice_spectrograms"
        ).as_posix()
        self.dirs.submission_csv = (
            self.dirs.DATA_ROOT / "sample_submission.csv"
        ).as_posix()

        self.inference.DEBUG = inference_debug
        if self.inference.DEBUG:
            self.dirs.test_soundscapes = self.dirs.train_soundscapes
            self.inference.DEBUG_COUNT = 10
        if self.inference.USE_SPECIFIC_FOLDS:
            self.inference.SELECTED_FOLDS = [0, 1, 2, 3, 4]

    def to_dict(self) -> dict:
        """Convert CFG object to a dictionary.

        Returns:
            dict: Dictionary representation of the configuration
        """
        import json
        from pathlib import Path

        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_paths(item) for item in obj]
            return obj

        # Convert to dict and handle Path objects
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        config_dict = convert_paths(config_dict)

        return config_dict


# This file can be copied AS IS into kaggle notebook
# uncomment the code below for kaggle inference. Training will create a new CFG() instance inside train_notebook.py
# cfg = CFG()
# cfg.update_machine_settings(machine="kaggle", inference_debug=True)
# cfg.num_classes = 206
