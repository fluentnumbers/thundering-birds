from datetime import datetime
from src.models.birdclef_model import BirdCLEFModel as MyModel
import src
from easydict import EasyDict
import json
import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm
import torchaudio
import torch.hub
from src.training.metrics import (
    analyze_class_performance,
    calculate_class_metrics,
    plot_class_metrics,
    plot_confusion_matrix,
)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
import torch
import gc
from src.utils.logger import setup_logger, WandbLogger
LOGS_DIR = Path("logs")

logger = setup_logger(__name__)
# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:

    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 2
    DATA_ROOT = Path("/workspace/thundering-birds/data/birdclef-2025")
    OUTPUT_DIR = '/outputs/'

    train_datadir = (
            DATA_ROOT / "train_audio_no_voice"
        ).as_posix()
    train_csv = (DATA_ROOT / "train.csv").as_posix()
    test_soundscapes = (DATA_ROOT / "test_soundscapes").as_posix()
    submission_csv = (DATA_ROOT / "sample_submission.csv").as_posix()
    taxonomy_csv = (DATA_ROOT / "taxonomy.csv").as_posix()

    spectrogram_npy = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'

    model_name = 'efficientnet_b3'
    pretrained = True
    in_channels = 1


    model = EasyDict()
    model.model_name = "efficientnet-b0"
    model.kernel_size = (3, 3)
    model.cfar_scaling_factors = (1, 2)
    model.mixup_alpha = 0.5

    training = EasyDict()
    training.EARLY_STOPPING_METRIC = 'f1'
    training.EARLY_STOPPING_MIN_DELTA = 0.005
    training.EARLY_STOPPING_PATIENCE = 20
    training.DEBUG = False

    LOAD_DATA = False
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (224,224)

    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 96 if device == 'cuda' else 32
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    optimizer = 'AdamW'
    lr = 1e-3
    weight_decay = 1e-6

    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5
    mixup_alpha = 0.5

    # VAD settings
    vad_threshold = 0.5
    vad_sampling_rate = 32000
    vad_min_speech_duration_ms = 250
    vad_min_silence_duration_ms = 100
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 10
            self.selected_folds = [0]
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


cfg = CFG()
set_seed(cfg.seed)


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm

def remove_human_voice(audio_data, sr, cfg):
    """人の声の部分を検出して除去"""
    try:
        # Silero VADモデルのロード
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )

        (get_speech_timestamps, _, read_audio, *_) = utils

        # 音声データをtensorに変換
        wav_tensor = torch.FloatTensor(audio_data)

        # 人の声のタイムスタンプを取得
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            model,
            threshold=0.5,
            sampling_rate=sr
        )

        # 人の声の部分をマスク
        mask = torch.ones_like(wav_tensor)
        for ts in speech_timestamps:
            mask[ts['start']:ts['end']] = 0

        # マスクを適用
        clean_audio = wav_tensor * mask

        return clean_audio.numpy()
    except Exception as e:
        logger.error(f"VAD error: {e}")
        return audio_data

def process_audio_file(audio_path, cfg):
    """Process a single audio file to get the mel spectrogram"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

        # 人の声の除去を追加
        # audio_data = remove_human_voice(audio_data, cfg.FS, cfg)

        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            # 短すぎる場合は繰り返し
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # ランダムな5秒セグメントを選択
        if len(audio_data) > target_samples:
            max_start_idx = len(audio_data) - target_samples
            start_idx = random.randint(0, max_start_idx)
            audio_data = audio_data[start_idx:start_idx + target_samples]

        # 長さが足りない場合はパディング
        if len(audio_data) < target_samples:
            audio_data = np.pad(
                audio_data,
                (0, target_samples - len(audio_data)),
                mode='constant'
            )

        mel_spec = audio2melspec(audio_data, cfg)

        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(
                mel_spec,
                cfg.TARGET_SHAPE,
                interpolation=cv2.INTER_LINEAR
            )

        return mel_spec.astype(np.float32)

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg):
    """Generate spectrograms from audio files"""
    logger.info("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if cfg.debug and i >= 1000:
            break

        try:
            samplename = row['samplename']
            filepath = row['filepath']

            mel_spec = process_audio_file(filepath, cfg)

            if mel_spec is not None:
                all_bird_data[samplename] = mel_spec

        except Exception as e:
            logger.error(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    logger.info(f"Failed to process {len(errors)} files")

    return all_bird_data


class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode

        self.spectrograms = spectrograms

        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename

        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            logger.info(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":  # Only print warning during training
                logger.warning(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        target = self.encode_label(row['primary_label'])

        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']

            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""

        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                width = random.randint(8, 16)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0

        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                height = random.randint(8, 16)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0

        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        # Randomly shift the spectrogram along the time axis
        if random.random() < 0.7:
            shift = random.randint(-3,3)
            spec = torch.roll(spec, shifts=shift, dims=1)
            num_masks = random.randint(2,4)
        return spec

    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target


def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}

    result = {key: [] for key in batch[0].keys()}

    for item in batch:
        for key, value in item.items():
            result[key].append(value)

    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])

    return result


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.4,
            drop_path_rate=0.4
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.feat_dim = backbone_out

        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):

        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits,
                                       logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_optimizer(model, cfg):

    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")

    return optimizer

def get_scheduler(optimizer, cfg):

    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None
    else:
        scheduler = None

    return scheduler

def get_criterion(cfg):

    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")

    return criterion


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, cfg=None, species_ids=None):

    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training", unit="batch")

    for step, batch in pbar:

        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []

            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()

                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())

            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()

        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())

        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    # Use enhanced metrics calculation
    metrics = calculate_class_metrics(
        all_targets, all_outputs, labels=species_ids, thresholds=None
    )

    # Calculate class distribution for analysis
    class_distribution = {
        label: np.sum(all_targets[:, i]) for i, label in enumerate(species_ids)
    }
    analysis = analyze_class_performance(metrics, species_ids, class_distribution)

    avg_loss = np.mean(losses)

    return avg_loss, metrics, analysis

def validate(model, loader, criterion, device, species_ids=None):

    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    pbar = tqdm(enumerate(loader), desc="Validation", unit="batch", total=len(loader))

    with torch.no_grad():
        for step, batch in pbar:
            if isinstance(batch['melspec'], list):
                batch_outputs = []
                batch_losses = []

                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)

                    output = model(inputs)
                    loss = criterion(output, target)

                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())

                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch['target'].numpy()

            else:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())
        pbar.set_postfix(
                {
                    "val_loss": f"{np.mean(losses[-10:]):.2f}",
                }
            )

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    metrics = calculate_class_metrics(
        all_targets, all_outputs, labels=species_ids, thresholds=None
    )

    # Calculate class distribution for analysis
    class_distribution = {
        label: np.sum(all_targets[:, i]) for i, label in enumerate(species_ids)
    }
    analysis = analyze_class_performance(metrics, species_ids, class_distribution)

    avg_loss = np.mean(losses)

    return avg_loss, metrics, analysis

def calculate_auc(targets, outputs):

    num_classes = targets.shape[1]
    aucs = []

    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):

        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)

    return np.mean(aucs) if aucs else 0.0


def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"training_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting training run in {run_dir}")

    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)

    if cfg.debug:
        cfg.update_debug_settings()

    spectrograms = None
    if cfg.LOAD_DATA:
        logger.info("Loading pre-computed mel spectrograms from NPY file...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            logger.info(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")
        except Exception as e:
            logger.error(f"Error loading pre-computed spectrograms: {e}")
            logger.info("Will generate spectrograms on-the-fly instead.")
            cfg.LOAD_DATA = False

    if not cfg.LOAD_DATA:
        logger.info("Will generate spectrograms on-the-fly during training.")
        if 'filepath' not in df.columns:
            df['filepath'] = cfg.train_datadir + '/' + df.filename
        if 'samplename' not in df.columns:
            df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    best_scores = []
    torch.backends.cudnn.benchmark = True

    wandb_group = f"train_{'DEBUG' if cfg.training.DEBUG else 'PROD'}_{cfg.num_classes}classes_{cfg.device.upper()}_{timestamp}"

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue

        logger.info(f'\n{"="*30} Fold {fold} {"="*30}')

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        logger.info(f'Training set: {len(train_df)} samples')
        logger.info(f'Validation set: {len(val_df)} samples')

        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=spectrograms, mode='valid')


        wandb_logger = WandbLogger(
            f"fold{fold}_{cfg.num_classes}classes",
            run_dir,
            group=wandb_group,
            tags=[
                f"fold_{fold}",
                f"n_classes_{cfg.num_classes}",
                f"{'DEBUG' if cfg.training.DEBUG else 'PROD'}",
                f"{cfg.device.upper()}",
            ],
            # config={
            #     "batch_size": cfg.training.BATCH_SIZE,
            #     "epochs": cfg.training.EPOCHS,
            #     "model": cfg.model.model_name,
            #     "device": cfg.device,
            #     "seed": cfg.seed,
            #     "n_classes": cfg.num_classes,
            #     "optimizer": cfg.training.OPTIMIZER,
            #     "scheduler": cfg.training.SCHEDULER,
            #     "criterion": cfg.training.CRITERION,
            #     "early_stopping_metric": cfg.training.EARLY_STOPPING_METRIC,
            #     "early_stopping_patience": cfg.training.EARLY_STOPPING_PATIENCE,
            #     "progressive_unfreezing": cfg.training.PROGRESSIVE_UNFREEZING,
            #     "loss_weighting": cfg.training.LOSS_WEIGHTING,
            #     "loss_momentum": cfg.training.LOSS_MOMENTUM,
            #     "loss_temperature": cfg.training.LOSS_TEMPERATURE,
            #     "loss_min_weight": cfg.training.LOSS_MIN_WEIGHT,
            #     "sampling_classes_weights": cfg.training.SAMPLING_CLASSES_WEIGHTS,
            #     "samples_per_epoch": cfg.training.SAMPLES_PER_EPOCH,
            #     "lr_scaling": cfg.training.LR_SCALING,
            #     "lr_scale_factor": cfg.training.LR_SCALE_FACTOR,
            #     "warmup_epochs": cfg.training.WARMUP_EPOCHS,
            #     "warmup_factor": cfg.training.WARMUP_FACTOR,
            #     "max_grad_norm": cfg.training.MAX_GRAD_NORM,
            #     "lr_layer_decay": cfg.training.LR_LAYER_DECAY,
            #     "min_lr": cfg.training.MIN_LR,
            #     "base_lr": cfg.training.BASE_LR,
            #     "t_max": cfg.training.T_MAX,
            #     "grad_accum_steps": cfg.training.GRAD_ACCUM_STEPS,
            #     "train_files": len(train_dataset.metadata_df["filename"].unique()),
            #     "train_segments": len(train_dataset.metadata_df),
            #     "train_classes": len(train_dataset.class_ids),
            #     "val_files": len(val_dataset.metadata_df["filename"].unique()),
            #     "val_segments": len(val_dataset.metadata_df),
            #     "val_classes": len(val_dataset.class_ids),
            # },
        )
        # After initializing wandb_logger
        # config_dict = cfg.to_dict()
        # wandb_logger.store_config_artifact(config_dict, artifact_name="training_config")


        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # model = BirdCLEFModel(cfg).to(cfg.device)
        model = MyModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)

        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)

        best_epoch = 0
        best_model_state = None
        best_optimizer_state = None
        best_scheduler_state = None

        # Early stopping variables
        no_improvement_epochs = 0
        best_metric = 0

        for epoch in range(cfg.epochs):
            logger.info(f"\nEpoch {epoch+1}/{cfg.epochs}")

            train_loss, train_metrics, train_analysis = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None,
                species_ids=species_ids
            )

            val_loss, val_metrics, val_analysis= validate(model, val_loader, criterion, cfg.device, species_ids)

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            logger.info(f"Train Loss: {train_loss:.4f}, Train AUC: {train_metrics['macro_metrics']['auc']:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['macro_metrics']['auc']:.4f}")

            train_metrics_plots = plot_class_metrics(train_metrics, species_ids)
            val_metrics_plots = plot_class_metrics(val_metrics, species_ids)
            train_cm_plot = plot_confusion_matrix(
                train_metrics["confusion_matrix"], species_ids
            )
            val_cm_plot = plot_confusion_matrix(
                val_metrics["confusion_matrix"], species_ids
            )

 # Log all metrics to wandb with fold grouping
            wandb_logger.log(
                {
                    "epoch": epoch + 1,
                    "fold": fold,
                    "TOP/val_loss": val_loss,
                    "TOP/train_loss": train_loss,
                    "TOP/val_f1": val_metrics["macro_metrics"]["f1"],
                    "TOP/val_confusion_matrix": val_cm_plot,
                    # Loss metrics
                    "train/loss": train_loss,
                    # "train/weighted_loss": train_metrics["train_weighted_loss"],
                    "val/loss": val_loss,
                    # Macro metrics
                    "train/auc": train_metrics["macro_metrics"]["auc"],
                    "train/f1": train_metrics["macro_metrics"]["f1"],
                    "train/precision": train_metrics["macro_metrics"]["precision"],
                    "train/recall": train_metrics["macro_metrics"]["recall"],
                    "val/auc": val_metrics["macro_metrics"]["auc"],
                    "val/f1": val_metrics["macro_metrics"]["f1"],
                    "val/precision": val_metrics["macro_metrics"]["precision"],
                    "val/recall": val_metrics["macro_metrics"]["recall"],
                    # Top-k accuracy
                    "train/top_k_accuracy": train_metrics["top_k_accuracy"],
                    "val/top_k_accuracy": val_metrics["top_k_accuracy"],
                    # Learning rate
                    "learning_rate": (
                        scheduler.get_last_lr()[0] if scheduler else cfg.lr
                    ),
                    "train/hard_classes": train_analysis["hard_classes"],
                    "train/easy_classes": train_analysis["easy_classes"],
                    "val/hard_classes": val_analysis["hard_classes"],
                    "val/easy_classes": val_analysis["easy_classes"],
                    # Visualizations
                    "train/precision_plot": train_metrics_plots[0],
                    "train/recall_plot": train_metrics_plots[1],
                    "train/f1_plot": train_metrics_plots[2],
                    "train/auc_plot": train_metrics_plots[3],
                    "val/precision_plot": val_metrics_plots[0],
                    "val/recall_plot": val_metrics_plots[1],
                    "val/f1_plot": val_metrics_plots[2],
                    "val/auc_plot": val_metrics_plots[3],
                    "train/confusion_matrix": train_cm_plot,
                    "val/confusion_matrix": val_cm_plot,
                },
            )


            if cfg.training.EARLY_STOPPING_METRIC == "loss":
                current_metric = -val_loss  # Negative because we want to minimize loss
            else:
                current_metric = val_metrics["macro_metrics"][
                    cfg.training.EARLY_STOPPING_METRIC
                ]
            logger.info(
                f"Current {cfg.training.EARLY_STOPPING_METRIC}: {current_metric:.3f}, best: {best_metric:.3f}, delta: {current_metric - best_metric:.3f}"
            )

            if current_metric > best_metric + cfg.training.EARLY_STOPPING_MIN_DELTA:
                best_metric = current_metric
                no_improvement_epochs = 0
                best_epoch = epoch + 1
                logger.info(
                    f"New best {cfg.training.EARLY_STOPPING_METRIC}: {best_metric:.3f} at epoch {best_epoch} \n"
                )
                # Save model checkpoint when metrics improve
                checkpoint_path = run_dir / f"model_fold{fold}_epoch{epoch+1}_best.pth"

                # Delete previous checkpoints for this fold
                for old_checkpoint in run_dir.glob(f"model_fold{fold}_*_best.pth"):
                    if old_checkpoint != checkpoint_path:
                        old_checkpoint.unlink()
                        logger.debug(f"Deleted old checkpoint {old_checkpoint}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    "val_auc": val_metrics["macro_metrics"]["auc"],
                        "val_f1": val_metrics["macro_metrics"]["f1"],
                        "train_auc": train_metrics["macro_metrics"]["auc"],
                        "train_f1": train_metrics["macro_metrics"]["f1"],
                    'cfg': cfg
                }, f"model_fold{fold}.pth")
                logger.debug(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                no_improvement_epochs += 1
                logger.info(
                    f"No improvement in {cfg.training.EARLY_STOPPING_METRIC} for {no_improvement_epochs} epochs (best value: {best_metric:.3f})"
                )

            # Check for early stopping
            if no_improvement_epochs >= cfg.training.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")

                # Load best model state if it exists
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    if best_optimizer_state is not None:
                        optimizer.load_state_dict(best_optimizer_state)
                    if scheduler and best_scheduler_state is not None:
                        scheduler.load_state_dict(best_scheduler_state)
                else:
                    logger.warning("No best model state found to load")

                break
            # Check for early stopping

        best_scores.append(
            {
                "auc": val_metrics["macro_metrics"]["auc"],
                "f1": val_metrics["macro_metrics"]["f1"],
                "epoch": best_epoch,
            }
        )
        logger.info(
            f"Best metrics for fold {fold}: AUC: {val_metrics['macro_metrics']['auc']:.4f}, F1: {val_metrics['macro_metrics']['f1']:.4f} at epoch {best_epoch}"
        )

        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("\n" + "="*60)
    logger.info("Cross-Validation Results:")
    for fold, scores in enumerate(best_scores):
        logger.info(f"Fold {fold}: AUC: {scores['auc']:.4f}, F1: {scores['f1']:.4f}")
    logger.info(f"Mean AUC: {np.mean([s['auc'] for s in best_scores]):.4f}")
    logger.info(f"Mean F1: {np.mean([s['f1'] for s in best_scores]):.4f}")
    logger.info("="*60)
    # Save final results
    results = {
        "best_scores": best_scores,
        "mean_auc": float(np.mean([s["auc"] for s in best_scores])),
        "mean_f1": float(np.mean([s["f1"] for s in best_scores])),
        "std_auc": float(np.std([s["auc"] for s in best_scores])),
        "std_f1": float(np.std([s["f1"] for s in best_scores])),
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved results to {run_dir / 'results.json'}")

if __name__ == "__main__":
    import time
    load_dotenv(".env")

    logger.info("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

    logger.info("\nStarting training...")
    logger.info(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        logger.info("Using pre-computed mel spectrograms from NPY file")
    else:
        logger.info("Will generate spectrograms on-the-fly during training")

    run_training(train_df, cfg)

    print("\nTraining complete!")
