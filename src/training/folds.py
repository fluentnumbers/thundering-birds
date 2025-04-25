import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from src.utils.logger import setup_logger
logger = setup_logger(__name__)
def get_folds(df, cfg):
    """Create folds using StratifiedGroupKFold ensuring each class has representation in both splits.

    Strategy:
    1. Use StratifiedGroupKFold to create folds
    2. Check if any class is missing from validation set
    3. If found, reallocate one file from training to validation
    4. Log detailed class distributions for each fold
    """
    # Groups are audio files (recordings)
    groups = df["filename"]
    labels = df["primary_label"]

    # Check class distributions
    class_file_counts = df.groupby("primary_label")["filename"].nunique()
    logger.info("\nClass file distribution before splitting:")
    for cls, count in class_file_counts.items():
        logger.info(f"Class {cls}: {count} files")

    # Verify all classes have at least 2 files
    single_file_classes = class_file_counts[class_file_counts == 1].index.tolist()
    if single_file_classes:
        logger.warning(f"Found {len(single_file_classes)} classes with only one file:")
        for cls in single_file_classes:
            logger.warning(
                f"Class '{cls}' has only one file: {df[df['primary_label'] == cls]['filename'].iloc[0]}"
            )
        raise ValueError(
            "Cannot create valid train/validation split with single-file classes. Please ensure all classes have at least 2 files."
        )

    # Create folds using StratifiedGroupKFold
    folds = []
    group_kfold = StratifiedGroupKFold(
        n_splits=cfg.training.N_FOLD, shuffle=True, random_state=cfg.seed
    )

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(df, labels, groups)):
        # Get class distributions
        train_class_files = (
            df.iloc[train_idx].groupby("primary_label")["filename"].nunique()
        )
        val_class_files = (
            df.iloc[val_idx].groupby("primary_label")["filename"].nunique()
        )

        # Check for classes missing in validation
        missing_in_val = set(train_class_files.index) - set(val_class_files.index)
        if missing_in_val:
            logger.info(
                f"\nFold {fold}: Found classes missing in validation: {missing_in_val}"
            )
            for cls in missing_in_val:
                # Get all files for this class in training
                cls_train_files = df.iloc[train_idx][
                    df.iloc[train_idx]["primary_label"] == cls
                ]["filename"].unique()
                # Select one file to move to validation
                file_to_move = cls_train_files[0]  # Take the first file
                # Get indices of this file
                file_indices = df[df["filename"] == file_to_move].index
                # Move from train to validation
                train_idx = np.setdiff1d(train_idx, file_indices)
                val_idx = np.concatenate([val_idx, file_indices])
                logger.info(
                    f"Moved file {file_to_move} from training to validation for class {cls}"
                )

        # Get updated class distributions after reallocation
        train_class_files = (
            df.iloc[train_idx].groupby("primary_label")["filename"].nunique()
        )
        val_class_files = (
            df.iloc[val_idx].groupby("primary_label")["filename"].nunique()
        )

        # Log distribution
        logger.info(f"\nDetailed class distribution for fold {fold}:")
        logger.info("Class | Train Files | Val Files | Total Files")
        logger.info("-" * 50)
        for cls in df["primary_label"].unique():
            train_count = train_class_files.get(cls, 0)
            val_count = val_class_files.get(cls, 0)
            total_count = class_file_counts[cls]
            logger.info(
                f"{cls} | {train_count:^10d} | {val_count:^9d} | {total_count:^11d}"
            )

        # Verify no data leakage
        train_files = set(df.iloc[train_idx]["filename"])
        val_files = set(df.iloc[val_idx]["filename"])
        assert (
            len(train_files & val_files) == 0
        ), f"Data leakage detected in fold {fold}"

        logger.info(f"\nFold {fold} Summary:")
        logger.info(f"Train: {len(train_files)} files")
        logger.info(f"Val: {len(val_files)} files")

        folds.append((train_idx, val_idx))

    return folds
