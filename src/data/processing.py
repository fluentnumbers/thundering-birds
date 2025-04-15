import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def normalize_values_by_group(values, groups, min_value=0.1, max_value=1.0):
    """
    Normalize values within each group using vectorized operations.

    Args:
        values (np.ndarray or list): Values to normalize
        groups (np.ndarray or list): Group identifiers for each value
        min_value (float): Minimum value after normalization
        max_value (float): Maximum value after normalization

    Returns:
        np.ndarray: Array of normalized values, same length as input values
    """
    # Convert inputs to numpy arrays if they aren't already
    values = np.asarray(values)
    groups = np.asarray(groups)

    # Create a DataFrame for vectorized group operations
    df = pd.DataFrame({"group": groups, "value": values})

    # Group by and transform to compute min/max per group
    df["group_min"] = df.groupby("group")["value"].transform(
        lambda x: np.nanmin(x[x > 0]) if np.any(x > 0) else 0
    )
    df["group_max"] = df.groupby("group")["value"].transform(
        lambda x: np.nanmax(x) if not np.all(np.isnan(x)) else max_value
    )

    # Compute range and handle zero range case
    df["range"] = df["group_max"] - df["group_min"]
    zero_range_mask = df["range"] <= 0

    # Vectorized normalization
    normalized_values = np.where(
        zero_range_mask,
        min_value,  # minimum value for invalid/same value
        np.clip((df["value"] - df["group_min"]) / df["range"], min_value, max_value),
    )

    return normalized_values


def align_df_and_metadata(df, df_cache):
    # Get the intersection of filenames present in both dataframes
    common_filenames = set(df["filename"]).intersection(set(df_cache["filename"]))

    # Filter both dataframes to only include common filenames
    df = df[df["filename"].isin(common_filenames)]
    df_cache = df_cache[df_cache["filename"].isin(common_filenames)]

    df = df.reset_index(drop=True)
    df_cache = df_cache.reset_index(drop=True)
    return df, df_cache
