"""Shared preprocessing utilities for mood model training and inference.

All feature engineering steps that must be identical between training
(`test_model.py` / `mood_model.py`) and runtime inference (`app.py`) live here.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

# Ordered categorical labels used in binning so training & inference align
DAY_PART_BINS = [-1, 5, 9, 16, 23]
DAY_PART_LABELS = ['Night', 'Morning', 'Afternoon', 'Evening']

LIGHT_BINS = [-1, 100, 500, 1000, float('inf')]
LIGHT_LABELS = ['Dark', 'Low', 'Medium', 'Bright']

FEATURE_COLUMNS = [
    'Temperature', 'Humidity', 'MQ-2', 'BH1750FVI', 'Radar', 'Ultrasonic',
    'song', 'day_part', 'light_category', 'temp_humidity_ratio', 'hour_sin', 'hour_cos'
]


def engineer_features(df: pd.DataFrame, drop_original_timestamp: bool = False) -> pd.DataFrame:
    """Apply feature engineering in a consistent, idempotent way.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least 'timestamp' and sensor columns.
    drop_original_timestamp : bool
        If True, remove raw timestamp & hour columns (for training). Leave them for inference if False.
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
    elif 'hour' not in df.columns:
        raise ValueError("Input must contain 'timestamp' or precomputed 'hour'.")

    # Day part (same cut points as previously chosen)
    df['day_part'] = pd.cut(
        df['hour'], bins=DAY_PART_BINS, labels=DAY_PART_LABELS
    )

    # Ratio feature
    if 'Temperature' in df and 'Humidity' in df:
        df['temp_humidity_ratio'] = df['Temperature'] / (df['Humidity'] + 1e-6)
    else:
        df['temp_humidity_ratio'] = np.nan

    # Light category
    if 'BH1750FVI' in df:
        df['light_category'] = pd.cut(
            df['BH1750FVI'], bins=LIGHT_BINS, labels=LIGHT_LABELS
        )
    else:
        df['light_category'] = pd.Categorical([np.nan] * len(df), categories=LIGHT_LABELS)

    # Cyclical hour encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # MQ-2 mapping: allow both already-numeric & categorical
    if 'MQ-2' in df.columns:
        # Only map if not numeric
        if not is_numeric_dtype(df['MQ-2']):
            df['MQ-2'] = df['MQ-2'].map({'OK': 0, 'N_OK': 1})

    if drop_original_timestamp:
        for col in ['timestamp', 'hour']:
            if col in df:
                df.drop(columns=col, inplace=True)
    return df


def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe restricted to FEATURE_COLUMNS in correct order (missing -> NaN)."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURE_COLUMNS]


__all__ = [
    'engineer_features',
    'ensure_feature_order',
    'FEATURE_COLUMNS'
]
