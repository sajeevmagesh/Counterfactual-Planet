"""Feature engineering utilities for multimodal training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CLEAN_TRAIN_PATH = Path("data_processed/cleaned_train.parquet")
CLEAN_VAL_PATH = Path("data_processed/cleaned_val.parquet")
FEATURE_OUTPUT_DIR = Path("data_processed/feature_tensors")
METADATA_PATH = FEATURE_OUTPUT_DIR / "metadata.json"

TIME_SERIES_COLUMNS = [
    "co2",
    "co2_per_capita",
    "population",
    "gdp",
    "primary_energy_consumption",
]
SEQ_LENGTH = 5  # number of years in sequence (current + 4 lags)


def load_cleaned_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned train and validation data."""

    if not CLEAN_TRAIN_PATH.exists() or not CLEAN_VAL_PATH.exists():
        raise FileNotFoundError("Cleaned train/val parquet files not found. Run data_cleaning.py first.")
    train_df = pd.read_parquet(CLEAN_TRAIN_PATH)
    val_df = pd.read_parquet(CLEAN_VAL_PATH)
    logger.info("Loaded cleaned train (%d rows) and val (%d rows)", len(train_df), len(val_df))
    return train_df, val_df


def add_growth_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Add year-over-year growth (difference) for selected columns."""

    df = df.sort_values(["iso_code", "year"])
    for col in columns:
        growth_col = f"{col}_growth"
        df[growth_col] = df.groupby("iso_code")[col].diff().fillna(0.0)
    return df


def build_time_series_tensor(df: pd.DataFrame, columns: List[str], seq_len: int) -> np.ndarray:
    """Build tensor [N, seq_len, len(columns)] with historical values."""

    feature_count = len(columns)
    tensor = np.zeros((len(df), seq_len, feature_count), dtype=np.float32)
    idx = 0
    for _, group in df.groupby("iso_code", sort=False):
        group = group.sort_values("year").reset_index()
        values = group[columns].to_numpy(np.float32)
        for i in range(len(group)):
            window = []
            for offset in range(seq_len):
                pos = i - (seq_len - 1 - offset)
                if pos < 0:
                    window.append(np.zeros(feature_count, dtype=np.float32))
                else:
                    window.append(values[pos])
            tensor[idx] = np.stack(window, axis=0)
            idx += 1
    return tensor


def separate_modalities(df: pd.DataFrame, ts_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Split dataframe into numeric features, policy embeddings, and time-series tensors."""

    policy_cols = [col for col in df.columns if col.startswith("policy_embed_")]
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {
        "co2_next",
        "co2_direction",
        "delta_co2",
    }
    static_numeric_cols = [
        col
        for col in numeric_candidates
        if col not in policy_cols and col not in drop_cols and col not in ts_cols
    ]
    growth_cols = [col for col in df.columns if col.endswith("_growth")]
    static_numeric_cols += growth_cols
    static_numeric = df[static_numeric_cols].astype(np.float32).copy()
    policy_embed = df[policy_cols].astype(np.float32).copy()

    # Ensure there are no missing values before tensor export
    static_numeric = static_numeric.fillna(0.0)
    policy_embed = policy_embed.fillna(0.0)
    ts_tensor = build_time_series_tensor(df, ts_cols, SEQ_LENGTH)
    logger.info(
        "Built numeric (%d cols), policy (%d cols), time-series tensor %s",
        static_numeric.shape[1],
        policy_embed.shape[1],
        ts_tensor.shape,
    )
    return static_numeric, ts_tensor, policy_embed


def log_cls_stats(name: str, cls_array: np.ndarray) -> None:
    uniques = np.unique(cls_array)
    print(
        f"{name} cls -> shape={cls_array.shape} unique={uniques} min={cls_array.min()} max={cls_array.max()}"
    )


def build_feature_tensors() -> None:
    """Main entry point for feature tensor creation."""

    train_df, val_df = load_cleaned_frames()
    train_df = add_growth_features(train_df, TIME_SERIES_COLUMNS)
    val_df = add_growth_features(val_df, TIME_SERIES_COLUMNS)

    train_numeric, train_ts, train_policy = separate_modalities(train_df, TIME_SERIES_COLUMNS)
    val_numeric, val_ts, val_policy = separate_modalities(val_df, TIME_SERIES_COLUMNS)

    train_cls = train_df["co2_direction"].astype(np.int64).to_numpy().reshape(-1)
    val_cls = val_df["co2_direction"].astype(np.int64).to_numpy().reshape(-1)
    log_cls_stats("train", train_cls)
    log_cls_stats("val", val_cls)
    targets_train = {
        "cls": train_cls,
        "reg": train_df["delta_co2"].astype(np.float32).to_numpy().reshape(-1),
    }
    targets_val = {
        "cls": val_cls,
        "reg": val_df["delta_co2"].astype(np.float32).to_numpy().reshape(-1),
    }

    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEATURE_OUTPUT_DIR / "train.npz",
        numeric=train_numeric.to_numpy(np.float32),
        policy=train_policy.to_numpy(np.float32),
        time=train_ts,
        cls=targets_train["cls"],
        reg=targets_train["reg"],
        iso=train_df["iso_code"].to_numpy(),
        year=train_df["year"].to_numpy(),
    )
    np.savez(
        FEATURE_OUTPUT_DIR / "val.npz",
        numeric=val_numeric.to_numpy(np.float32),
        policy=val_policy.to_numpy(np.float32),
        time=val_ts,
        cls=targets_val["cls"],
        reg=targets_val["reg"],
        iso=val_df["iso_code"].to_numpy(),
        year=val_df["year"].to_numpy(),
    )

    metadata: Dict[str, List[str] | int] = {
        "numeric_columns": train_numeric.columns.tolist(),
        "policy_columns": train_policy.columns.tolist(),
        "time_series_columns": TIME_SERIES_COLUMNS,
        "sequence_length": SEQ_LENGTH,
    }
    with METADATA_PATH.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved feature tensors and metadata to %s", FEATURE_OUTPUT_DIR)


if __name__ == "__main__":
    build_feature_tensors()
