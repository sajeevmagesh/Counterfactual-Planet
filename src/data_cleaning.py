"""Data cleaning and normalization utilities for the multimodal pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MASTER_PANEL_PATH = Path("data_processed/master_panel.parquet")
POLICY_EMBED_PATH = Path("data_processed/policy_embeddings.parquet")
CLEAN_FULL_PATH = Path("data_processed/cleaned.csv")
CLEAN_TRAIN_PATH = Path("data_processed/cleaned_train.parquet")
CLEAN_VAL_PATH = Path("data_processed/cleaned_val.parquet")
SCALER_PATH = Path("data_processed/scaler_stats.parquet")

LOG_COLUMNS = ["co2", "co2_per_capita", "gdp", "population", "primary_energy_consumption"]
CLIP_COLUMNS = {"delta_co2": 0.995}


@dataclass
class CleanedData:
    """Container for cleaned datasets and scalers."""

    train: pd.DataFrame
    val: pd.DataFrame
    scaler: StandardScaler
    numeric_columns: List[str]


def load_master_panel(path: Path = MASTER_PANEL_PATH) -> pd.DataFrame:
    """Load the master panel parquet."""

    if not path.exists():
        raise FileNotFoundError(f"Master panel not found at {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded master panel with %d rows and %d columns", len(df), df.shape[1])
    return df


def merge_policy_embeddings(df: pd.DataFrame, embed_path: Path = POLICY_EMBED_PATH) -> pd.DataFrame:
    """Attach policy embeddings, creating mask/count features."""

    if not embed_path.exists():
        logger.warning("Policy embeddings not found at %s; using zero vectors.", embed_path)
        df["policy_mask"] = 0.0
        df["num_active_policies"] = 0.0
        return df

    embeds = pd.read_parquet(embed_path)
    logger.info("Loaded %d policy embedding rows", len(embeds))
    embed_cols = [col for col in embeds.columns if col.startswith("policy_embed_")]
    df = df.merge(embeds, on=["iso_code", "year"], how="left")
    mask = (~df[embed_cols].isna().all(axis=1)).astype(float)
    df["policy_mask"] = mask
    df[embed_cols] = df[embed_cols].fillna(0.0)
    if "policy_total" in df.columns:
        df["num_active_policies"] = df["policy_total"].fillna(0.0)
    elif "policy_type_count_unknown" in df.columns:
        df["num_active_policies"] = df[[col for col in df.columns if col.startswith("policy_type_count_")]].sum(axis=1)
    else:
        df["num_active_policies"] = 0.0
    return df


def apply_log_transforms(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Apply log1p to selected columns."""

    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regression deltas and smoothed classification direction."""

    df = df.sort_values(["iso_code", "year"]).reset_index(drop=True)
    df["co2_next"] = df.groupby("iso_code")["co2"].shift(-1)
    df["delta_co2"] = df["co2_next"] - df["co2"]

    horizons = (1, 2, 3, 4, 5)
    for horizon in horizons:
        df[f"co2_future_{horizon}"] = df.groupby("iso_code")["co2"].shift(-horizon)

    future_cols = [f"co2_future_{h}" for h in horizons]
    complete_window = df[future_cols].notna().all(axis=1)
    df = df[complete_window].copy()
    df["future_avg"] = df[future_cols].mean(axis=1)
    df["co2_direction"] = (df["future_avg"] > df["co2"]).astype(np.int64)

    balance = df["co2_direction"].value_counts(normalize=True).to_dict()
    logger.info("Smoothed label balance: %s", balance)
    print("Smoothed label balance:", balance)

    df = df.drop(columns=future_cols + ["future_avg"])
    logger.info("Computed delta_co2 and smoothed co2_direction targets")
    return df


def clip_extremes(df: pd.DataFrame, clip_spec: Dict[str, float]) -> pd.DataFrame:
    """Clip columns by specified upper quantile."""

    for col, quantile in clip_spec.items():
        if col in df.columns:
            high = df[col].quantile(quantile)
            low = df[col].quantile(1 - quantile)
            df[col] = df[col].clip(lower=low, upper=high)
            logger.info("Clipped %s to quantiles %.3f/%.3f", col, 1 - quantile, quantile)
    return df


def drop_zero_variance(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    """Remove zero-variance columns from numeric list."""

    keep_cols = []
    for col in numeric_cols:
        if df[col].var() == 0 or np.isnan(df[col].var()):
            logger.warning("Dropping zero-variance column: %s", col)
        else:
            keep_cols.append(col)
    return keep_cols


def train_val_split(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by country."""

    iso_codes = df["iso_code"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(iso_codes)
    val_size = max(1, int(len(iso_codes) * val_frac))
    val_set = set(iso_codes[:val_size])
    val_df = df[df["iso_code"].isin(val_set)].copy()
    train_df = df[~df["iso_code"].isin(val_set)].copy()
    logger.info("Train rows: %d, Val rows: %d", len(train_df), len(val_df))
    return train_df, val_df


def standardize_frames(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train set and transform both splits."""

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    train_df[numeric_cols] = train_df[numeric_cols].astype(np.float32)
    val_df[numeric_cols] = val_df[numeric_cols].astype(np.float32)
    return train_df, val_df, scaler


def clean_data(
    master_path: Path = MASTER_PANEL_PATH,
    embed_path: Path = POLICY_EMBED_PATH,
    val_frac: float = 0.2,
    seed: int = 42,
) -> CleanedData:
    """End-to-end data cleaning pipeline."""

    df = load_master_panel(master_path)
    df = merge_policy_embeddings(df, embed_path)
    df = build_targets(df)
    df = apply_log_transforms(df, LOG_COLUMNS)
    df = clip_extremes(df, CLIP_COLUMNS)
    numeric_cols = [col for col in df.columns if pd.api.types.is_float_dtype(df[col])]
    numeric_cols = [col for col in numeric_cols if col not in {"co2_next", "co2_direction"}]
    numeric_cols = drop_zero_variance(df, numeric_cols)
    df[numeric_cols] = df[numeric_cols].astype(np.float32).fillna(0.0)
    train_df, val_df = train_val_split(df, val_frac, seed)
    train_df, val_df, scaler = standardize_frames(train_df, val_df, numeric_cols)

    combined = pd.concat(
        [train_df.assign(dataset_split="train"), val_df.assign(dataset_split="val")],
        ignore_index=True,
    )

    CLEAN_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CLEAN_FULL_PATH, index=False)
    train_df.to_parquet(CLEAN_TRAIN_PATH, index=False)
    val_df.to_parquet(CLEAN_VAL_PATH, index=False)
    logger.info("Saved cleaned train/val data")
    return CleanedData(train=train_df, val=val_df, scaler=scaler, numeric_columns=numeric_cols)


if __name__ == "__main__":
    clean_data()
