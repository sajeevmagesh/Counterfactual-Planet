from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path("data_processed")
MASTER_PANEL_PATH = PROCESSED_DIR / "master_panel.parquet"
CPD_EXPANDED_PATH = PROCESSED_DIR / "cpd_expanded.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

ACTIVE_STATUS_KEYWORDS = ("in force", "implementation", "active", "operational")
HIGH_STRINGENCY_THRESHOLD = 4

RATING_SCORE_MAP = {
    "critically insufficient": 1,
    "highly insufficient": 2,
    "insufficient": 3,
    "almost sufficient": 4,
    "almost sufficient ": 4,
    "1.5°c compatible": 5,
    "1.5c compatible": 5,
    "1.5°c paris agreement compatible": 5,
    "1.5c paris agreement compatible": 5,
}


def slugify(value: str, prefix: str) -> str:
    """Return a safe column name."""
    value = value or "unknown"
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return f"{prefix}_{value or 'unknown'}"


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-year CO2 targets per country and drop trailing rows."""
    df = df.sort_values(["iso_code", "year"]).reset_index(drop=True)
    grouped = df.groupby("iso_code", group_keys=False)
    df["co2_next"] = grouped["co2"].shift(-1)
    df["delta_co2"] = df["co2_next"] - df["co2"]
    df["co2_direction"] = (df["delta_co2"] > 0).astype(int)
    df = df[df["co2_next"].notna()].reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame, columns: Sequence[str], lags: Iterable[int]) -> None:
    """Append lagged values for each numeric column."""
    grouped = df.groupby("iso_code")
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = grouped[col].shift(lag)


def add_rolling_statistics(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Add rolling means and std devs for core macro variables."""
    grouped = df.groupby("iso_code")
    for col in columns:
        for window in (3, 5):
            df[f"{col}_roll_mean_{window}"] = (
                grouped[col]
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )
        df[f"{col}_roll_std_3"] = (
            grouped[col]
            .rolling(window=3, min_periods=3)
            .std()
            .reset_index(level=0, drop=True)
        )


def compute_policy_features(path: Path = CPD_EXPANDED_PATH) -> pd.DataFrame:
    """Aggregate CPD policy metadata into yearly features."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CPD data at {path}")

    cpd = pd.read_parquet(path)
    if cpd.empty:
        return pd.DataFrame(columns=["iso_code", "year"])

    status = cpd["policy_status"].fillna("").str.lower()
    cpd["is_active"] = status.apply(
        lambda text: any(keyword in text for keyword in ACTIVE_STATUS_KEYWORDS)
    )
    cpd["is_high_stringency"] = cpd["stringency_score"] >= HIGH_STRINGENCY_THRESHOLD

    summary = (
        cpd.groupby(["iso_code", "year"])
        .agg(
            policy_total=("policy_id", "nunique"),
            policy_active=("is_active", "sum"),
            policy_high_stringency=("is_high_stringency", "sum"),
            policy_avg_stringency=("stringency_score", "mean"),
        )
        .reset_index()
    )

    # Sector counts
    sector_counts = (
        cpd.assign(sector_slug=cpd["sector"].fillna("Unknown").apply(lambda x: slugify(x, "sector")))
        .groupby(["iso_code", "year", "sector_slug"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Policy instrument counts
    instrument_counts = (
        cpd.assign(
            instrument_slug=cpd["policy_instrument"]
            .fillna("Unknown")
            .apply(lambda x: slugify(x, "instrument"))
        )
        .groupby(["iso_code", "year", "instrument_slug"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    policy_features = summary.merge(sector_counts, on=["iso_code", "year"], how="left")
    policy_features = policy_features.merge(instrument_counts, on=["iso_code", "year"], how="left")
    policy_features = policy_features.fillna(0)
    return policy_features


def add_policy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join policy aggregations onto the master panel."""
    policy_df = compute_policy_features()
    df = df.merge(policy_df, on=["iso_code", "year"], how="left")
    return df


def add_cat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer extra CAT scenario features."""
    df["cat_current_mean"] = df[["cat_current_min", "cat_current_max"]].mean(axis=1)
    df["cat_ndc_mean"] = df[["cat_ndc_min", "cat_ndc_max"]].mean(axis=1)
    df["cat_current_vs_1p5"] = df["cat_current_mean"] - df["cat_1p5_domestic"]
    df["cat_current_vs_ndc"] = df["cat_current_mean"] - df["cat_ndc_mean"]

    fair_share_score = (
        df["cat_fair_share_target"]
        .fillna("")
        .str.strip()
        .str.lower()
        .map(RATING_SCORE_MAP)
    )
    df["cat_fair_share_score"] = fair_share_score
    df["cat_fair_share_gap"] = df["cat_overall_rating_score"] - fair_share_score
    return df


def encode_cat_rating(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode CAT overall rating after lag creation."""
    rating_series = df["cat_overall_rating"].str.strip().str.lower()
    rating_dummies = pd.get_dummies(
        rating_series,
        prefix="cat_rating",
        dummy_na=True,
    )
    return pd.concat([df, rating_dummies], axis=1)


def add_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional energy-structure indicators."""
    df["fossil_share"] = df[["coal_share", "gas_share", "oil_share"]].sum(axis=1, min_count=1)
    df["fossil_to_renewable_ratio"] = df["fossil_share"] / df["renewable_share"].replace(0, np.nan)
    df["elec_co2_intensity_per_gdp"] = df["elec_co2_intensity"] / df["gdp"].replace(0, np.nan)
    df["electricity_emissions_intensity"] = df["elec_co2_intensity"]
    return df


def scale_continuous_features(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    """Standardize continuous-valued columns with sklearn."""
    exclude_set = set(exclude)
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_float_dtype(df[col]) and col not in exclude_set
    ]
    if not numeric_cols:
        return df

    scaler = StandardScaler()
    numeric_frame = df[numeric_cols]
    fill_values = numeric_frame.mean()
    filled = numeric_frame.fillna(fill_values)
    scaled = scaler.fit_transform(filled)
    scaled_df = pd.DataFrame(
        scaled,
        columns=[f"{col}_scaled" for col in numeric_cols],
        index=df.index,
    )
    scaled_df[numeric_frame.isna()] = np.nan
    df = pd.concat([df, scaled_df], axis=1)
    return df


def build_features() -> pd.DataFrame:
    """End-to-end feature engineering pipeline."""
    if not MASTER_PANEL_PATH.exists():
        raise FileNotFoundError("Master panel not found. Run merge_all first.")

    df = pd.read_parquet(MASTER_PANEL_PATH)
    df = create_targets(df)
    df = add_policy_features(df)
    df = add_cat_features(df)
    df = add_energy_features(df)

    base_numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"year", "co2_next", "delta_co2", "co2_direction"}
    ]
    add_lag_features(df, base_numeric_cols, lags=(1, 3, 5))
    add_rolling_statistics(
        df,
        columns=[
            "co2",
            "gdp",
            "population",
            "primary_energy_consumption",
        ],
    )
    df = encode_cat_rating(df)

    df = scale_continuous_features(
        df,
        exclude={"year", "co2_direction"},
    )

    df.to_parquet(FEATURES_PATH, index=False)
    print(
        f"Built feature matrix with {len(df)} rows and {len(df.columns)} columns -> {FEATURES_PATH}"
    )
    return df


if __name__ == "__main__":
    build_features()
