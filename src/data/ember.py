from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .utils import PROCESSED_DIR, normalize_iso_code, write_parquet

EMBER_FEATURES_PATH = PROCESSED_DIR / "ember_features.parquet"

FEATURE_SPECS = {
    "coal_share": {
        "Category": "Electricity generation",
        "Subcategory": "Fuel",
        "Variable": "Coal",
        "Unit": "%",
    },
    "gas_share": {
        "Category": "Electricity generation",
        "Subcategory": "Fuel",
        "Variable": "Gas",
        "Unit": "%",
    },
    "oil_share": {
        "Category": "Electricity generation",
        "Subcategory": "Fuel",
        "Variable": "Other Fossil",
        "Unit": "%",
    },
    "renewable_share": {
        "Category": "Electricity generation",
        "Subcategory": "Aggregate fuel",
        "Variable": "Renewables",
        "Unit": "%",
    },
    "wind_share": {
        "Category": "Electricity generation",
        "Subcategory": "Fuel",
        "Variable": "Wind",
        "Unit": "%",
    },
    "solar_share": {
        "Category": "Electricity generation",
        "Subcategory": "Fuel",
        "Variable": "Solar",
        "Unit": "%",
    },
    "elec_co2_intensity": {
        "Category": "Power sector emissions",
        "Subcategory": "CO2 intensity",
        "Variable": "CO2 intensity",
        "Unit": "gCO2/kWh",
    },
}


def load_ember(path: str | Path) -> pd.DataFrame:
    """Load the Ember energy release."""
    df = pd.read_csv(path)
    df = df.rename(columns={"ISO 3 code": "iso_code", "Year": "year"})
    df["iso_code"] = df["iso_code"].apply(normalize_iso_code)
    df = df[df["Area type"].str.lower() == "country"]
    df = df.dropna(subset=["iso_code", "year"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def pivot_ember_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot Ember series to iso/year level feature columns."""
    feature_frames: List[pd.DataFrame] = []
    for feature, spec in FEATURE_SPECS.items():
        mask = pd.Series(True, index=df.index)
        for col, value in spec.items():
            mask &= df[col] == value
        subset = df.loc[mask, ["iso_code", "year", "Value"]].dropna(subset=["Value"])
        subset = subset.rename(columns={"Value": feature})
        feature_frames.append(subset)

    features: pd.DataFrame | None = None
    for frame in feature_frames:
        if features is None:
            features = frame
        else:
            features = features.merge(frame, on=["iso_code", "year"], how="outer")

    if features is None:
        features = pd.DataFrame(columns=["iso_code", "year"])

    features = features.drop_duplicates(subset=["iso_code", "year"]).sort_values(
        ["iso_code", "year"]
    )
    for feature in FEATURE_SPECS:
        if feature not in features.columns:
            features[feature] = pd.NA
    ordered_cols = ["iso_code", "year"] + list(FEATURE_SPECS.keys())
    features = features[ordered_cols]
    write_parquet(features, EMBER_FEATURES_PATH)
    return features


if __name__ == "__main__":
    ember = load_ember("data/energy_yearly_full_release_long_format.csv")
    pivot_ember_to_features(ember)
