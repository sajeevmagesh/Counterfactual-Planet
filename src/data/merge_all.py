from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import PROCESSED_DIR, write_parquet

MASTER_PANEL_PATH = PROCESSED_DIR / "master_panel.parquet"


def _read_processed(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_parquet(path)


def _aggregate_cpd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["iso_code", "year", "cpd_policy_count", "cpd_avg_stringency", "cpd_avg_impact"]
        )
    aggregated = (
        df.groupby(["iso_code", "year"])
        .agg(
            cpd_policy_count=("policy_id", "nunique"),
            cpd_avg_stringency=("stringency_score", "mean"),
            cpd_avg_impact=("impact_value", "mean"),
        )
        .reset_index()
    )
    return aggregated


def build_master_panel() -> pd.DataFrame:
    """Merge every processed dataset into the modeling panel."""
    owid = _read_processed("owid_clean.parquet")
    cpd_expanded = _read_processed("cpd_expanded.parquet")
    ember = _read_processed("ember_features.parquet")
    cat_ts = _read_processed("cat_timeseries.parquet")
    cat_ratings = _read_processed("cat_ratings.parquet")

    cpd_yearly = _aggregate_cpd(cpd_expanded)

    panel = owid.merge(cpd_yearly, on=["iso_code", "year"], how="left")
    panel = panel.merge(ember, on=["iso_code", "year"], how="left")
    panel = panel.merge(cat_ts, on=["iso_code", "year"], how="left")
    panel = panel.merge(cat_ratings, on="iso_code", how="left")

    panel = panel.dropna(subset=["co2"]).sort_values(["iso_code", "year"]).reset_index(drop=True)
    write_parquet(panel, MASTER_PANEL_PATH)
    return panel


if __name__ == "__main__":
    build_master_panel()
