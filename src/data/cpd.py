from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .utils import PROCESSED_DIR, normalize_iso_code, write_parquet

CPD_EXPANDED_PATH = PROCESSED_DIR / "cpd_expanded.parquet"


def _parse_year(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.year.astype("Int64")


def load_cpd(path: str | Path) -> pd.DataFrame:
    """Load and normalize Climate Policy Database records."""
    df = pd.read_csv(path)
    df = df.rename(columns={"impact_indicators.value": "impact_value"})

    df["country_iso"] = df["country_iso"].apply(normalize_iso_code)
    df = df.dropna(subset=["country_iso"])

    df["start_year"] = _parse_year(df["start_date"])
    df["end_year"] = _parse_year(df["end_date"])
    df["start_year"] = df["start_year"].fillna(_parse_year(df["decision_date"]))
    df["end_year"] = df["end_year"].fillna(df["start_year"])

    df = df.dropna(subset=["start_year"])
    df["start_year"] = df["start_year"].astype(int)
    df["end_year"] = df["end_year"].astype(int)

    df["stringency_score"] = pd.to_numeric(df["stringency"], errors="coerce")
    df["impact_value"] = pd.to_numeric(df["impact_value"], errors="coerce")

    df = df[
        [
            "policy_id",
            "country_iso",
            "policy_name",
            "sector",
            "policy_instrument",
            "policy_type",
            "stringency_score",
            "policy_status",
            "impact_value",
            "start_year",
            "end_year",
        ]
    ].copy()
    return df


def expand_policies_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Expand every policy to one row per (iso_code, year)."""
    rows: List[dict] = []
    for row in df.itertuples():
        start, end = int(row.start_year), int(row.end_year)
        if pd.isna(start):
            continue
        if pd.isna(end):
            end = start
        if end < start:
            end = start
        years: Iterable[int] = range(start, end + 1)
        for year in years:
            rows.append(
                {
                    "iso_code": row.country_iso,
                    "year": year,
                    "policy_id": row.policy_id,
                    "policy_name": row.policy_name,
                    "sector": row.sector,
                    "policy_instrument": row.policy_instrument,
                    "policy_type": row.policy_type,
                    "stringency_score": row.stringency_score,
                    "policy_status": row.policy_status,
                    "impact_value": row.impact_value,
                }
            )
    expanded = pd.DataFrame(rows)
    expanded = expanded.dropna(subset=["iso_code", "year"])
    expanded["year"] = expanded["year"].astype(int)

    write_parquet(expanded, CPD_EXPANDED_PATH)
    return expanded


if __name__ == "__main__":
    policies = load_cpd("data/climate_policy_database_policies_export.csv")
    expand_policies_by_year(policies)
