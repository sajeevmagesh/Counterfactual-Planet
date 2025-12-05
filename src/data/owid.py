from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import PROCESSED_DIR, normalize_iso_code, write_parquet

OWID_COLUMNS = [
    "iso_code",
    "country",
    "year",
    "co2",
    "co2_per_capita",
    "gdp",
    "population",
    "primary_energy_consumption",
]


def load_owid(path: str | Path) -> pd.DataFrame:
    """Load and clean the OWID CO₂ dataset."""
    df = pd.read_csv(path, usecols=OWID_COLUMNS)
    df["iso_code"] = df["iso_code"].apply(normalize_iso_code)
    df = df.dropna(subset=["iso_code"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    numeric_cols = [col for col in OWID_COLUMNS if col not in {"iso_code", "country", "year"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["co2"])
    df = df.sort_values(["iso_code", "year"]).reset_index(drop=True)

    output_path = PROCESSED_DIR / "owid_clean.parquet"
    write_parquet(df, output_path)
    return df


if __name__ == "__main__":
    load_owid("data/owid-co2-data.csv")
