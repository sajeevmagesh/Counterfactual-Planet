from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


RAW_DATA_DIR = Path("data")
PROCESSED_DIR = Path("data_processed")


def ensure_processed_dir() -> None:
    """Make sure the processed-data directory exists."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_iso_code(value: str | float | None) -> str | None:
    """Return a three-letter ISO code in uppercase or None if invalid."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    code = str(value).strip().upper()
    if len(code) == 3 and code.isalpha():
        return code
    return None


def load_country_iso_map(
    owid_path: str | Path = RAW_DATA_DIR / "owid-co2-data.csv",
) -> Dict[str, str]:
    """Create a mapping from country names to ISO-3 codes using the OWID file."""
    df = pd.read_csv(owid_path, usecols=["country", "iso_code"])
    df["iso_code"] = df["iso_code"].apply(normalize_iso_code)
    mapping = (
        df.dropna(subset=["iso_code"])
        .drop_duplicates(subset=["country"])
        .set_index("country")["iso_code"]
        .to_dict()
    )
    return mapping


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a DataFrame to Parquet, ensuring directories exist."""
    ensure_processed_dir()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
