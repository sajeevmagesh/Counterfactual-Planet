from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import PROCESSED_DIR, load_country_iso_map, write_parquet

CAT_RATINGS_PATH = PROCESSED_DIR / "cat_ratings.parquet"

RATING_SCORES = {
    "critically insufficient": 1,
    "highly insufficient": 2,
    "insufficient": 3,
    "almost sufficient": 4,
    "1.5°c compatible": 5,
    "1.5c compatible": 5,
}

RENAME_COLUMNS = {
    "Country": "country",
    "Overall rating": "overall_rating",
    "Policies and action": "policies_and_action",
    "Domestic or supported target": "domestic_or_supported_target",
    "Fair share target": "fair_share_target",
    "Climate finance": "climate_finance",
    "Net zero target": "net_zero_target",
    "Update date": "update_date",
}


def load_cat_ratings(path: str | Path) -> pd.DataFrame:
    """Load CAT rating table and map ratings to numeric scores."""
    df = pd.read_csv(path)
    df = df.rename(columns=RENAME_COLUMNS)

    iso_map = load_country_iso_map()
    df["iso_code"] = df["country"].map(iso_map)
    df = df.dropna(subset=["iso_code"])

    df["overall_rating_clean"] = df["overall_rating"].str.strip()
    df["overall_rating_score"] = (
        df["overall_rating_clean"].str.lower().map(RATING_SCORES).astype("Int64")
    )

    update_str = df["update_date"].astype(str).str.strip()
    parsed_dates = pd.to_datetime(update_str, format="%b %Y", errors="coerce")
    missing_mask = parsed_dates.isna()
    if missing_mask.any():
        parsed_dates[missing_mask] = pd.to_datetime(
            update_str[missing_mask], errors="coerce"
        )
    df["rating_year"] = parsed_dates.dt.year.astype("Int64")

    output_cols = [
        "iso_code",
        "country",
        "overall_rating_clean",
        "overall_rating_score",
        "policies_and_action",
        "domestic_or_supported_target",
        "fair_share_target",
        "climate_finance",
        "net_zero_target",
        "rating_year",
    ]
    result = df[output_cols].rename(
        columns={
            "country": "cat_country",
            "overall_rating_clean": "cat_overall_rating",
            "overall_rating_score": "cat_overall_rating_score",
            "policies_and_action": "cat_policies_and_action",
            "domestic_or_supported_target": "cat_domestic_or_supported_target",
            "fair_share_target": "cat_fair_share_target",
            "climate_finance": "cat_climate_finance",
            "net_zero_target": "cat_net_zero_target",
        }
    )
    write_parquet(result, CAT_RATINGS_PATH)
    return result


if __name__ == "__main__":
    load_cat_ratings("data/country ratings data.csv")
