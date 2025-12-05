from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .utils import PROCESSED_DIR, normalize_iso_code, write_parquet

CAT_TIMESERIES_PATH = PROCESSED_DIR / "cat_timeseries.parquet"
ALLOWED_INDICATORS = {
    "Emissions (absolute)",
    "Modelled Domestic Pathways boundaries (absolute)",
}

SCENARIO_COLUMN_MAP: Dict[str, str] = {
    "Historical": "cat_hist",
    "Current Policy, Min": "cat_current_min",
    "Current Policy, Max": "cat_current_max",
    "NDC Unconditional, Min": "cat_ndc_min",
    "NDC Unconditional, Max": "cat_ndc_max",
    "NDC Conditional, Min": "cat_ndc_cond_min",
    "NDC Conditional, Max": "cat_ndc_cond_max",
    "NDC Other, Min": "cat_ndc_other_min",
    "NDC Other, Max": "cat_ndc_other_max",
    "1.5C compatible": "cat_1p5_domestic",
    "Almost sufficient": "cat_almost_sufficient",
    "Highly insufficient": "cat_highly_insufficient",
    "Insufficient": "cat_insufficient",
    "Lower limit": "cat_lower_limit",
    "Upper limit": "cat_upper_limit",
    "Critically insufficient": "cat_critically_insufficient",
}


def _find_header_row(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            stripped = line.lstrip(",")
            if stripped.startswith("Version"):
                return idx
    raise RuntimeError("Failed to locate CAT header row.")


def load_cat_timeseries(path: str | Path) -> pd.DataFrame:
    """Load, filter, and melt the CAT time-series export."""
    path = Path(path)
    header_row = _find_header_row(path)
    df = pd.read_csv(path, skiprows=header_row, encoding="utf-8-sig")
    df = df.loc[
        df["Sector"] == "Economy-wide, excluding LULUCF",
        [
            "Version",
            "Country",
            "Scenario",
            "Sector",
            "Indicator",
            "Unit",
        ]
        + [col for col in df.columns if col.isdigit()],
    ]
    df = df[df["Indicator"].isin(ALLOWED_INDICATORS)]
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    df["iso_code"] = df["Country"].apply(normalize_iso_code)
    df = df.dropna(subset=["iso_code"])

    year_columns = [col for col in df.columns if isinstance(col, str) and col.isdigit()]
    id_vars = ["Version", "iso_code", "Country", "Scenario", "Indicator"]
    melted = df.melt(id_vars=id_vars, value_vars=year_columns, var_name="year", value_name="value")
    melted["year"] = pd.to_numeric(melted["year"], errors="coerce")
    melted = melted.dropna(subset=["year"])
    melted["year"] = melted["year"].astype(int)
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted["Version"] = pd.to_numeric(melted["Version"], errors="coerce")
    melted = melted.dropna(subset=["year"])
    return melted


def build_cat_scenario_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot scenario-specific series into a wide panel."""
    df = df.copy()
    df = df.sort_values("Version").drop_duplicates(
        subset=["iso_code", "Scenario", "year"], keep="last"
    )

    df = df[df["Scenario"].isin(SCENARIO_COLUMN_MAP)]
    df["scenario_column"] = df["Scenario"].map(SCENARIO_COLUMN_MAP)

    panel = (
        df.pivot_table(
            index=["iso_code", "year"],
            columns="scenario_column",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    write_parquet(panel, CAT_TIMESERIES_PATH)
    return panel


if __name__ == "__main__":
    long_df = load_cat_timeseries("data/CAT_30092025_CountryAssessmentData_DataExplorer.csv")
    build_cat_scenario_panel(long_df)
