import argparse
from pathlib import Path

import pandas as pd


YEARS = list(range(2000, 2023))


def filter_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows for countries that have data for every year in YEARS."""
    # Only consider rows in the requested year span.
    df_years = df[df["year"].between(YEARS[0], YEARS[-1])]

    expected = set(YEARS)
    eligible_countries = {
        country
        for country, group in df_years.groupby("country")
        if set(group["year"].unique()) == expected
    }

    return df_years[df_years["country"].isin(eligible_countries)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter OWID CO₂ data for countries with complete 2000-2022 coverage."
    )
    parser.add_argument(
        "--input",
        default="owid-co2-data.csv",
        type=Path,
        help="Path to the raw OWID CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="owid-co2-data-2000-2022.csv",
        type=Path,
        help="Path for the filtered CSV (default: %(default)s)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    filtered = filter_countries(df)
    filtered.to_csv(args.output, index=False)

    print(
        f"Wrote {len(filtered)} rows covering "
        f"{filtered['country'].nunique()} countries to {args.output}"
    )


if __name__ == "__main__":
    main()
