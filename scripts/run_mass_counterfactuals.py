"""Batch counterfactual experiments across countries and years."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation.counterfactuals import (
    _apply_numeric_adjustments,
    _apply_policy_embedding_adjustments,
    _build_column_map,
    _compute_policy_delta,
    _find_row,
    _load_datasets,
    _load_metadata,
    _load_model,
    _run_model,
)


RESULT_DIR = Path("results/counterfactuals")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MASS_OUTPUT = RESULT_DIR / "mass_simulation.jsonl"
SUMMARY_OUTPUT = RESULT_DIR / "summary.json"


def load_iso_income_map(metadata: Dict[str, List[str]], datasets: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]) -> Dict[str, str]:
    income_map = metadata.get("income_map")
    if income_map:
        return income_map
    train, _ = datasets
    iso_arr = train["iso"]
    growth_cols = [c for c in metadata["numeric_columns"] if c.endswith("gdp_growth")]
    gdp_idx = metadata["numeric_columns"].index(growth_cols[0]) if growth_cols else None
    iso_income: Dict[str, Tuple[float, int]] = {}
    for iso_code, numeric in zip(iso_arr, train["numeric"]):
        iso_str = iso_code.decode("utf-8") if isinstance(iso_code, bytes) else str(iso_code)
        value = float(numeric[gdp_idx]) if gdp_idx is not None else float(numeric[0])
        total, count = iso_income.get(iso_str, (0.0, 0))
        iso_income[iso_str] = (total + value, count + 1)
    averages = {iso: total / count for iso, (total, count) in iso_income.items() if count > 0}
    values = np.array(list(averages.values()))
    high_threshold = np.percentile(values, 66)
    low_threshold = np.percentile(values, 33)
    income_label = {}
    for iso, avg in averages.items():
        if avg >= high_threshold:
            group = "high"
        elif avg >= low_threshold:
            group = "middle"
        else:
            group = "low"
        income_label[iso] = group
    return income_label


def sample_country_years(
    datasets: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    metadata: Dict[str, Any],
    target_per_group: int = 30,
    total_samples: int = 150,
) -> List[Tuple[str, int]]:
    train_ds, val_ds = datasets
    iso_income = load_iso_income_map(metadata, datasets)
    combined = []
    for ds in (train_ds, val_ds):
        iso_arr = [code.decode("utf-8") if isinstance(code, bytes) else str(code) for code in ds["iso"]]
        for iso, year in zip(iso_arr, ds["year"]):
            if 2005 <= year <= 2020:
                combined.append((iso, year))

    grouped: Dict[str, List[Tuple[str, int]]] = {"high": [], "middle": [], "low": []}
    for iso, year in combined:
        tier = iso_income.get(iso)
        if tier:
            grouped[tier].append((iso, year))

    rng = random.Random(42)
    selected = []
    for tier in ("high", "middle", "low"):
        candidates = grouped.get(tier, [])
        rng.shuffle(candidates)
        selected.extend(candidates[:target_per_group])

    if len(selected) < total_samples:
        remaining = [pair for pair in combined if pair not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: total_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:total_samples]


def run_mass_counterfactuals() -> None:
    metadata = _load_metadata()
    datasets = _load_datasets()
    model = _load_model(metadata)
    numeric_map = _build_column_map(metadata["numeric_columns"])
    policy_delta = _compute_policy_delta(len(metadata["policy_columns"]))

    sample_pairs = sample_country_years(datasets, metadata)
    changes = {
        "add_carbon_tax": True,
        "increase_renewables_policies": 5,
        "raise_stringency": 2,
        "remove_fossil_subsidies": True,
    }

    records = []
    for iso, year in sample_pairs:
        try:
            numeric, time_ts, policy = _find_row(iso, year, datasets)
        except ValueError:
            continue
        baseline = _run_model(model, numeric.copy(), time_ts.copy(), policy.copy())
        modified_numeric = numeric.copy()
        modified_policy = policy.copy()
        _apply_numeric_adjustments(modified_numeric, numeric_map, changes)
        _apply_policy_embedding_adjustments(modified_policy, policy_delta, changes)
        counterfactual = _run_model(model, modified_numeric, time_ts, modified_policy)
        record = {
            "iso": iso,
            "year": int(year),
            "baseline_prob": baseline["prob"],
            "counterfactual_prob": counterfactual["prob"],
            "delta_prob": counterfactual["prob"] - baseline["prob"],
            "baseline_reg": baseline["delta_co2"],
            "counterfactual_reg": counterfactual["delta_co2"],
            "delta_reg": counterfactual["delta_co2"] - baseline["delta_co2"],
            "direction_shift": (baseline["prob"] > 0.5) != (counterfactual["prob"] > 0.5),
        }
        records.append(record)

    with MASS_OUTPUT.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    if not records:
        print("No records produced; check data availability.")
        return

    deltas_prob = [r["delta_prob"] for r in records]
    deltas_reg = [r["delta_reg"] for r in records]
    histogram = Counter(int(dp * 100) / 100 for dp in deltas_prob)
    summary = {
        "mean_delta_prob": mean(deltas_prob),
        "max_delta_prob": max(deltas_prob),
        "min_delta_prob": min(deltas_prob),
        "mean_delta_reg": mean(deltas_reg),
        "direction_shifts": sum(r["direction_shift"] for r in records),
        "histogram_delta_prob": histogram,
        "n_records": len(records),
    }
    with SUMMARY_OUTPUT.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("=== Mass Counterfactual Summary ===")
    print(f"Records: {len(records)}")
    print(f"Mean delta prob: {summary['mean_delta_prob']:.6f}")
    print(f"Max delta prob: {summary['max_delta_prob']:.6f}")
    print(f"Min delta prob: {summary['min_delta_prob']:.6f}")
    print(f"Mean delta regression: {summary['mean_delta_reg']:.6f}")
    print(f"Direction shifts: {summary['direction_shifts']}")
    print("Histogram of delta prob (rounded to 0.01):")
    for bucket, count in sorted(histogram.items()):
        print(f"  {bucket:+.2f}: {count}")


if __name__ == "__main__":
    run_mass_counterfactuals()
