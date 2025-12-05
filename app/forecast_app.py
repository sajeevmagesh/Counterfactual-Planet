"""Gradio-based CO₂ forecast sandbox (no Streamlit required)."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import torch

import sys

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


@functools.lru_cache(maxsize=1)
def _load_app_state() -> Dict[str, Any]:
    metadata = _load_metadata()
    train_ds, val_ds = _load_datasets()
    iso_codes = sorted(set(
        str(i, "utf-8") if isinstance(i, bytes) else str(i)
        for ds in (train_ds, val_ds)
        for i in ds["iso"]
    ))
    years = sorted(set(np.concatenate([train_ds["year"], val_ds["year"]]).tolist()))
    model = _load_model(metadata)
    numeric_map = _build_column_map(metadata["numeric_columns"])
    policy_delta = _compute_policy_delta(len(metadata["policy_columns"]))
    return {
        "metadata": metadata,
        "train": train_ds,
        "val": val_ds,
        "iso_codes": iso_codes,
        "years": years,
        "model": model,
        "numeric_map": numeric_map,
        "policy_delta": policy_delta,
    }


def _prep_tables(
    numeric: np.ndarray,
    modified_numeric: np.ndarray,
    policy: np.ndarray,
    modified_policy: np.ndarray,
    metadata: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_cols = metadata["numeric_columns"]
    numeric_df = pd.DataFrame([numeric], columns=numeric_cols).T.rename(columns={0: "baseline"})
    delta_numeric = modified_numeric - numeric
    numeric_df["delta"] = delta_numeric
    changed_df = numeric_df[numeric_df["delta"] != 0].sort_values("delta", key=np.abs, ascending=False)

    policy_cols = metadata["policy_columns"]
    baseline_policy = pd.DataFrame(
        {
            "dimension": policy_cols,
            "baseline": policy,
            "modified": modified_policy,
            "delta": modified_policy - policy,
        }
    )
    top_policy = baseline_policy.reindex(baseline_policy["delta"].abs().sort_values(ascending=False).index).head(10)

    return numeric_df.head(20), changed_df.head(20), top_policy


def forecast_interface(
    iso_code: str,
    year: int,
    add_carbon_tax: bool,
    renewals: int,
    stringency: float,
    remove_fossil: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = _load_app_state()
    metadata = state["metadata"]
    model = state["model"]
    numeric_map = state["numeric_map"]
    policy_delta = state["policy_delta"]

    numeric, time_ts, policy = _find_row(iso_code, year, (state["val"], state["train"]))

    changes = {
        "add_carbon_tax": add_carbon_tax,
        "increase_renewables_policies": renewals,
        "raise_stringency": stringency,
        "remove_fossil_subsidies": remove_fossil,
    }

    baseline = _run_model(model, numeric.copy(), time_ts.copy(), policy.copy())

    modified_numeric = numeric.copy()
    modified_policy = policy.copy()
    _apply_numeric_adjustments(modified_numeric, numeric_map, changes)
    _apply_policy_embedding_adjustments(modified_policy, policy_delta, changes)
    counterfactual = _run_model(model, modified_numeric, time_ts, modified_policy)

    summary = {
        "country": iso_code,
        "year": year,
        "changes": changes,
        "baseline_prob": round(baseline["prob"], 4),
        "counterfactual_prob": round(counterfactual["prob"], 4),
        "delta_prob": round(counterfactual["prob"] - baseline["prob"], 4),
        "baseline_dco2": round(baseline["delta_co2"], 4),
        "counterfactual_dco2": round(counterfactual["delta_co2"], 4),
        "delta_dco2": round(counterfactual["delta_co2"] - baseline["delta_co2"], 4),
        "direction_shift": (baseline["prob"] > 0.5) != (counterfactual["prob"] > 0.5),
    }

    tables = _prep_tables(numeric, modified_numeric, policy, modified_policy, metadata)
    return summary, tables[0], tables[1], tables[2]


def app_main() -> None:  # pragma: no cover - UI entry
    state = _load_app_state()
    with gr.Blocks(title="CO₂ Policy Sandbox") as demo:
        gr.Markdown("# CO₂ Policy Sandbox\n### Explore how policy levers shift multimodal forecasts")
        with gr.Row():
            iso_input = gr.Dropdown(state["iso_codes"], label="Country", value="USA" if "USA" in state["iso_codes"] else state["iso_codes"][0])
            year_input = gr.Dropdown(state["years"], label="Year", value=state["years"][-1])
        with gr.Row():
            tax_input = gr.Checkbox(label="Add carbon tax")
            renew_input = gr.Slider(0, 10, value=0, step=1, label="Increase renewables policies")
            stringency_input = gr.Slider(0, 5, value=0, step=0.1, label="Raise stringency")
            fossil_input = gr.Checkbox(label="Remove fossil subsidies")

        run_button = gr.Button("Run Scenario")

        summary_out = gr.JSON(label="Prediction Summary")
        baseline_numeric_out = gr.Dataframe(label="Baseline Numeric Snapshot", interactive=False)
        delta_numeric_out = gr.Dataframe(label="Changed Numeric Features", interactive=False)
        policy_out = gr.Dataframe(label="Policy Embedding Changes", interactive=False)

        run_button.click(
            forecast_interface,
            inputs=[iso_input, year_input, tax_input, renew_input, stringency_input, fossil_input],
            outputs=[summary_out, baseline_numeric_out, delta_numeric_out, policy_out],
        )

    demo.launch()


if __name__ == "__main__":
    app_main()
