"""Counterfactual scenario engine for policy shocks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.fusion_model import MultimodalFusionModel, load_config_from_metadata


FEATURE_DIR = Path("data_processed/feature_tensors")
TRAIN_NPZ = FEATURE_DIR / "train.npz"
VAL_NPZ = FEATURE_DIR / "val.npz"
METADATA_PATH = FEATURE_DIR / "metadata.json"
MODEL_PATH = Path("models/multimodal/fusion_model.pt")
COUNTERFACTUAL_DIR = Path("results/counterfactuals")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_metadata() -> Dict[str, Any]:
    with METADATA_PATH.open() as fh:
        return json.load(fh)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "numeric": data["numeric"],
        "policy": data["policy"],
        "time": data["time"],
        "iso": data["iso"],
        "year": data["year"],
    }


def _load_datasets() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    return _load_npz(TRAIN_NPZ), _load_npz(VAL_NPZ)


def _build_column_map(columns: list[str]) -> Dict[str, int]:
    return {col: idx for idx, col in enumerate(columns)}


def _find_row(
    iso_code: str,
    year: int,
    datasets: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for ds in datasets:
        iso_arr = np.array([i.decode("utf-8") if isinstance(i, bytes) else str(i) for i in ds["iso"]])
        mask = (iso_arr == iso_code) & (ds["year"] == year)
        idx = np.where(mask)[0]
        if idx.size:
            row = idx[0]
            return ds["numeric"][row].copy(), ds["time"][row].copy(), ds["policy"][row].copy()
    raise ValueError(f"Row for {iso_code} {year} not found in feature tensors")


def _load_model(metadata: Dict[str, Any]) -> MultimodalFusionModel:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    config = load_config_from_metadata(metadata)
    model = MultimodalFusionModel(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _compute_policy_delta(policy_dim: int, scale: float = 0.05) -> np.ndarray:
    base = np.ones(policy_dim, dtype=np.float32)
    norm = np.linalg.norm(base)
    return (base / norm) * scale


def _apply_numeric_adjustments(
    numeric: np.ndarray,
    numeric_map: Dict[str, int],
    changes: Dict[str, Any],
) -> None:
    def bump(col: str, delta: float) -> None:
        if col in numeric_map:
            numeric[numeric_map[col]] += delta

    if changes.get("add_carbon_tax"):
        bump("cpd_policy_count", 1.0)
        bump("num_active_policies", 1.0)
        bump("cpd_avg_impact", 0.1)

    renew_delta = changes.get("increase_renewables_policies")
    if isinstance(renew_delta, (int, float)):
        bump("cpd_policy_count", float(renew_delta))
        bump("cpd_avg_impact", 0.05 * float(renew_delta))

    stringency_delta = changes.get("raise_stringency")
    if isinstance(stringency_delta, (int, float)):
        bump("cpd_avg_stringency", float(stringency_delta))

    if changes.get("remove_fossil_subsidies"):
        for col in ("coal_share", "oil_share", "gas_share"):
            bump(col, -0.02)
        for col in ("renewable_share", "cpd_avg_impact"):
            bump(col, 0.02)


def _apply_policy_embedding_adjustments(
    policy: np.ndarray,
    delta_vec: np.ndarray,
    changes: Dict[str, Any],
) -> None:
    shift = 0.0
    if changes.get("add_carbon_tax"):
        shift += 1.0
    renew = changes.get("increase_renewables_policies")
    if isinstance(renew, (int, float)):
        shift += float(renew) * 0.2
    if changes.get("remove_fossil_subsidies"):
        shift += 0.5
    if isinstance(changes.get("raise_stringency"), (int, float)):
        shift += float(changes["raise_stringency"]) * 0.3
    if shift != 0.0:
        policy += delta_vec * shift


def _run_model(
    model: MultimodalFusionModel,
    numeric: np.ndarray,
    time_ts: np.ndarray,
    policy: np.ndarray,
) -> Dict[str, float]:
    numeric_t = torch.from_numpy(numeric).unsqueeze(0).to(DEVICE)
    time_t = torch.from_numpy(time_ts).unsqueeze(0).to(DEVICE)
    policy_t = torch.from_numpy(policy).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(numeric_t, time_t, policy_t, modality_dropout=False)
    prob = torch.sigmoid(outputs["logits"]).item()
    delta_co2 = outputs["regression"].item()
    return {"prob": float(prob), "delta_co2": float(delta_co2)}


def simulate_policy_change(iso_code: str, year: int, changes: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _load_metadata()
    train_ds, val_ds = _load_datasets()
    numeric_cols = metadata["numeric_columns"]
    numeric_map = _build_column_map(numeric_cols)
    policy_dim = len(metadata["policy_columns"])

    numeric, time_ts, policy = _find_row(iso_code, year, (val_ds, train_ds))

    model = _load_model(metadata)
    baseline = _run_model(model, numeric.copy(), time_ts.copy(), policy.copy())

    modified_numeric = numeric.copy()
    modified_policy = policy.copy()
    _apply_numeric_adjustments(modified_numeric, numeric_map, changes)
    delta_vec = _compute_policy_delta(policy_dim)
    _apply_policy_embedding_adjustments(modified_policy, delta_vec, changes)

    counterfactual = _run_model(model, modified_numeric, time_ts, modified_policy)
    delta_prob = counterfactual["prob"] - baseline["prob"]
    direction_shift = (baseline["prob"] > 0.5) != (counterfactual["prob"] > 0.5)

    payload = {
        "iso_code": iso_code,
        "year": year,
        "changes": changes,
        "baseline": baseline,
        "counterfactual": counterfactual,
        "delta_prob": delta_prob,
        "direction_shift": direction_shift,
    }

    COUNTERFACTUAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = COUNTERFACTUAL_DIR / f"{iso_code}_{year}_{timestamp}.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def _parse_changes(changes_str: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(changes_str)
        if not isinstance(parsed, dict):  # pragma: no cover - CLI guard
            raise ValueError
        return parsed
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise argparse.ArgumentTypeError("changes must be valid JSON object") from exc


def main() -> None:  # pragma: no cover - convenience CLI
    parser = argparse.ArgumentParser(description="Simulate policy change counterfactuals")
    parser.add_argument("iso_code", type=str)
    parser.add_argument("year", type=int)
    parser.add_argument("changes", type=_parse_changes, help='JSON, e.g. "{\\"add_carbon_tax\\": true}"')
    args = parser.parse_args()
    result = simulate_policy_change(args.iso_code, args.year, args.changes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
