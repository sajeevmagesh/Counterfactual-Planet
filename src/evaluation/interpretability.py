"""Interpretability engine for multimodal model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.fusion_model import MultimodalFusionModel, load_config_from_metadata


FEATURE_DIR = Path("data_processed/feature_tensors")
VAL_NPZ = FEATURE_DIR / "val.npz"
METADATA_PATH = FEATURE_DIR / "metadata.json"
MODEL_PATH = Path("models/multimodal/fusion_model.pt")
OUTPUT_DIR = Path("results/interpretability")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_metadata() -> Dict:
    with METADATA_PATH.open() as fh:
        return json.load(fh)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_model(metadata: Dict) -> MultimodalFusionModel:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    config = load_config_from_metadata(metadata)
    model = MultimodalFusionModel(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _prepare_samples(arr: np.ndarray, max_samples: int) -> np.ndarray:
    if len(arr) <= max_samples:
        return arr
    idx = np.random.default_rng(0).choice(len(arr), size=max_samples, replace=False)
    idx.sort()
    return arr[idx]


def _shap_numeric(model, data: Dict[str, np.ndarray], metadata: Dict) -> Dict[str, float]:
    numeric = data["numeric"].astype(np.float32)
    time_tensor = data["time"].astype(np.float32)
    policy_tensor = data["policy"].astype(np.float32)
    numeric_cols = metadata["numeric_columns"]

    background_numeric = _prepare_samples(numeric, 30)
    sample_numeric = _prepare_samples(numeric, 80)
    time_mean = np.mean(time_tensor, axis=0)[None, ...]
    policy_mean = np.mean(policy_tensor, axis=0)[None, ...]

    def predict_fn(batch_numeric: np.ndarray) -> np.ndarray:
        bn = torch.from_numpy(batch_numeric.astype(np.float32)).to(DEVICE)
        bt = torch.from_numpy(np.repeat(time_mean, len(batch_numeric), axis=0)).to(DEVICE)
        bp = torch.from_numpy(np.repeat(policy_mean, len(batch_numeric), axis=0)).to(DEVICE)
        with torch.no_grad():
            outputs = model(bn, bt, bp, modality_dropout=False)
        return torch.sigmoid(outputs["logits"]).cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, background_numeric)
    shap_values = explainer.shap_values(sample_numeric, nsamples=100)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    feature_importance = dict(zip(numeric_cols, mean_abs.tolist()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = np.argsort(mean_abs)[::-1][:20]
    plt.figure(figsize=(8, 6))
    plt.barh([numeric_cols[i] for i in indices[::-1]], mean_abs[indices[::-1]])
    plt.title("Top Numeric SHAP Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_shap.png", dpi=300)
    plt.close()

    return feature_importance


def _integrated_gradients_policy(
    model,
    data: Dict[str, np.ndarray],
    steps: int = 50,
    sample_count: int = 64,
) -> np.ndarray:
    numeric = _prepare_samples(data["numeric"].astype(np.float32), sample_count)
    time_ts = _prepare_samples(data["time"].astype(np.float32), sample_count)
    policy = _prepare_samples(data["policy"].astype(np.float32), sample_count)
    baseline = np.zeros_like(policy)

    grads_accum = np.zeros(policy.shape[1], dtype=np.float64)

    for i in range(len(policy)):
        interpolated = [baseline[i] + (float(alpha) / steps) * (policy[i] - baseline[i]) for alpha in range(steps + 1)]
        interpolated = torch.from_numpy(np.stack(interpolated)).to(DEVICE)
        numeric_batch = torch.from_numpy(np.repeat(numeric[i][None, :], steps + 1, axis=0)).to(DEVICE)
        time_batch = torch.from_numpy(np.repeat(time_ts[i][None, :, :], steps + 1, axis=0)).to(DEVICE)
        interpolated.requires_grad_(True)
        outputs = model(numeric_batch, time_batch, interpolated, modality_dropout=False)
        preds = torch.sigmoid(outputs["logits"]).sum()
        grads = torch.autograd.grad(preds, interpolated)[0]
        avg_grads = grads.mean(dim=0).detach().cpu().numpy()
        grads_accum += avg_grads * (policy[i] - baseline[i])

    grads_accum /= len(policy)
    return grads_accum


def _summarize_top_features(numeric_imp: Dict[str, float], policy_ig: np.ndarray, metadata: Dict) -> Dict[str, list]:
    numeric_sorted = sorted(numeric_imp.items(), key=lambda kv: kv[1], reverse=True)
    policy_cols = metadata["policy_columns"]
    policy_scores = list(zip(policy_cols, policy_ig.tolist()))
    policy_sorted = sorted(policy_scores, key=lambda kv: abs(kv[1]), reverse=True)

    energy_features = [col for col in numeric_imp if "share" in col or "energy" in col]
    energy_sorted = sorted(((col, numeric_imp[col]) for col in energy_features), key=lambda kv: kv[1], reverse=True)

    return {
        "top_numeric": numeric_sorted[:15],
        "top_policy": policy_sorted[:15],
        "top_energy": energy_sorted[:15],
    }


def main() -> None:  # pragma: no cover - CLI utility
    metadata = _load_metadata()
    val_data = _load_npz(VAL_NPZ)
    model = _load_model(metadata)

    numeric_importance = _shap_numeric(model, val_data, metadata)
    policy_ig = _integrated_gradients_policy(model, val_data)
    summary = _summarize_top_features(numeric_importance, policy_ig, metadata)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "feature_importance.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Save policy importance bar chart
    policy_cols = metadata["policy_columns"]
    top_policy = summary["top_policy"][:20]
    plt.figure(figsize=(10, 6))
    plt.barh([name for name, _ in top_policy[::-1]], [abs(val) for _, val in top_policy[::-1]])
    plt.title("Top Policy IG Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "policy_ig.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
