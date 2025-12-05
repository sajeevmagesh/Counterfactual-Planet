"""Evaluation utilities for the multimodal fusion model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from torch.utils.data import DataLoader, Dataset

from src.fusion_model import MultimodalFusionModel, load_config_from_metadata
from src.metrics import compute_classification_metrics


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


FEATURE_DIR = Path("data_processed/feature_tensors")
VAL_NPZ = FEATURE_DIR / "val.npz"
METADATA_PATH = FEATURE_DIR / "metadata.json"
MODEL_PATH = Path("models/multimodal/fusion_model.pt")
RESULT_PATH = Path("results/multimodal_metrics.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureDataset(Dataset):
    """Simple Dataset wrapper around the feature tensors."""

    def __init__(self, npz_path: Path) -> None:
        data = np.load(npz_path)
        self.numeric = torch.from_numpy(data["numeric"]).float()
        self.policy = torch.from_numpy(data["policy"]).float()
        self.time = torch.from_numpy(data["time"]).float()
        self.cls = torch.from_numpy(data["cls"]).float()
        self.reg = torch.from_numpy(data["reg"]).float()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.cls)

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        return (
            self.numeric[idx],
            self.time[idx],
            self.policy[idx],
            self.cls[idx],
            self.reg[idx],
        )


def load_metadata() -> Dict:
    """Load metadata describing tensor shapes."""

    with METADATA_PATH.open() as f:
        return json.load(f)


def load_checkpoint() -> Dict:
    """Load trained model checkpoint."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    return torch.load(MODEL_PATH, map_location=DEVICE)


def evaluate_model() -> Dict[str, float]:
    """Run evaluation on the validation split and persist metrics."""

    metadata = load_metadata()
    checkpoint = load_checkpoint()
    config = load_config_from_metadata(metadata)
    model = MultimodalFusionModel(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = FeatureDataset(VAL_NPZ)
    loader = DataLoader(dataset, batch_size=128)

    logits_list = []
    prob_list = []
    cls_list = []
    reg_pred_list = []
    reg_true_list = []

    with torch.no_grad():
        for numeric, time, policy, cls, reg in loader:
            numeric = numeric.to(DEVICE)
            time = time.to(DEVICE)
            policy = policy.to(DEVICE)
            cls = cls.to(DEVICE).float()
            reg = reg.to(DEVICE)
            outputs = model(numeric, time, policy, modality_dropout=False)
            logits_list.append(outputs["logits"].cpu())
            prob_list.append(outputs["prob"].cpu())
            cls_list.append(cls.cpu())
            reg_pred_list.append(outputs["regression"].cpu())
            reg_true_list.append(reg.cpu())

    prob = torch.cat(prob_list).numpy().reshape(-1)
    cls_target = torch.cat(cls_list).numpy().reshape(-1)
    reg_pred = torch.cat(reg_pred_list).numpy().reshape(-1)
    reg_true = torch.cat(reg_true_list).numpy().reshape(-1)

    class_metrics = compute_classification_metrics(prob, cls_target)
    metrics = {
        **class_metrics,
        "mae": float(mean_absolute_error(reg_true, reg_pred)),
        "rmse": float(root_mean_squared_error(reg_true, reg_pred)),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "metrics": metrics,
        "n_samples": len(dataset),
        "model_path": str(MODEL_PATH),
    }
    with RESULT_PATH.open("w") as f:
        json.dump(result_payload, f, indent=2)
    logger.info("Validation metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    evaluate_model()
