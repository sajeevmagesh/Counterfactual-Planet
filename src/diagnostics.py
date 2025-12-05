"""Diagnostics, ablations, and interpretability utilities for the multimodal model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluate import compute_scores, load_checkpoint, load_metadata
from src.fusion_model import MultimodalFusionModel, load_config_from_metadata


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


FEATURE_DIR = Path("data_processed/feature_tensors")
TRAIN_NPZ = FEATURE_DIR / "train.npz"
VAL_NPZ = FEATURE_DIR / "val.npz"
RESULTS_DIR = Path("results")
DIAGNOSTICS_PATH = RESULTS_DIR / "diagnostics.json"
ATTENTION_PNG = RESULTS_DIR / "attention_heatmap.png"
SHAP_VALUES_PATH = RESULTS_DIR / "shap_numeric_values.npy"
SHAP_PNG = RESULTS_DIR / "shap_numeric.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _subset_indices(n: int, max_samples: Optional[int], seed: int) -> np.ndarray:
    if max_samples is None or max_samples >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    idx.sort()
    return idx


class DiagnosticDataset(Dataset):
    """Dataset wrapper with modality-masking modes for ablations."""

    def __init__(
        self,
        npz_path: Path,
        mode: str = "fused",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        with np.load(npz_path, allow_pickle=True) as data:
            total = len(data["cls"])
            idx = _subset_indices(total, max_samples, seed)
            self.numeric = torch.from_numpy(data["numeric"][idx]).float()
            self.policy = torch.from_numpy(data["policy"][idx]).float()
            self.time = torch.from_numpy(data["time"][idx]).float()
            self.cls = torch.from_numpy(data["cls"][idx]).float()
            self.reg = torch.from_numpy(data["reg"][idx]).float()
        self.mode = mode

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.cls)

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        numeric = self.numeric[idx].clone()
        policy = self.policy[idx].clone()
        time = self.time[idx].clone()
        if self.mode == "numeric_only":
            policy.zero_()
        elif self.mode == "policy_only":
            numeric.zero_()
            time.zero_()
        elif self.mode != "fused":
            raise ValueError(f"Unknown ablation mode: {self.mode}")
        return numeric, time, policy, self.cls[idx], self.reg[idx]


def _train_ablation_model(
    mode: str,
    metadata: Dict,
    epochs: int = 5,
    train_samples: Optional[int] = 4000,
    val_samples: Optional[int] = 2000,
) -> Dict[str, float]:
    logger.info("Running %s ablation (epochs=%d)...", mode, epochs)
    train_dataset = DiagnosticDataset(TRAIN_NPZ, mode=mode, max_samples=train_samples)
    val_dataset = DiagnosticDataset(VAL_NPZ, mode=mode, max_samples=val_samples, seed=7)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    config = load_config_from_metadata(metadata)
    model = MultimodalFusionModel(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.L1Loss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for numeric, time, policy, cls, reg in train_loader:
            numeric = numeric.to(DEVICE)
            time = time.to(DEVICE)
            policy = policy.to(DEVICE)
            cls = cls.to(DEVICE)
            reg = reg.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(numeric, time, policy)
            loss = criterion_cls(outputs["logits"], cls) + criterion_reg(outputs["regression"], reg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * numeric.size(0)
        logger.info("%s epoch %d loss %.4f", mode, epoch, running_loss / len(train_dataset))

    metrics = _evaluate_loader(model, val_loader, criterion_cls, criterion_reg)
    return metrics


def _evaluate_loader(model, loader, criterion_cls, criterion_reg) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    prob_list: List[torch.Tensor] = []
    cls_list: List[torch.Tensor] = []
    reg_pred_list: List[torch.Tensor] = []
    reg_true_list: List[torch.Tensor] = []
    with torch.no_grad():
        for numeric, time, policy, cls, reg in loader:
            numeric = numeric.to(DEVICE)
            time = time.to(DEVICE)
            policy = policy.to(DEVICE)
            cls = cls.to(DEVICE)
            reg = reg.to(DEVICE)
            outputs = model(numeric, time, policy, modality_dropout=False)
            loss = criterion_cls(outputs["logits"], cls) + criterion_reg(outputs["regression"], reg)
            total_loss += loss.item() * numeric.size(0)
            prob_list.append(outputs["prob"].cpu())
            cls_list.append(cls.cpu())
            reg_pred_list.append(outputs["regression"].cpu())
            reg_true_list.append(reg.cpu())
    prob = torch.cat(prob_list).numpy()
    cls_target = torch.cat(cls_list).numpy()
    reg_pred = torch.cat(reg_pred_list).numpy()
    reg_true = torch.cat(reg_true_list).numpy()
    metrics = compute_scores(prob, cls_target, reg_pred, reg_true)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def run_ablation_experiments(metadata: Dict) -> Dict[str, Dict[str, float]]:
    """Train/evaluate numeric-only, policy-only, and fused baselines."""

    modes = ["numeric_only", "policy_only", "fused"]
    results = {}
    for mode in modes:
        results[mode] = _train_ablation_model(mode, metadata)
    return results


def _load_npz_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def generate_attention_heatmap(model: MultimodalFusionModel, max_iso: int = 20) -> str:
    """Aggregate attention weights and save a heatmap PNG."""

    arrays = _load_npz_arrays(VAL_NPZ)
    n = len(arrays["cls"])
    batch_size = 256
    attn_values: List[float] = []
    iso_list: List[str] = []
    year_list: List[int] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        numeric = torch.from_numpy(arrays["numeric"][start:end]).float().to(DEVICE)
        time = torch.from_numpy(arrays["time"][start:end]).float().to(DEVICE)
        policy = torch.from_numpy(arrays["policy"][start:end]).float().to(DEVICE)
        with torch.no_grad():
            outputs = model(numeric, time, policy, modality_dropout=False)
            attn = outputs["attention"].cpu().numpy().reshape(-1)
        attn_values.extend(attn.tolist())
        iso_batch = arrays["iso"][start:end]
        year_batch = arrays["year"][start:end]
        iso_list.extend([str(x) for x in iso_batch])
        year_list.extend(year_batch.astype(int).tolist())

    df = pd.DataFrame({"iso_code": iso_list, "year": year_list, "attention": attn_values})
    top_iso = df.groupby("iso_code").size().sort_values(ascending=False).head(max_iso).index
    pivot = df[df["iso_code"].isin(top_iso)].pivot_table(
        index="iso_code", columns="year", values="attention", aggfunc="mean"
    ).fillna(0.0)
    pivot = pivot.sort_index()
    plt.figure(figsize=(16, 9))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title("Policy attention weights (validation)")
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(ATTENTION_PNG, dpi=300)
    plt.close()
    logger.info("Saved attention heatmap to %s", ATTENTION_PNG)
    return str(ATTENTION_PNG)


def compute_numeric_shap(model: MultimodalFusionModel, metadata: Dict) -> Dict[str, str]:
    """Compute SHAP values for numeric features using a kernel explainer."""

    arrays = _load_npz_arrays(VAL_NPZ)
    numeric = arrays["numeric"].astype(np.float32)
    time = arrays["time"].astype(np.float32)
    policy = arrays["policy"].astype(np.float32)
    rng = np.random.default_rng(0)
    background_idx = _subset_indices(len(numeric), 40, seed=0)
    sample_idx = _subset_indices(len(numeric), 80, seed=1)
    background_numeric = numeric[background_idx]
    sample_numeric = numeric[sample_idx]
    time_baseline = torch.from_numpy(np.mean(time[background_idx], axis=0, keepdims=True)).float()
    policy_baseline = torch.from_numpy(np.mean(policy[background_idx], axis=0, keepdims=True)).float()

    def model_fn(x_numeric: np.ndarray) -> np.ndarray:
        numeric_tensor = torch.from_numpy(x_numeric.astype(np.float32)).to(DEVICE)
        batch_size = numeric_tensor.size(0)
        time_tensor = time_baseline.repeat(batch_size, 1, 1).to(DEVICE)
        policy_tensor = policy_baseline.repeat(batch_size, 1).to(DEVICE)
        with torch.no_grad():
            outputs = model(numeric_tensor, time_tensor, policy_tensor, modality_dropout=False)
        return outputs["prob"].cpu().numpy()

    explainer = shap.KernelExplainer(model_fn, background_numeric)
    shap_values = explainer.shap_values(sample_numeric, nsamples=100)
    if isinstance(shap_values, list):
        shap_array = shap_values[0]
    else:
        shap_array = shap_values

    np.save(SHAP_VALUES_PATH, shap_array)
    shap.summary_plot(
        shap_array,
        sample_numeric,
        feature_names=metadata["numeric_columns"],
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=300)
    plt.close()
    logger.info("Saved SHAP outputs to %s and %s", SHAP_VALUES_PATH, SHAP_PNG)
    return {"values": str(SHAP_VALUES_PATH), "plot": str(SHAP_PNG)}


def counterfactual_remove_policy(
    model: MultimodalFusionModel,
    metadata: Dict,
    iso_code: Optional[str] = None,
    year: Optional[int] = None,
) -> Dict[str, float | str]:
    """Simulate removing policy embeddings for a specific country-year."""

    arrays = _load_npz_arrays(VAL_NPZ)
    iso_arr = arrays["iso"].astype(str)
    year_arr = arrays["year"].astype(int)
    numeric = arrays["numeric"].astype(np.float32)
    policy = arrays["policy"].astype(np.float32)
    time = arrays["time"].astype(np.float32)

    mask_candidates = np.where(policy.sum(axis=1) != 0)[0]
    target_idx = None
    if iso_code is not None and year is not None:
        matches = np.where((iso_arr == iso_code) & (year_arr == year))[0]
        if len(matches) > 0:
            target_idx = int(matches[0])
    if target_idx is None:
        if len(mask_candidates) == 0:
            raise ValueError("No samples with active policies available for counterfactual simulation")
        target_idx = int(mask_candidates[0])
        iso_code = iso_arr[target_idx]
        year = int(year_arr[target_idx])

    numeric_tensor = torch.from_numpy(numeric[target_idx]).float().to(DEVICE)
    time_tensor = torch.from_numpy(time[target_idx: target_idx + 1]).float().to(DEVICE)
    policy_tensor = torch.from_numpy(policy[target_idx]).float().to(DEVICE)

    numeric_cols = metadata["numeric_columns"]
    numeric_cf = numeric_tensor.clone()
    for feature in ("num_active_policies", "policy_mask"):
        if feature in numeric_cols:
            idx = numeric_cols.index(feature)
            numeric_cf[idx] = 0.0
    policy_cf = torch.zeros_like(policy_tensor)

    with torch.no_grad():
        base_out = model(
            numeric_tensor.unsqueeze(0),
            time_tensor,
            policy_tensor.unsqueeze(0),
            modality_dropout=False,
        )
        cf_out = model(
            numeric_cf.unsqueeze(0),
            time_tensor,
            policy_cf.unsqueeze(0),
            modality_dropout=False,
        )

    return {
        "iso_code": iso_code,
        "year": int(year),
        "baseline_delta_co2": float(base_out["regression"].item()),
        "counterfactual_delta_co2": float(cf_out["regression"].item()),
        "delta_difference": float(cf_out["regression"].item() - base_out["regression"].item()),
        "baseline_direction_prob": float(base_out["prob"].item()),
        "counterfactual_direction_prob": float(cf_out["prob"].item()),
    }


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnostics for multimodal fusion model")
    parser.add_argument("--iso", type=str, default=None, help="ISO code for counterfactual run")
    parser.add_argument("--year", type=int, default=None, help="Year for counterfactual run")
    parser.add_argument("--ablation_epochs", type=int, default=5, help="Epochs for ablation training")
    args = parser.parse_args(list(argv) if argv is not None else None)

    metadata = load_metadata()
    ablation_results = run_ablation_experiments(metadata)

    checkpoint = load_checkpoint()
    config = load_config_from_metadata(metadata)
    model = MultimodalFusionModel(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    attention_path = generate_attention_heatmap(model)
    shap_info = compute_numeric_shap(model, metadata)
    counterfactual = counterfactual_remove_policy(model, metadata, args.iso, args.year)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ablation": ablation_results,
        "attention_heatmap": attention_path,
        "shap": shap_info,
        "counterfactual": counterfactual,
    }
    with DIAGNOSTICS_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Diagnostics saved to %s", DIAGNOSTICS_PATH)


if __name__ == "__main__":
    main()
