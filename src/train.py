"""Training script for the multimodal fusion model."""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.fusion_model import FusionConfig, MultimodalFusionModel, load_config_from_metadata
from src.metrics import compute_classification_metrics


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_DIR = Path("data_processed/feature_tensors")
TRAIN_NPZ = FEATURE_DIR / "train.npz"
VAL_NPZ = FEATURE_DIR / "val.npz"
METADATA_PATH = FEATURE_DIR / "metadata.json"
MODEL_DIR = Path("models/multimodal")
MODEL_PATH = MODEL_DIR / "fusion_model.pt"
HISTORY_PATH = MODEL_DIR / "training_history.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
EPOCHS = int(os.getenv("TRAIN_EPOCHS", "50"))
BATCH_SIZE = 64
CLIP_NORM = 1.0

# ---- multitask loss weights (classification > regression) ----
CLS_WEIGHT = 10.0
REG_SCALE = 0.01

# Debug flag to run classification-only training when needed
# Toggle via env var to disable regression head for debugging (1=True)
CLASSIFICATION_ONLY_DEBUG = bool(int(os.getenv("CLASSIFICATION_ONLY_DEBUG", "0")))
FREEZE_REGRESSION = True


class TensorDataset(Dataset):
    def __init__(self, npz_path: Path) -> None:
        data = np.load(npz_path)
        self.numeric = torch.from_numpy(data["numeric"]).float()
        self.policy = torch.from_numpy(data["policy"]).float()
        self.time = torch.from_numpy(data["time"]).float()
        self.cls = torch.from_numpy(data["cls"]).float().reshape(-1)
        self.reg = torch.from_numpy(data["reg"]).float()

    def __len__(self) -> int:
        return len(self.cls)

    def __getitem__(self, idx: int):
        return (
            self.numeric[idx],
            self.time[idx],
            self.policy[idx],
            self.cls[idx],
            self.reg[idx],
        )


def load_metadata() -> dict:
    with METADATA_PATH.open() as f:
        return json.load(f)



def train_one_epoch(model, loader, optimizer, scaler, criterion_cls, criterion_reg, epoch: int):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if DEVICE.type == "cuda"
        else nullcontext()
    )

    for numeric, time, policy, cls, reg in loader:
        numeric = numeric.to(DEVICE)
        time = time.to(DEVICE)
        policy = policy.to(DEVICE)
        cls = cls.to(DEVICE).float()
        reg = reg.to(DEVICE)

        optimizer.zero_grad()

        with autocast_ctx:
            outputs = model(numeric, time, policy)
            loss_cls = criterion_cls(outputs["logits"], cls)
            if CLASSIFICATION_ONLY_DEBUG:
                loss_reg = torch.zeros(1, device=DEVICE)
                loss = loss_cls
            else:
                loss_reg = criterion_reg(outputs["regression"], reg) * REG_SCALE
                if epoch < 5:
                    loss = loss_cls
                else:
                    loss = CLS_WEIGHT * loss_cls + loss_reg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        bs = numeric.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_cls.item() * bs
        total_reg_loss += loss_reg.item() * bs

    n = len(loader.dataset)
    return (
        total_loss / n,
        total_cls_loss / n,
        total_reg_loss / n,
    )


def evaluate(model, loader, criterion_cls, criterion_reg):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    all_probs = []
    all_targets = []
    all_preds = []
    all_reg = []
    all_reg_targets = []
    all_logits = []

    with torch.no_grad():
        for numeric, time, policy, cls, reg in loader:
            numeric = numeric.to(DEVICE)
            time = time.to(DEVICE)
            policy = policy.to(DEVICE)
            cls = cls.to(DEVICE).float()
            reg_target = reg.to(DEVICE)

            outputs = model(numeric, time, policy, modality_dropout=False)
            loss_cls = criterion_cls(outputs["logits"], cls)
            if CLASSIFICATION_ONLY_DEBUG:
                loss_reg = torch.zeros(1, device=DEVICE)
                loss = loss_cls
            else:
                loss_reg = criterion_reg(outputs["regression"], reg_target) * REG_SCALE
                loss = CLS_WEIGHT * loss_cls + loss_reg

            bs = numeric.size(0)
            total_loss += loss.item() * bs
            total_cls_loss += loss_cls.item() * bs
            total_reg_loss += loss_reg.item() * bs

            probs = outputs["prob"].cpu()
            preds = (probs > 0.5).float()
            all_probs.append(probs)
            all_targets.append(cls.cpu())
            all_preds.append(preds)
            all_reg.append(outputs["regression"].cpu())
            all_reg_targets.append(reg.cpu())
            all_logits.append(outputs["logits"].detach().cpu())

    prob = torch.cat(all_probs).numpy().reshape(-1)
    target = torch.cat(all_targets).numpy().reshape(-1)
    pred = torch.cat(all_preds).numpy().reshape(-1)
    reg_pred = torch.cat(all_reg).numpy().reshape(-1)
    reg_true = torch.cat(all_reg_targets).numpy().reshape(-1)
    logits = torch.cat(all_logits).numpy().reshape(-1)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "reg_loss": total_reg_loss / n,
        "prob": prob,
        "target": target,
        "reg_pred": reg_pred,
        "reg_true": reg_true,
        "pred": pred,
        "logits": logits,
    }


def train():
    metadata = load_metadata()

    # just log class balance for debugging
    train_npz = np.load(TRAIN_NPZ)
    cls_labels = train_npz["cls"]
    pos_frac = float(cls_labels.mean())
    neg_frac = 1.0 - pos_frac
    pos_count = max(float((cls_labels == 1).sum()), 1.0)
    neg_count = max(float((cls_labels == 0).sum()), 1.0)
    pos_weight_value = neg_count / pos_count
    logger.info(
        "Class balance: pos_frac=%.4f neg_frac=%.4f",
        pos_frac,
        neg_frac,
    )
    logger.info("Classification-only debug mode: %s", CLASSIFICATION_ONLY_DEBUG)
    logger.info("Using pos_weight=%.4f", pos_weight_value)

    train_dataset = TensorDataset(TRAIN_NPZ)
    val_dataset = TensorDataset(VAL_NPZ)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    config = load_config_from_metadata(metadata)
    model = MultimodalFusionModel(config).to(DEVICE)
    if FREEZE_REGRESSION:
        for param in model.regressor.parameters():
            param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32, device=DEVICE)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_reg = nn.L1Loss()

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion_cls, criterion_reg, epoch
        )
        val_metrics = evaluate(model, val_loader, criterion_cls, criterion_reg)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_cls_loss": train_cls_loss,
                "train_reg_loss": train_reg_loss,
                "val_loss": val_metrics["loss"],
                "val_cls_loss": val_metrics["cls_loss"],
                "val_reg_loss": val_metrics["reg_loss"],
            }
        )

        logger.info(
            "Epoch %d: "
            "train=%.4f (cls=%.4f reg=%.4f)  "
            "val=%.4f (cls=%.4f reg=%.4f)",
            epoch,
            train_loss,
            train_cls_loss,
            train_reg_loss,
            val_metrics["loss"],
            val_metrics["cls_loss"],
            val_metrics["reg_loss"],
        )

        class_metrics = compute_classification_metrics(val_metrics["prob"], val_metrics["target"])
        logger.info(
            "Val metrics: acc=%.4f f1=%.4f auc=%.4f pos_rate=%.4f pred_pos_rate=%.4f",
            class_metrics["accuracy"],
            class_metrics["f1"],
            class_metrics["roc_auc"],
            class_metrics["pos_rate"],
            class_metrics["pred_pos_rate"],
        )
        logger.info(
            "Val logits distribution: mean=%.4f std=%.4f",
            float(val_metrics["logits"].mean()),
            float(val_metrics["logits"].std()),
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "config": config.__dict__},
                MODEL_PATH,
            )

    with HISTORY_PATH.open("w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete")


if __name__ == "__main__":
    train()
