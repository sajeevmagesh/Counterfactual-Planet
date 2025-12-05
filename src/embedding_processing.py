"""Policy embedding processing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RAW_CPD_PATH = Path("data/climate_policy_database_policies_export.csv")
CPD_EXPANDED_PATH = Path("data_processed/cpd_expanded.parquet")
OUTPUT_PARQUET = Path("data_processed/policy_embeddings.parquet")
EMBEDDING_NPY = Path("embeddings/policy_embeddings.npy")
INDEX_PARQUET = Path("embeddings/policy_embeddings_index.parquet")
MASTER_PANEL_PATH = Path("data_processed/master_panel.parquet")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32


def load_cpd_raw(raw_path: Path = RAW_CPD_PATH) -> pd.DataFrame:
    """Load raw CPD dataset with textual metadata."""

    if not raw_path.exists():
        raise FileNotFoundError(f"CPD raw file missing at {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d raw CPD rows", len(df))
    keep_cols = [
        "policy_id",
        "policy_name",
        "policy_description",
        "policy_type",
        "sector",
    ]
    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CPD raw file: {missing_cols}")
    return df[keep_cols].fillna("")


def build_policy_text(row: pd.Series) -> str:
    """Create a descriptive text for embedding."""

    parts = []
    if row["policy_description"].strip():
        parts.append(row["policy_description"].strip())
    elif row["policy_name"].strip():
        parts.append(row["policy_name"].strip())
    context = ", ".join(
        bit.strip()
        for bit in [row["policy_type"], row["sector"]]
        if isinstance(bit, str) and bit.strip()
    )
    if context:
        parts.append(f"Context: {context}")
    return " ".join(parts).strip()


def encode_policies(texts: Iterable[str]) -> Tuple[np.ndarray, int]:
    """Encode policy texts and L2-normalize."""

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        list(texts),
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings.astype(np.float32), embeddings.shape[1]


def load_cpd_expanded(path: Path = CPD_EXPANDED_PATH) -> pd.DataFrame:
    """Load expanded CPD file with iso/year rows."""

    if not path.exists():
        raise FileNotFoundError(f"cpd_expanded.parquet missing at {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded %d expanded policy-year rows", len(df))
    expected = {"iso_code", "year", "policy_id"}
    if not expected.issubset(df.columns):
        raise ValueError(f"cpd_expanded missing columns {expected - set(df.columns)}")
    return df


def aggregate_embeddings(
    expanded: pd.DataFrame,
    policy_embeddings: pd.DataFrame,
    embed_dim: int,
) -> pd.DataFrame:
    """Average policy embeddings per (iso_code, year)."""

    merged = expanded.merge(policy_embeddings, on="policy_id", how="left")
    embed_cols = [f"policy_embed_{i}" for i in range(embed_dim)]
    merged[embed_cols] = merged[embed_cols].fillna(0.0)
    grouped = merged.groupby(["iso_code", "year"], as_index=False)

    agg_vectors = grouped[embed_cols].mean()
    counts = grouped["policy_id"].count().rename(columns={"policy_id": "num_active_policies"})
    result = agg_vectors.merge(counts, on=["iso_code", "year"], how="left")
    result["policy_mask"] = (result["num_active_policies"] > 0).astype(np.float32)
    result["num_active_policies"] = result["num_active_policies"].astype(np.float32)
    # ensure float32 dtype
    result[embed_cols] = result[embed_cols].astype(np.float32)
    return result


def ensure_all_years(
    aggregated: pd.DataFrame,
    reference_panel: pd.DataFrame,
    embed_dim: int,
) -> pd.DataFrame:
    """Align aggregated embeddings with reference country-years."""

    embed_cols = [f"policy_embed_{i}" for i in range(embed_dim)]
    aligned = reference_panel[["iso_code", "year"]].drop_duplicates().merge(
        aggregated,
        on=["iso_code", "year"],
        how="left",
    )
    aligned[embed_cols] = aligned[embed_cols].fillna(0.0)
    aligned["num_active_policies"] = aligned["num_active_policies"].fillna(0.0)
    aligned["policy_mask"] = aligned["policy_mask"].fillna(0.0)
    aligned[embed_cols] = aligned[embed_cols].astype(np.float32)
    aligned["num_active_policies"] = aligned["num_active_policies"].astype(np.float32)
    aligned["policy_mask"] = aligned["policy_mask"].astype(np.float32)
    return aligned


def build_policy_embeddings() -> None:
    """Main entry point for processing policy embeddings."""

    raw = load_cpd_raw()
    raw["text"] = raw.apply(build_policy_text, axis=1)
    raw = raw[raw["text"].str.len() > 0].reset_index(drop=True)
    embeddings, dim = encode_policies(raw["text"])
    embed_cols = [f"policy_embed_{i}" for i in range(dim)]
    policy_embeddings = pd.DataFrame(embeddings, columns=embed_cols)
    policy_embeddings.insert(0, "policy_id", raw["policy_id"].astype(str).values)

    expanded = load_cpd_expanded()
    expanded["policy_id"] = expanded["policy_id"].astype(str)
    aggregated = aggregate_embeddings(expanded, policy_embeddings, dim)
    if not MASTER_PANEL_PATH.exists():
        raise FileNotFoundError(f"Master panel required for alignment at {MASTER_PANEL_PATH}")
    reference_panel = pd.read_parquet(MASTER_PANEL_PATH)
    aligned = ensure_all_years(aggregated, reference_panel, dim)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Saved policy embeddings to %s", OUTPUT_PARQUET)

    EMBEDDING_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_NPY, aligned[embed_cols].to_numpy(np.float32))
    aligned[["iso_code", "year", "policy_mask", "num_active_policies"]].to_parquet(
        INDEX_PARQUET, index=False
    )
    logger.info("Saved numpy embeddings and index metadata")


if __name__ == "__main__":
    build_policy_embeddings()
