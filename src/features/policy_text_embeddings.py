from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

CPD_EXPANDED_PATH = Path("data_processed/cpd_expanded.parquet")
CPD_RAW_PATH = Path("data/climate_policy_database_policies_export.csv")
POLICY_EMBED_PATH = Path("data_processed/policy_embeddings.parquet")
EMBEDDINGS_DIR = Path("embeddings")

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32


def slugify(value: str) -> str:
    value = value or "unknown"
    keep = []
    for ch in value.lower():
        if ch.isalnum():
            keep.append(ch)
        else:
            keep.append("_")
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def load_policy_records() -> pd.DataFrame:
    if not CPD_EXPANDED_PATH.exists():
        raise FileNotFoundError(
            "Missing CPD expanded parquet. Run src.data.cpd to generate it first."
        )
    expanded = pd.read_parquet(CPD_EXPANDED_PATH)
    if expanded.empty:
        raise ValueError("cpd_expanded.parquet is empty; cannot build embeddings.")

    raw_cols = [
        "policy_id",
        "policy_name",
        "policy_description",
        "policy_type",
        "sector",
    ]
    raw = pd.read_csv(CPD_RAW_PATH, usecols=raw_cols, dtype=str)
    raw = raw.fillna("")
    expanded["policy_id"] = expanded["policy_id"].astype(str)
    raw["policy_id"] = raw["policy_id"].astype(str)

    df = expanded.merge(raw, on="policy_id", how="left", suffixes=("", "_raw"))
    text_parts = []
    for _, row in df.iterrows():
        description = row.get("policy_description", "")
        name = row.get("policy_name_raw", row.get("policy_name", ""))
        sector = row.get("sector_raw", row.get("sector", ""))
        policy_type = row.get("policy_type_raw", row.get("policy_type", ""))
        pieces = []
        if description:
            pieces.append(description.strip())
        if not pieces and name:
            pieces.append(name.strip())
        context_bits = ", ".join(
            bit.strip()
            for bit in [policy_type, sector]
            if isinstance(bit, str) and bit.strip()
        )
        if context_bits:
            pieces.append(f"Context: {context_bits}")
        text_parts.append(" ".join(pieces).strip())
    df["text"] = text_parts
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No policy descriptions available for embedding.")

    df["text_length"] = df["text"].str.len()
    return df[["iso_code", "year", "policy_id", "policy_type", "sector", "text", "text_length"]]


def encode_texts(texts: Iterable[str], model_name: str = DEFAULT_MODEL) -> Tuple[np.ndarray, int]:
    model_name = os.getenv("POLICY_EMBED_MODEL", model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    dim = embeddings.shape[1]
    return embeddings, dim


def aggregate_embeddings(df: pd.DataFrame, embeddings: np.ndarray, dim: int) -> pd.DataFrame:
    group = df.groupby(["iso_code", "year"])
    iso_codes: List[str] = []
    years: List[int] = []
    mean_vectors: List[np.ndarray] = []
    max_vectors: List[np.ndarray] = []
    longest_vectors: List[np.ndarray] = []

    for (iso_code, year), idx in tqdm(group.indices.items(), desc="Aggregating policies"):
        vecs = embeddings[idx]
        mean_vectors.append(vecs.mean(axis=0))
        max_vectors.append(vecs.max(axis=0))
        group_lengths = df.loc[idx, "text_length"].to_numpy()
        longest_idx = idx[group_lengths.argmax()]
        longest_vectors.append(embeddings[longest_idx])
        iso_codes.append(iso_code)
        years.append(year)

    meta_df = pd.DataFrame({"iso_code": iso_codes, "year": years})
    mean_df = pd.DataFrame(mean_vectors, columns=[f"policy_embed_mean_{i}" for i in range(dim)])
    max_df = pd.DataFrame(max_vectors, columns=[f"policy_embed_max_{i}" for i in range(dim)])
    longest_df = pd.DataFrame(
        longest_vectors, columns=[f"policy_embed_longest_{i}" for i in range(dim)]
    )

    aggregated = pd.concat([meta_df, mean_df, max_df, longest_df], axis=1)
    return aggregated


def policy_type_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["iso_code", "year", "policy_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    rename_map = {
        col: f"policy_type_count_{slugify(str(col))}" for col in counts.columns if col not in {"iso_code", "year"}
    }
    counts = counts.rename(columns=rename_map)
    return counts


def save_numpy_arrays(mean_vectors: np.ndarray, max_vectors: np.ndarray, longest_vectors: np.ndarray) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DIR / "policy_mean.npy", mean_vectors)
    np.save(EMBEDDINGS_DIR / "policy_max.npy", max_vectors)
    np.save(EMBEDDINGS_DIR / "policy_longest.npy", longest_vectors)


def build_policy_embeddings() -> pd.DataFrame:
    df = load_policy_records()
    embeddings, dim = encode_texts(df["text"])
    aggregated = aggregate_embeddings(df, embeddings, dim)
    counts = policy_type_counts(df)
    merged = aggregated.merge(counts, on=["iso_code", "year"], how="left")
    merged = merged.sort_values(["iso_code", "year"]).reset_index(drop=True)

    POLICY_EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(POLICY_EMBED_PATH, index=False)

    mean_columns = [col for col in merged.columns if col.startswith("policy_embed_mean_")]
    max_columns = [col for col in merged.columns if col.startswith("policy_embed_max_")]
    longest_columns = [col for col in merged.columns if col.startswith("policy_embed_longest_")]
    save_numpy_arrays(
        merged[mean_columns].to_numpy(),
        merged[max_columns].to_numpy(),
        merged[longest_columns].to_numpy(),
    )
    print(f"Saved policy embedding parquet to {POLICY_EMBED_PATH} with dim={dim}")
    return merged


if __name__ == "__main__":
    build_policy_embeddings()
