#!/usr/bin/env python3
"""Counterfactual Planet baseline fine-tuning script."""

import os
import random
from inspect import signature
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


SEED = 42
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./models/flan-t5-owid-baseline"
MAX_INPUT_LENGTH = 128
MAX_LABEL_LENGTH = 16
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
DEVICE = 'cuda'

def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def prepare_data():
    # Data loading
    columns = [
        "country",
        "year",
        "co2",
        "gdp",
        "population",
        "coal_co2",
        "oil_co2",
        "gas_co2",
    ]
    df = pd.read_csv("owid-co2-data.csv", usecols=columns)
    df = df.dropna(subset=["co2"]).copy()

    # Label computation
    df = df.sort_values(["country", "year"])
    df["delta_co2"] = df.groupby("country")["co2"].diff()
    df = df.dropna(subset=["delta_co2"])
    df["label"] = np.where(df["delta_co2"] > 0, "increase", "decrease")
    df = df[(df["year"] >= 2000) & (df["year"] <= 2020)]
    sample_size = min(1000, len(df))
    if sample_size == 0:
        raise ValueError("No data available between 2000 and 2020 after preprocessing.")
    df = (
        df.sample(n=sample_size, random_state=SEED)
        .reset_index(drop=True)
        .copy()
    )

    # Text generation
    templates = [
        "In {year}, {country} implemented environmental policies affecting CO₂ emissions.",
        "In {year}, {country} changed its energy strategy influencing CO₂ emissions.",
        "In {year}, {country} launched climate initiatives targeting CO₂ output.",
        "In {year}, {country} reported policy shifts shaping CO₂ emissions trajectories.",
    ]
    rng = np.random.default_rng(SEED)
    df["text"] = [
        templates[rng.integers(len(templates))].format(
            year=int(row["year"]), country=row["country"]
        )
        for _, row in df.iterrows()
    ]
    df = df[["text", "label"]]

    print("Sample processed records:")
    print(df.head(5))

    return df


def main() -> None:
    set_seeds()
    ensure_output_dir(OUTPUT_DIR)

    df = prepare_data()

    stratify_labels = df["label"] if df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify_labels,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    def drop_index_column(dataset: Dataset) -> Dataset:
        if "__index_level_0__" in dataset.column_names:
            return dataset.remove_columns("__index_level_0__")
        return dataset

    train_dataset = drop_index_column(train_dataset)
    val_dataset = drop_index_column(val_dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    def preprocess_function(examples: Dict[str, list]) -> Dict[str, list]:
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LENGTH,
        )
        labels = tokenizer(
            examples["label"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LABEL_LENGTH,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
        decoded_preds = tokenizer.batch_decode(
            preds.tolist(), skip_special_tokens=True
        )
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        decoded_preds = [pred.strip().lower() for pred in decoded_preds]
        decoded_labels = [label.strip().lower() for label in decoded_labels]
        return {"accuracy": accuracy_score(decoded_labels, decoded_preds)}

    # Model fine-tuning
    init_params = signature(Seq2SeqTrainingArguments.__init__).parameters

    def arg_supported(name: str) -> bool:
        return name in init_params

    training_kwargs = dict(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.0,
        save_total_limit=2,
        num_train_epochs=NUM_EPOCHS,
        seed=SEED,
    )

    if arg_supported("evaluation_strategy"):
        training_kwargs["evaluation_strategy"] = "epoch"
    elif arg_supported("eval_strategy"):
        training_kwargs["eval_strategy"] = "epoch"
    elif arg_supported("evaluate_during_training"):
        training_kwargs["evaluate_during_training"] = True

    if arg_supported("predict_with_generate"):
        training_kwargs["predict_with_generate"] = True
    if arg_supported("logging_strategy"):
        training_kwargs["logging_strategy"] = "epoch"
    else:
        training_kwargs["logging_steps"] = 500
    if arg_supported("report_to"):
        training_kwargs["report_to"] = "none"

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluation and saving
    eval_metrics = trainer.evaluate()
    val_accuracy = eval_metrics.get("eval_accuracy", float("nan"))
    print(f"Validation accuracy: {val_accuracy:.4f}")

    sample_predictions = val_df.sample(
        n=min(10, len(val_df)),
        random_state=SEED,
    ).reset_index(drop=True)
    print("Sample validation predictions:")
    for _, row in sample_predictions.iterrows():
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_INPUT_LENGTH,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=MAX_LABEL_LENGTH,
            )
        prediction = tokenizer.decode(
            generated[0].cpu(),
            skip_special_tokens=True,
        )
        print(f"Text: {row['text']}")
        print(f"True label: {row['label']} | Predicted: {prediction}")
        print("-" * 60)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete – baseline model ready for milestone.")


if __name__ == "__main__":
    main()
