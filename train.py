import ast
import os
import random
from inspect import signature
from typing import Dict
from collections import defaultdict

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
LEARNING_RATE = 5e-5
DEVICE = 'cuda'

def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def generate_prompts():
    # Text generation
    df = pd.read_csv('dataset/data.csv')
    df['policies'] = df['policies'].apply(ast.literal_eval)

    prompt_list = []
    labels = []
    imbalance = 0
    for _, row in df.iterrows():
        country = row['country time'].split('_')[0]
        current_year = row['country time'].split('_')[1]
        prompt = """You are an expert in climate policy, and your job is to determine given a set of policies enacted by
                  a country in the past 10 years how the country's CO2 levels will shift."""
        prompt += "<Historical Policies>\n"

        nonzero = set()

        for i in range(10, 0, -1):
            if len(row['policies'][i]) == 0:
                continue

            prompt += f"In the year {int(current_year) - i} {country} enacted the following policies:"

            for stringency, policy in row['policies'][i]:
                if stringency > 0 or policy in nonzero:
                    prompt += f"A {policy} policy with stringency {stringency}\n"
                    nonzero.add(policy)

        prompt += "\n"
        prompt += "<Intervention>\n"

        prompt += f"{country} plans to implement the following policies in {int(current_year)}:\n"
        for stringency, policy in row['policies'][0]:
            prompt += f"A {policy} policy with stringency {stringency}\n"

        prompt += "\n"

        prompt += "Output format: a single word, either \"increasing\" or \"decreasing\"\n"
        prompt += "Predict the direction change in CO₂ emissions (MtCO₂). Output MUST match the format."
        prompt_list.append(prompt)

        if row['delta co2'] > 0:
            labels.append('increase')
            imbalance += 1
        else:
            labels.append('decrease')

    prepared_df = pd.DataFrame({
        "text": prompt_list,
        "label": labels
    })
    print(prepared_df.head())
    print(imbalance / len(prompt_list))
    return prepared_df


def main() -> None:
    set_seeds()
    ensure_output_dir(OUTPUT_DIR)

    df = generate_prompts()
    rng = np.random.default_rng(SEED)

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
