from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import warnings
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*logging_dir.*")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")

# Keep startup output clean on low-power machines.
hf_logging.set_verbosity_error()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / "bert_model").resolve()
RESULTS_DIR = (BASE_DIR / "results").resolve()
LOGS_DIR = (BASE_DIR / "logs").resolve()
SEED = 42


def load_dataset():
    fake = pd.read_csv(BASE_DIR / "dataset" / "Fake.csv")
    real = pd.read_csv(BASE_DIR / "dataset" / "True.csv")

    fake["labels"] = 0
    real["labels"] = 1

    data = pd.concat([fake, real], ignore_index=True)
    data = data[["text", "labels"]].dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]

    return data


def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=256)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


def build_training_args():
    base_kwargs = dict(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=3,  # Reduced from 4 (DistilBERT trains faster)
        learning_rate=2e-5,
        warmup_steps=500,  # Use warmup_steps instead of deprecated warmup_ratio
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        fp16=False,  # Keep as float32 for CPU stability
        dataloader_pin_memory=False,  # Fix: Disable pin_memory for CPU (no GPU)
        remove_unused_columns=False,  # Keep all columns
        optim="adamw_torch",  # Stable optimizer
    )
    try:
        return TrainingArguments(evaluation_strategy="epoch", **base_kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **base_kwargs)


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Transformers v5+ expects this env var instead of TrainingArguments.logging_dir.
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(LOGS_DIR))

    print("Loading dataset...")
    data = load_dataset()

    train_df, val_df = train_test_split(
        data,
        test_size=0.2,
        random_state=SEED,
        stratify=data["labels"],
    )
    print(f"Dataset loaded: {len(train_df)} train, {len(val_df)} val")

    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "Fake News", 1: "Real News"},
        label2id={"Fake News": 0, "Real News": 1},
    )

    # Calculate class weights for imbalanced data
    class_counts = train_df["labels"].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / (2 * class_counts.get(0, 1)), len(train_df) / (2 * class_counts.get(1, 1))],
        dtype=torch.float,
    )

    print("Starting training...")
    trainer = WeightedTrainer(
        model=model,
        args=build_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    train_output = trainer.train()
    metrics = trainer.evaluate()

    print("Saving model...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Training loss: {train_output.training_loss:.4f}")
    print(
        "Validation metrics:",
        {k: round(v, 4) for k, v in metrics.items() if k.startswith("eval_")},
    )
    print("="*60)


if __name__ == "__main__":
    main()
