"""
Transformer fine-tuning script for six-label disaster semantic classification.

Supports BERT-base, multilingual BERT, and other Hugging Face transformer models.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from src.utils import (
        ALL_LABELS,
        parse_labels,
        set_seed,
        compute_classification_metrics,
    )
except ModuleNotFoundError:
    from utils import (
        ALL_LABELS,
        parse_labels,
        set_seed,
        compute_classification_metrics,
    )


class DisasterDataset(Dataset):
    """PyTorch dataset for disaster text classification."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.FloatTensor(self.labels[idx]),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Transformer for six-label disaster semantic classification."
    )
    parser.add_argument("--data", default="data/processed/crisissense_converted.csv")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="results/bert_model")
    parser.add_argument("--text-column", default="text_clean")
    parser.add_argument("--label-column", default="multilabel",
                        help="Pipe-separated label column.")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev",
                        help="Validation split used for checkpoint selection.")
    parser.add_argument("--test-split", default="test",
                        help="Held-out final evaluation split.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None,
                        help="Optional local Hugging Face cache directory.")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save evaluation predictions as CSV.")
    return parser.parse_args()




def select_split(df, split_name, text_column, label_column):
    """Filter and prepare a split from the dataframe."""
    split_df = df[df["__split__"] == split_name].copy()
    if "is_duplicate" in split_df.columns:
        split_df = split_df[
            split_df["is_duplicate"].fillna("false").astype(str).str.lower() != "true"
        ]
    split_df[text_column] = split_df[text_column].fillna("").astype(str)
    split_df = split_df[split_df[text_column].str.strip() != ""]
    split_df["__labels__"] = split_df[label_column].apply(parse_labels)
    split_df = split_df[split_df["__labels__"].map(sum) > 0].reset_index(drop=True)
    return split_df


def load_data(args):
    """Load and split the dataset."""
    df = pd.read_csv(args.data)
    df["__split__"] = df[args.split_column].fillna("").astype(str)
    train_df = select_split(df, args.train_split, args.text_column, args.label_column)
    eval_df = select_split(df, args.eval_split, args.text_column, args.label_column)
    test_df = select_split(df, args.test_split, args.text_column, args.label_column)

    if train_df.empty or eval_df.empty or test_df.empty:
        raise SystemExit(
            f"Need non-empty {args.train_split}, {args.eval_split}, and "
            f"{args.test_split} splits."
        )

    print(f"Train: {len(train_df)} ({args.train_split}), "
          f"Validation: {len(eval_df)} ({args.eval_split}), "
          f"Test: {len(test_df)} ({args.test_split})")
    return train_df, eval_df, test_df


def build_compute_metrics(threshold):
    """Build a metrics function compatible with the Trainer API."""
    def compute_metrics(pred):
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        labels = pred.label_ids
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        return compute_classification_metrics(labels, preds)

    return compute_metrics


def save_predictions(trainer, dataset, split_df, output_dir, threshold, split_name):
    """Save per-sample predictions and probabilities."""
    prediction_output = trainer.predict(dataset)
    logits = (
        prediction_output.predictions[0]
        if isinstance(prediction_output.predictions, tuple)
        else prediction_output.predictions
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    pred_df = split_df.copy()
    pred_df["predicted_labels"] = [
        "|".join(label for label, value in zip(ALL_LABELS, row) if value) for row in preds
    ]
    for idx, label in enumerate(ALL_LABELS):
        pred_df[f"{label}_prob"] = probs[:, idx]
        pred_df[f"{label}_pred"] = preds[:, idx]

    pred_path = output_dir / f"{split_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    train_df, eval_df, test_df = load_data(args)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ALL_LABELS),
        problem_type="multi_label_classification",
        cache_dir=args.cache_dir,
    )

    train_dataset = DisasterDataset(
        train_df[args.text_column].tolist(),
        train_df["__labels__"].tolist(),
        tokenizer,
        args.max_len,
    )
    eval_dataset = DisasterDataset(
        eval_df[args.text_column].tolist(),
        eval_df["__labels__"].tolist(),
        tokenizer,
        args.max_len,
    )
    test_dataset = DisasterDataset(
        test_df[args.text_column].tolist(),
        test_df["__labels__"].tolist(),
        tokenizer,
        args.max_len,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(args.threshold),
    )

    print("Training...")
    trainer.train()

    print("Evaluating validation split...")
    validation_results = trainer.evaluate(
        eval_dataset=eval_dataset,
        metric_key_prefix="validation",
    )
    print("\nValidation Results:")
    print(f"  Subset Accuracy: {validation_results['validation_subset_accuracy']:.4f}")
    print(f"  F1 Macro: {validation_results['validation_f1_macro']:.4f}")
    print(f"  F1 Micro: {validation_results['validation_f1_micro']:.4f}")
    print("\nPer-label F1:")
    for label in ALL_LABELS:
        print(f"  {label}: {validation_results[f'validation_{label}_f1']:.4f}")

    print("\nEvaluating held-out test split...")
    test_results = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )
    print("\nTest Results:")
    print(f"  Subset Accuracy: {test_results['test_subset_accuracy']:.4f}")
    print(f"  F1 Macro: {test_results['test_f1_macro']:.4f}")
    print(f"  F1 Micro: {test_results['test_f1_micro']:.4f}")
    print("\nPer-label F1:")
    for label in ALL_LABELS:
        print(f"  {label}: {test_results[f'test_{label}_f1']:.4f}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_split": args.eval_split,
                "test_split": args.test_split,
                **validation_results,
                **test_results,
            },
            f,
            indent=2,
        )
    print(f"\nMetrics saved: {metrics_path}")

    run_config = {
        "data": args.data,
        "model_name": args.model_name,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "split_column": args.split_column,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "test_split": args.test_split,
        "model_selection_metric": "validation_f1_macro",
        "threshold": args.threshold,
        "seed": args.seed,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    model.save_pretrained(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")
    print(f"Model saved: {output_dir / 'best_model'}")

    if args.save_predictions:
        save_predictions(trainer, eval_dataset, eval_df, output_dir, args.threshold, args.eval_split)
        save_predictions(trainer, test_dataset, test_df, output_dir, args.threshold, args.test_split)


if __name__ == "__main__":
    main()
