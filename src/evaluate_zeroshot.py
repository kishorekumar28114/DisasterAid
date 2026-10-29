"""Zero-shot NLI evaluation for disaster text classification."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


try:
    from src.utils import ALL_LABELS, parse_labels, compute_classification_metrics
except ModuleNotFoundError:
    from utils import ALL_LABELS, parse_labels, compute_classification_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot classification evaluation")
    parser.add_argument("--data", default="data/processed/crisissense_converted.csv")
    parser.add_argument("--output-dir", default="results/zeroshot")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Number of test samples to evaluate. Use 0 for the full test split.")
    parser.add_argument("--model-name", default="facebook/bart-large-mnli")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda or a pipeline device id.")
    return parser.parse_args()


def load_test_data(data_path, sample_size=None):
    """Load and optionally subsample the test split."""
    df = pd.read_csv(data_path)
    test_df = df[df["split"] == "test"]

    if sample_size and len(test_df) > sample_size:
        test_df = test_df.sample(n=sample_size, random_state=42)

    texts = test_df["text_clean"].tolist()
    true_labels = [parse_labels(x) for x in test_df["multilabel"]]
    print(f"Loaded {len(texts)} test samples")
    return texts, true_labels


def resolve_pipeline_device(device):
    """Convert user-facing device options to Hugging Face pipeline ids."""
    if device == "auto":
        import torch

        return 0 if torch.cuda.is_available() else -1
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    return int(device)


def call_local_model_batch(texts, model_name="facebook/bart-large-mnli", batch_size=8, device="auto"):
    """Run zero-shot classification using a local BART-MNLI model."""
    from transformers import pipeline

    print(f"Loading model {model_name}...")
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=resolve_pipeline_device(device),
    )
    print("Model loaded. Starting predictions...")

    all_predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i : i + batch_size]
        try:
            results = classifier(batch_texts, ALL_LABELS, multi_label=True)
            for result in results:
                predictions = []
                for label in ALL_LABELS:
                    idx = result["labels"].index(label)
                    score = result["scores"][idx]
                    predictions.append(1 if score > 0.5 else 0)
                all_predictions.append(predictions)
        except Exception:
            for _ in batch_texts:
                all_predictions.append([0] * len(ALL_LABELS))

    return all_predictions, []




def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")
    texts, true_labels = load_test_data(args.data, args.sample_size)

    print("Running zero-shot with local NLI model...")
    predictions, errors = call_local_model_batch(
        texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    predictions = predictions[: len(texts)]

    print("\nComputing metrics...")
    metrics = compute_classification_metrics(true_labels, predictions)

    print("\n" + "=" * 50)
    print("Zero-Shot Classification Results:")
    print("=" * 50)
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"F1 Micro: {metrics['f1_micro']:.4f}")
    print("\nPer-label F1:")
    for label in ALL_LABELS:
        print(f"  {label}: {metrics[f'{label}_f1']:.4f}")

    if errors:
        print(f"\nErrors encountered: {len(errors)}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    results_df = pd.DataFrame({
        "text": texts,
        "true_labels": [
            "|".join([ALL_LABELS[i] for i, v in enumerate(l) if v])
            for l in true_labels
        ],
        "pred_labels": [
            "|".join([ALL_LABELS[i] for i, v in enumerate(l) if v])
            for l in predictions
        ],
    })
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Predictions saved: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
