"""
Cross-lingual or cross-dataset prediction and evaluation script.

Loads a trained model and evaluates it on a target dataset, optionally
computing metrics against gold labels.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


try:
    from src.utils import (
        ALL_LABELS,
        parse_labels,
        compute_classification_metrics,
        resolve_device,
    )
except ModuleNotFoundError:
    from utils import (
        ALL_LABELS,
        parse_labels,
        compute_classification_metrics,
        resolve_device,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-lingual or cross-dataset prediction and evaluation"
    )
    parser.add_argument("--model-path", default="results/bert_model/best_model")
    parser.add_argument("--data", required=True, help="Input CSV")
    parser.add_argument("--text-column", default="text_clean")
    parser.add_argument("--label-column", default=None,
                        help="Optional gold label column (pipe-separated).")
    parser.add_argument("--output-dir", default="results/cross_lingual")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    return parser.parse_args()




def load_model(model_path, device):
    """Load a trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model




def predict_batch(texts, tokenizer, model, batch_size, max_len, threshold, device):
    """Run batch prediction and return probabilities and binary predictions."""
    probabilities = []
    predictions = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu()
            preds = (probs >= threshold).int()

        probabilities.extend(probs.tolist())
        predictions.extend(preds.tolist())

    return probabilities, predictions


def analyze_predictions(predictions):
    """Summarize prediction statistics."""
    label_counts = {label: 0 for label in ALL_LABELS}
    multi_label_count = 0

    for pred in predictions:
        if sum(pred) > 1:
            multi_label_count += 1
        for idx, value in enumerate(pred):
            if value:
                label_counts[ALL_LABELS[idx]] += 1

    total = len(predictions)
    return {
        "total": total,
        "multi_label_ratio": multi_label_count / total if total else 0.0,
        "label_distribution": label_counts,
    }




def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.data)
    df = df[df[args.text_column].fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
    texts = df[args.text_column].astype(str).tolist()
    print(f"Loaded {len(texts)} texts")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    tokenizer, model = load_model(args.model_path, device)

    print("\nPredicting...")
    probabilities, predictions = predict_batch(
        texts, tokenizer, model,
        args.batch_size, args.max_len, args.threshold, device,
    )

    print("\nAnalyzing predictions...")
    analysis = analyze_predictions(predictions)
    analysis["model_path"] = args.model_path
    analysis["threshold"] = args.threshold
    analysis["device"] = device

    print("\n" + "=" * 50)
    print("Prediction Results:")
    print("=" * 50)
    print(f"Total samples: {analysis['total']}")
    print(f"Multi-label ratio: {analysis['multi_label_ratio']:.2%}")
    print("\nLabel distribution:")
    for label, count in sorted(analysis["label_distribution"].items(), key=lambda x: -x[1]):
        pct = (count / analysis["total"] * 100.0) if analysis["total"] else 0.0
        print(f"  {label}: {count} ({pct:.1f}%)")

    results_df = df.copy()
    results_df["predicted_labels"] = [
        "|".join(label for label, value in zip(ALL_LABELS, row) if value)
        for row in predictions
    ]
    for idx, label in enumerate(ALL_LABELS):
        results_df[f"{label}_prob"] = [row[idx] for row in probabilities]
        results_df[f"{label}_pred"] = [row[idx] for row in predictions]

    if args.label_column and args.label_column in df.columns:
        gold_labels = [parse_labels(value) for value in df[args.label_column]]
        metrics = compute_classification_metrics(gold_labels, predictions)
        analysis["metrics"] = metrics
        print("\nEvaluation Metrics:")
        print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 Micro: {metrics['f1_micro']:.4f}")
        for label in ALL_LABELS:
            print(f"  {label}: {metrics[f'{label}_f1']:.4f}")

    pred_path = output_dir / "predictions.csv"
    results_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved: {pred_path}")

    analysis_path = output_dir / "analysis.json"
    with analysis_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved: {analysis_path}")


if __name__ == "__main__":
    main()
