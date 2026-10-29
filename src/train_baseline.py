"""
TF-IDF + Logistic Regression baseline for six-label disaster semantic classification.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


try:
    from src.utils import ALL_LABELS
except ModuleNotFoundError:
    from utils import ALL_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF baseline for six-label disaster semantic classification."
    )
    parser.add_argument("--data", default="data/processed/crisissense_converted.csv",
                        help="Labeled CSV path.")
    parser.add_argument("--task", default="multilabel",
                        help="Task name for output files.")
    parser.add_argument("--model", choices=["tfidf_svm", "tfidf_logreg"],
                        default="tfidf_logreg", help="Baseline model.")
    parser.add_argument("--text-column", default="text_clean",
                        help="Text column name.")
    parser.add_argument("--split-column", default="split",
                        help="Split column name.")
    parser.add_argument("--label-column", default="multilabel",
                        help="Pipe-separated label column.")
    parser.add_argument("--output-dir", default="results/tfidf_baseline",
                        help="Directory for metrics and predictions.")
    return parser.parse_args()


def parse_labels(label_str: str) -> list[str]:
    """Parse a pipe-separated label string, keeping only known labels."""
    if not label_str:
        return []
    labels = [l.strip() for l in label_str.split("|") if l.strip()]
    return [l for l in labels if l in ALL_LABELS]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_rows(
    rows: list[dict[str, str]],
    split: str,
    text_column: str,
    label_column: str,
    split_column: str,
) -> list[tuple[str, list[str]]]:
    """Select and filter rows for a given split."""
    selected: list[tuple[str, list[str]]] = []
    for row in rows:
        text = (row.get(text_column) or "").strip()
        labels = parse_labels(row.get(label_column, ""))
        if row.get("is_duplicate", "false").lower() == "true":
            continue
        if row.get(split_column) != split:
            continue
        if not text or not labels:
            continue
        selected.append((text, labels))
    return selected


def make_pipeline(model_name: str) -> Pipeline:
    """Build a TF-IDF + classifier pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    if model_name == "tfidf_logreg":
        estimator = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
    else:
        from sklearn.svm import LinearSVC
        estimator = OneVsRestClassifier(LinearSVC())
    return Pipeline([("tfidf", vectorizer), ("clf", estimator)])


def compute_metrics(y_true: Any, y_pred: Any, labels: list[str]) -> dict[str, Any]:
    """Compute overall and per-label metrics."""
    subset_acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    label_metrics = {}
    for i, label in enumerate(labels):
        label_metrics[label] = {
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }

    return {
        "subset_accuracy": subset_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "label_metrics": label_metrics,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_path = root / args.data
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(data_path)
    train_data = select_rows(
        rows, "train", args.text_column, args.label_column, args.split_column
    )
    test_data = select_rows(
        rows, "test", args.text_column, args.label_column, args.split_column
    )

    if not train_data or not test_data:
        raise SystemExit("Need at least train and test rows with text and labels.")

    train_x = [t for t, _ in train_data]
    train_labels = [l for _, l in train_data]
    test_x = [t for t, _ in test_data]
    test_labels = [l for _, l in test_data]

    mlb = MultiLabelBinarizer(classes=ALL_LABELS)
    train_y = mlb.fit_transform(train_labels)
    test_y = mlb.transform(test_labels)

    print(f"Labels: {mlb.classes_.tolist()}")
    print(f"Train: {len(train_x)}, Test: {len(test_x)}")

    model = make_pipeline(args.model)
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)

    metrics = {
        "task": args.task,
        "model": args.model,
        "train_size": len(train_x),
        "test_size": len(test_x),
        "labels": mlb.classes_.tolist(),
        "metrics": compute_metrics(test_y, test_pred, mlb.classes_.tolist()),
    }

    metrics_path = output_dir / f"{args.task}_{args.model}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Subset Accuracy: {metrics['metrics']['subset_accuracy']:.4f}")
    print(f"  F1 Macro: {metrics['metrics']['f1_macro']:.4f}")
    print(f"  F1 Micro: {metrics['metrics']['f1_micro']:.4f}")
    print(f"\nPer-label metrics:")
    for label, lm in metrics["metrics"]["label_metrics"].items():
        print(f"  {label}: P={lm['precision']:.3f} R={lm['recall']:.3f} F1={lm['f1']:.3f}")
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
