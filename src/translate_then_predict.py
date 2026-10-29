"""Evaluate an English classifier on pre-translated Chinese disaster posts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate translation-bridge predictions with an English classifier."
    )
    parser.add_argument("--data", default="data/chinese_gold_177_translated.csv")
    parser.add_argument("--translation-column", default="text_en_translated")
    parser.add_argument("--label-column", default="final_multilabel_gold")
    parser.add_argument("--model-path", default="results/bert_model/best_model")
    parser.add_argument("--output-dir", default="results/chinese_translate_bert")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser.parse_args()






def iter_batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_input(path: Path, translation_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in (translation_column, label_column) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df[translation_column] = df[translation_column].fillna("").astype(str)
    df = df[df[translation_column].str.strip() != ""].reset_index(drop=True)
    return df


def predict(
    texts: list[str],
    model_path: str,
    batch_size: int,
    max_len: int,
    threshold: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    probabilities = []
    for batch_texts in tqdm(list(iter_batches(texts, batch_size)), desc="Predicting"):
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits).cpu().numpy()
        probabilities.append(probs)

    probs = np.vstack(probabilities)
    preds = (probs >= threshold).astype(int)
    return probs, preds




def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(Path(args.data), args.translation_column, args.label_column)
    device = resolve_device(args.device)

    texts = df[args.translation_column].tolist()
    gold = np.array([parse_labels(value) for value in df[args.label_column]])
    probabilities, predictions = predict(
        texts=texts,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        threshold=args.threshold,
        device=device,
    )

    metrics = compute_classification_metrics(gold, predictions)
    result = {
        "total": int(len(df)),
        "input_data": args.data,
        "classifier_model_path": args.model_path,
        "translation_column": args.translation_column,
        "label_column": args.label_column,
        "threshold": args.threshold,
        "metrics": metrics,
    }

    results_df = df.copy()
    results_df["predicted_labels"] = [
        "|".join(label for label, value in zip(ALL_LABELS, row) if value)
        for row in predictions
    ]
    for idx, label in enumerate(ALL_LABELS):
        results_df[f"{label}_prob"] = probabilities[:, idx]
        results_df[f"{label}_pred"] = predictions[:, idx]

    results_df.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
