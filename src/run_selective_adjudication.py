"""
Offline simulation for selective adjudication under a fixed review budget.

The experiment keeps the original evaluation split untouched, then:
  1. Carves a small labeled seed set out of the training split.
  2. Treats the remaining training rows as an unlabeled adjudication pool.
  3. Compares fixed-budget sample selection strategies.
  4. Reveals oracle labels from the pool and retrains the classifier.

This produces a reproducible experiment without a live annotation platform.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


try:
    from src.utils import ALL_LABELS
except ModuleNotFoundError:
    from utils import ALL_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline selective adjudication experiments."
    )
    parser.add_argument("--data", default="data/processed/crisissense_converted.csv")
    parser.add_argument("--external-data", default="data/processed/humaid_converted_test.csv")
    parser.add_argument("--output-dir", default="results/selective_adjudication")
    parser.add_argument("--text-column", default="text_clean")
    parser.add_argument("--label-column", default="multilabel")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--event-group-column", default="event_group")
    parser.add_argument("--doc-id-column", default="doc_id")
    parser.add_argument("--seed-ratio", type=float, default=0.10)
    parser.add_argument("--min-seed-per-label", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--strategies",
        default="random,uncertainty,disagreement,selective_adjudication",
        help="Comma-separated strategies.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_labels(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    labels = [label.strip() for label in str(value).split("|") if label.strip()]
    return [label for label in labels if label in ALL_LABELS]


def labels_to_vector(labels: list[str]) -> list[int]:
    label_set = set(labels)
    return [1 if label in label_set else 0 for label in ALL_LABELS]


def load_frame(
    path: Path,
    text_column: str,
    label_column: str,
    split_column: str | None = None,
    split_name: str | None = None,
) -> pd.DataFrame:
    """Load and filter a CSV dataframe."""
    df = pd.read_csv(path)
    if split_column and split_name and split_column in df.columns:
        df = df[df[split_column].fillna("").astype(str) == split_name]
    if "is_duplicate" in df.columns:
        df = df[df["is_duplicate"].fillna("false").astype(str).str.lower() != "true"]
    df[text_column] = df[text_column].fillna("").astype(str)
    df = df[df[text_column].str.strip() != ""].copy()
    df["__labels__"] = df[label_column].apply(parse_labels)
    df = df[df["__labels__"].map(bool)].copy()
    df["__label_vector__"] = df["__labels__"].apply(labels_to_vector)
    df["__primary_label__"] = df["__labels__"].apply(lambda labels: labels[0])
    return df.reset_index(drop=True)


def build_pipeline(analyzer: str) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline."""
    if analyzer == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=2, max_df=0.95, sublinear_tf=True,
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, sublinear_tf=True,
        )
    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def ensure_non_empty_prediction(binary_preds: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Assign the most probable label when a sample has no positive predictions."""
    empty_rows = binary_preds.sum(axis=1) == 0
    if np.any(empty_rows):
        best_idx = probs[empty_rows].argmax(axis=1)
        binary_preds[empty_rows, best_idx] = 1
    return binary_preds


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    row_sums = probs.sum(axis=1, keepdims=True)
    return np.divide(
        probs, np.where(row_sums == 0.0, 1.0, row_sums),
        out=np.zeros_like(probs), where=True,
    )


def predict_probabilities(model: Pipeline, texts: list[str]) -> np.ndarray:
    probs = model.predict_proba(texts)
    if isinstance(probs, list):
        probs = np.asarray(probs)
    return np.asarray(probs, dtype=float)


def predict_binary(probs: np.ndarray, threshold: float) -> np.ndarray:
    preds = (probs >= threshold).astype(int)
    return ensure_non_empty_prediction(preds, probs)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for idx, label in enumerate(ALL_LABELS):
        metrics[f"{label}_f1"] = float(per_label_f1[idx])
    return {key: float(value) for key, value in metrics.items()}


def evaluate_model(
    model: Pipeline, df: pd.DataFrame, text_column: str, threshold: float
) -> dict[str, float]:
    probs = predict_probabilities(model, df[text_column].tolist())
    preds = predict_binary(probs, threshold)
    y_true = np.asarray(df["__label_vector__"].tolist(), dtype=int)
    return compute_metrics(y_true, preds)


def split_seed_and_pool(
    train_df: pd.DataFrame, seed_ratio: float,
    min_seed_per_label: int, random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into a small seed set and a large adjudication pool."""
    rng = np.random.default_rng(random_seed)
    seed_indices: list[int] = []

    for label in ALL_LABELS:
        label_indices = train_df.index[train_df["__primary_label__"] == label].to_numpy()
        if len(label_indices) == 0:
            continue
        sample_size = max(min_seed_per_label, int(round(len(label_indices) * seed_ratio)))
        sample_size = min(sample_size, len(label_indices))
        sampled = rng.choice(label_indices, size=sample_size, replace=False)
        seed_indices.extend(sampled.tolist())

    seed_indices = sorted(set(seed_indices))
    seed_df = train_df.loc[seed_indices].copy().reset_index(drop=True)
    pool_df = train_df.drop(seed_indices).copy().reset_index(drop=True)
    return seed_df, pool_df


def fit_committee(
    labeled_df: pd.DataFrame, text_column: str
) -> tuple[Pipeline, Pipeline]:
    """Train a primary model and a committee model for disagreement estimation."""
    x_train = labeled_df[text_column].tolist()
    y_train = np.asarray(labeled_df["__label_vector__"].tolist(), dtype=int)
    primary = build_pipeline("word")
    committee = build_pipeline("char")
    primary.fit(x_train, y_train)
    committee.fit(x_train, y_train)
    return primary, committee


def compute_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Margin-based uncertainty: 1 - (top_prob - second_prob)."""
    normalized = normalize_probabilities(probs)
    sorted_probs = np.sort(normalized, axis=1)
    if sorted_probs.shape[1] == 1:
        return np.ones(sorted_probs.shape[0], dtype=float)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return 1.0 - margins


def compute_disagreement(
    primary_probs: np.ndarray, committee_probs: np.ndarray, threshold: float,
) -> np.ndarray:
    """Combined probability and label disagreement between two models."""
    primary_preds = predict_binary(primary_probs.copy(), threshold)
    committee_preds = predict_binary(committee_probs.copy(), threshold)
    prob_gap = np.mean(np.abs(primary_probs - committee_probs), axis=1)
    label_gap = np.mean(primary_preds != committee_preds, axis=1)
    return 0.7 * prob_gap + 0.3 * label_gap


def scale_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_val = float(values.min(initial=0.0))
    max_val = float(values.max(initial=0.0))
    if max_val - min_val < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)


def select_batch(
    strategy: str, pool_df: pd.DataFrame, scores: np.ndarray,
    batch_size: int, event_group_column: str, random_seed: int,
) -> list[int]:
    """Select samples to adjudicate based on strategy scores."""
    if len(pool_df) <= batch_size:
        return pool_df.index.tolist()

    if strategy == "random":
        rng = np.random.default_rng(random_seed)
        return rng.choice(pool_df.index.to_numpy(), size=batch_size, replace=False).tolist()

    ranked = pool_df.assign(__score__=scores).sort_values("__score__", ascending=False)

    if strategy != "selective_adjudication" or event_group_column not in ranked.columns:
        return ranked.head(batch_size).index.tolist()

    # Diversify across event groups for selective adjudication
    max_per_group = max(1, int(batch_size * 0.8))
    selected: list[int] = []
    group_counts: Counter[str] = Counter()

    for row_index, row in ranked.iterrows():
        group_name = str(row.get(event_group_column, "unknown"))
        if group_counts[group_name] >= max_per_group:
            continue
        selected.append(row_index)
        group_counts[group_name] += 1
        if len(selected) == batch_size:
            return selected

    for row_index in ranked.index.tolist():
        if row_index in selected:
            continue
        selected.append(row_index)
        if len(selected) == batch_size:
            break

    return selected


def build_strategy_scores(
    strategy: str, pool_df: pd.DataFrame, labeled_df: pd.DataFrame,
    primary_probs: np.ndarray, committee_probs: np.ndarray,
    threshold: float, event_group_column: str,
) -> np.ndarray:
    """Compute sample selection scores for the given strategy."""
    if strategy == "random":
        return np.zeros(len(pool_df), dtype=float)

    uncertainty = compute_uncertainty(primary_probs)
    if strategy == "uncertainty":
        return uncertainty

    disagreement = compute_disagreement(primary_probs, committee_probs, threshold)
    if strategy == "disagreement":
        return disagreement

    # Selective adjudication combines uncertainty and disagreement
    uncertainty = scale_scores(uncertainty)
    disagreement = scale_scores(disagreement)
    interaction = uncertainty * disagreement
    return 0.55 * uncertainty + 0.25 * disagreement + 0.20 * interaction


def run_strategy(
    strategy, seed_df, pool_df, eval_df, external_df,
    text_column, threshold, rounds, batch_size,
    event_group_column, doc_id_column, random_seed,
):
    """Execute one adjudication strategy across all rounds."""
    labeled_df = seed_df.copy()
    remaining_pool = pool_df.copy()
    round_records = []
    selection_records = []

    initial_eval_f1 = None
    initial_external_f1 = None

    for round_idx in range(rounds + 1):
        primary_model, committee_model = fit_committee(labeled_df, text_column)
        eval_metrics = evaluate_model(primary_model, eval_df, text_column, threshold)
        external_metrics = (
            evaluate_model(primary_model, external_df, text_column, threshold)
            if external_df is not None and not external_df.empty
            else None
        )

        if initial_eval_f1 is None:
            initial_eval_f1 = eval_metrics["f1_macro"]
        if external_metrics and initial_external_f1 is None:
            initial_external_f1 = external_metrics["f1_macro"]

        adjudicated_total = len(labeled_df) - len(seed_df)
        eval_gain = eval_metrics["f1_macro"] - initial_eval_f1
        eval_gain_per_100 = (eval_gain / adjudicated_total * 100.0) if adjudicated_total else 0.0

        round_record = {
            "strategy": strategy,
            "round": round_idx,
            "labeled_size": int(len(labeled_df)),
            "pool_size": int(len(remaining_pool)),
            "adjudicated_total": int(adjudicated_total),
            "eval_subset_accuracy": eval_metrics["subset_accuracy"],
            "eval_f1_macro": eval_metrics["f1_macro"],
            "eval_f1_micro": eval_metrics["f1_micro"],
            "eval_gain_macro": float(eval_gain),
            "eval_gain_macro_per_100": float(eval_gain_per_100),
        }
        if external_metrics:
            round_record["external_subset_accuracy"] = external_metrics["subset_accuracy"]
            round_record["external_f1_macro"] = external_metrics["f1_macro"]
            round_record["external_f1_micro"] = external_metrics["f1_micro"]
            if initial_external_f1 is not None:
                ext_gain = external_metrics["f1_macro"] - initial_external_f1
                round_record["external_gain_macro"] = float(ext_gain)
                round_record["external_gain_macro_per_100"] = (
                    float(ext_gain / adjudicated_total * 100.0) if adjudicated_total else 0.0
                )
        round_records.append(round_record)

        if round_idx == rounds or remaining_pool.empty:
            continue

        primary_probs = predict_probabilities(primary_model, remaining_pool[text_column].tolist())
        committee_probs = predict_probabilities(committee_model, remaining_pool[text_column].tolist())
        scores = build_strategy_scores(
            strategy, remaining_pool, labeled_df,
            primary_probs, committee_probs, threshold, event_group_column,
        )
        selected_indices = select_batch(
            strategy, remaining_pool, scores,
            batch_size, event_group_column, random_seed + round_idx,
        )
        selected_df = remaining_pool.loc[selected_indices].copy()
        labeled_df = pd.concat([labeled_df, selected_df], ignore_index=True)
        remaining_pool = remaining_pool.drop(index=selected_indices).reset_index(drop=True)

        for _, row in selected_df.iterrows():
            selection_records.append({
                "strategy": strategy,
                "round_selected_for": round_idx + 1,
                "doc_id": row.get(doc_id_column, ""),
                "event_group": row.get(event_group_column, ""),
                "labels": "|".join(row["__labels__"]),
                "primary_label": row["__primary_label__"],
            })

    return round_records, selection_records


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = root / args.data
    train_df = load_frame(
        data_path, args.text_column, args.label_column,
        args.split_column, args.train_split,
    )
    eval_df = load_frame(
        data_path, args.text_column, args.label_column,
        args.split_column, args.eval_split,
    )
    external_df = None
    if args.external_data:
        external_path = root / args.external_data
        if external_path.exists():
            external_df = load_frame(
                external_path, args.text_column, args.label_column,
            )

    seed_df, pool_df = split_seed_and_pool(
        train_df, args.seed_ratio, args.min_seed_per_label, args.seed,
    )

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    all_round_records = []
    all_selection_records = []

    summary = {
        "data": args.data,
        "external_data": args.external_data if external_df is not None else None,
        "seed_ratio": args.seed_ratio,
        "min_seed_per_label": args.min_seed_per_label,
        "rounds": args.rounds,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "strategies": strategies,
        "seed_size": int(len(seed_df)),
        "pool_size": int(len(pool_df)),
        "eval_size": int(len(eval_df)),
        "external_size": int(len(external_df)) if external_df is not None else 0,
        "results": {},
    }

    print(f"Train seed size: {len(seed_df)}")
    print(f"Adjudication pool size: {len(pool_df)}")
    print(f"Eval size: {len(eval_df)}")
    if external_df is not None:
        print(f"External eval size: {len(external_df)}")

    for idx, strategy in enumerate(strategies):
        print(f"\nRunning strategy: {strategy}")
        round_records, selection_records = run_strategy(
            strategy, seed_df, pool_df, eval_df, external_df,
            args.text_column, args.threshold, args.rounds, args.batch_size,
            args.event_group_column, args.doc_id_column, args.seed + idx * 1000,
        )
        all_round_records.extend(round_records)
        all_selection_records.extend(selection_records)
        summary["results"][strategy] = round_records

        final_round = round_records[-1]
        print(
            f"  Final eval Macro-F1: {final_round['eval_f1_macro']:.4f} "
            f"(gain {final_round['eval_gain_macro']:+.4f})"
        )
        if "external_f1_macro" in final_round:
            print(f"  Final external Macro-F1: {final_round['external_f1_macro']:.4f}")

    rounds_df = pd.DataFrame(all_round_records)
    rounds_df.to_csv(output_dir / "round_metrics.csv", index=False)

    selections_df = pd.DataFrame(all_selection_records)
    selections_df.to_csv(output_dir / "selected_samples.csv", index=False)

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=dict)

    print(f"\nRound metrics saved: {output_dir / 'round_metrics.csv'}")
    print(f"Selected sample log saved: {output_dir / 'selected_samples.csv'}")
    print(f"Summary saved: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
