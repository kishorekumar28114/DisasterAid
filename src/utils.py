"""
Common utilities for fine-grained disaster social media text classification.
"""

import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

# Actionable semantic categories for disaster-related social media posts
ALL_LABELS = [
    "casualty",
    "infrastructure_damage",
    "request_for_help",
    "rescue_or_aid",
    "shelter_or_supply",
    "situation_update",
]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_labels(label_str) -> list[int]:
    """Convert a pipe-separated label string to a binary vector representation."""
    if pd.isna(label_str) or not str(label_str).strip():
        return [0] * len(ALL_LABELS)
    labels = {label.strip() for label in str(label_str).split("|") if label.strip()}
    return [1 if label in labels else 0 for label in ALL_LABELS]


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute Subset Accuracy, F1 Macro, F1 Micro, and per-label F1 scores."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for idx, label in enumerate(ALL_LABELS):
        metrics[f"{label}_f1"] = float(per_label_f1[idx])
        
    return metrics


def resolve_device(device_name: str) -> str:
    """Resolve user-facing device option into PyTorch device name."""
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name
