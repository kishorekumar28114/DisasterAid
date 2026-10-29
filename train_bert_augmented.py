import os
import json
import random
import re
import string
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset

REPO = "/content/FGSC"
CSV = f"{REPO}/data/processed/crisissense_converted.csv"
OUT = f"{REPO}/results_ours/bert_augmented"
os.makedirs(OUT, exist_ok=True)

LABELS = [
    "casualty",
    "infrastructure_damage",
    "request_for_help",
    "rescue_or_aid",
    "shelter_or_supply",
    "situation_update",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Reproducible, conservative character-level noise.
def add_typo(text, rng):
    words = text.split()
    candidates = [
        i for i, word in enumerate(words)
        if len(word) >= 5 and any(c.isalpha() for c in word)
    ]

    if not candidates:
        return text

    i = rng.choice(candidates)
    word = words[i]
    positions = [
        j for j in range(len(word))
        if word[j].isalpha()
    ]

    if not positions:
        return text

    j = rng.choice(positions)
    operation = rng.choice(["delete", "swap", "replace"])

    if operation == "delete" and len(word) > 4:
        word = word[:j] + word[j + 1:]
    elif operation == "swap" and j < len(word) - 1:
        word = (
            word[:j] + word[j + 1] + word[j]
            + word[j + 2:]
        )
    elif operation == "replace":
        alphabet = string.ascii_lowercase
        replacement = rng.choice(alphabet)
        word = word[:j] + replacement + word[j + 1:]

    words[i] = word
    return " ".join(words)


df = pd.read_csv(CSV)

# Match the original script's filtering and target preparation.
if "is_duplicate" in df.columns:
    dup = (
        df["is_duplicate"].fillna("false")
        .astype(str).str.lower().eq("true")
    )
    df = df[~dup].copy()

df["text_clean"] = (
    df["text_clean"].fillna("").astype(str).str.strip()
)

df["target"] = df["multilabel"]
df = df[
    df["target"].isin(LABELS) & df["text_clean"].ne("")
].copy()

splits = {
    name: df[df["split"] == name].reset_index(drop=True)
    for name in ["train", "dev", "test"]
}

assert all(len(frame) > 0 for frame in splits.values())
assert set(splits["train"]["target"]) == set(LABELS)

# Keep all original training examples and add one noisy copy per example.
rng = random.Random(SEED)
train_original = splits["train"].copy()
train_noisy = train_original.copy()
train_noisy["text_clean"] = train_noisy["text_clean"].apply(
    lambda text: add_typo(text, rng)
)
train_augmented = pd.concat(
    [train_original, train_noisy], ignore_index=True
)

print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
print("Original split sizes:", {
    name: len(frame) for name, frame in splits.items()
})
print("Augmented training size:", len(train_augmented))
print("Validation and test remain unchanged.")

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TextDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame["text_clean"].tolist()
        self.labels = [
            LABEL2ID[label] for label in frame["target"]
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            self.labels[idx], dtype=torch.long
        )
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro"
        ),
        "micro_f1": f1_score(
            labels, predictions, average="micro"
        ),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(LABELS),
    id2label={i: label for label, i in LABEL2ID.items()},
    label2id=LABEL2ID,
    problem_type="single_label_classification",
)

args = TrainingArguments(
    output_dir=f"{OUT}/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    seed=SEED,
    data_seed=SEED,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=TextDataset(train_augmented),
    eval_dataset=TextDataset(splits["dev"]),
    compute_metrics=compute_metrics,
)

print("\nStarting BERT typo-augmentation experiment...")
trainer.train()

results = {}
for name in ["dev", "test"]:
    frame = splits[name]
    pred_output = trainer.predict(TextDataset(frame))
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    y_true = np.array([
        LABEL2ID[label] for label in frame["target"]
    ])
    y_pred = probs.argmax(axis=1)

    results[name] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro")
        ),
    }

    np.save(f"{OUT}/{name}_probabilities.npy", probs)

    pred_frame = pd.DataFrame({
        "text_clean": frame["text_clean"],
        "true_label": frame["target"],
        "predicted_label": [LABELS[i] for i in y_pred],
    })
    for i, label in enumerate(LABELS):
        pred_frame[f"prob_{label}"] = probs[:, i]

    pred_frame.to_csv(
        f"{OUT}/{name}_predictions.csv", index=False
    )

    print(f"\n{name.upper()} RESULTS")
    print(json.dumps(results[name], indent=2))

    if name == "test":
        print(classification_report(
            y_true, y_pred,
            labels=range(len(LABELS)),
            target_names=LABELS,
            zero_division=0,
            digits=4,
        ))

model.save_pretrained(f"{OUT}/best_model")
tokenizer.save_pretrained(f"{OUT}/best_model")

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({
        "model": model_name,
        "experiment": "BERT with one typo-augmented copy per training example",
        "labels": LABELS,
        "seed": SEED,
        "epochs": 3,
        "max_length": 128,
        "train_batch_size": 16,
        "learning_rate": 2e-5,
        "fp16": torch.cuda.is_available(),
        "results": results,
    }, f, indent=2)

print("\nExperiment saved to:", OUT)
