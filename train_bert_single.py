import os, json, random
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
OUT = f"{REPO}/results_ours/bert_single"
os.makedirs(OUT, exist_ok=True)

LABELS = [
    "casualty", "infrastructure_damage", "request_for_help",
    "rescue_or_aid", "shelter_or_supply", "situation_update"
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(CSV)
if "is_duplicate" in df.columns:
    dup = df["is_duplicate"].fillna("false").astype(str).str.lower().eq("true")
    df = df[~dup].copy()

def get_label(value):
    parts = [x.strip() for x in str(value).split("|") if x.strip()]
    return parts[0] if len(parts) == 1 and parts[0] in LABEL2ID else None

df["target"] = df["multilabel"].apply(get_label)
df["text_clean"] = df["text_clean"].fillna("").astype(str).str.strip()
df = df[df["target"].notna() & df["text_clean"].ne("")].copy()

splits = {
    name: df[df["split"] == name].reset_index(drop=True)
    for name in ["train", "dev", "test"]
}
assert all(len(x) for x in splits.values())
assert set(splits["train"]["target"]) == set(LABELS)

print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
print("Split sizes:", {k: len(v) for k, v in splits.items()})

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TextDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame["text_clean"].tolist()
        self.labels = [LABEL2ID[x] for x in frame["target"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i], max_length=128,
            truncation=True, padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    y_true = eval_pred.label_ids
    y_pred = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro")
    }

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(LABELS),
    id2label={i: label for label, i in LABEL2ID.items()},
    label2id=LABEL2ID,
    problem_type="single_label_classification"
)

args = TrainingArguments(
    output_dir=f"{OUT}/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
    seed=SEED,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=TextDataset(splits["train"]),
    eval_dataset=TextDataset(splits["dev"]),
    compute_metrics=compute_metrics
)

print("\nStarting BERT fine-tuning...")
trainer.train()

results = {}
for name in ["dev", "test"]:
    frame = splits[name]
    pred_output = trainer.predict(TextDataset(frame))
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    # Stable softmax probabilities for multiclass classification.
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_true = np.array([LABEL2ID[x] for x in frame["target"]])
    y_pred = probs.argmax(axis=1)

    results[name] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro"))
    }

    np.save(f"{OUT}/{name}_probabilities.npy", probs)
    frame_out = pd.DataFrame({
        "text_clean": frame["text_clean"],
        "true_label": frame["target"],
        "predicted_label": [LABELS[i] for i in y_pred]
    })
    for i, label in enumerate(LABELS):
        frame_out[f"prob_{label}"] = probs[:, i]
    frame_out.to_csv(f"{OUT}/{name}_predictions.csv", index=False)

    print(f"\n{name.upper()} RESULTS")
    print(json.dumps(results[name], indent=2))
    if name == "test":
        print(classification_report(
            y_true, y_pred, labels=range(len(LABELS)),
            target_names=LABELS, zero_division=0, digits=4
        ))

model.save_pretrained(f"{OUT}/best_model")
tokenizer.save_pretrained(f"{OUT}/best_model")

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({
        "model": model_name,
        "task": "single-label six-class classification",
        "labels": LABELS,
        "seed": SEED,
        "epochs": 3,
        "max_length": 128,
        "train_batch_size": 16,
        "learning_rate": 2e-5,
        "fp16": torch.cuda.is_available(),
        "results": results
    }, f, indent=2)

print("\nBERT results and probabilities saved to:", OUT)
