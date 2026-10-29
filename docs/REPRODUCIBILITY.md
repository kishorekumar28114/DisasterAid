# Reproducibility Notes

These commands assume they are run from the repository root.

## 1. Prepare Environment

```bash
pip install -r requirements.txt
```

## 2. Prepare English Data

Place CrisisSense-LLM at:

```text
data/external/CrisisSense_LLM/data/dialog.json
```

Then run:

```bash
python src/convert_crisissense.py
```

Place HumAID event-wise files at:

```text
data/external/humaid_hf_events/
```

Then run:

```bash
python src/convert_humaid.py
```

The scripts create:

```text
data/processed/crisissense_converted.csv
data/processed/humaid_converted.csv
data/processed/humaid_converted_test.csv
```

The CrisisSense-LLM conversion script builds a deterministic proxy event group from normalized disaster type and primary location. All posts in the same proxy group are assigned to the same split. The intended split ratio is 70% training, 10% validation, and 20% test.

## 3. Main Experiments

TF-IDF + Logistic Regression:

```bash
python src/train_baseline.py \
  --data data/processed/crisissense_converted.csv \
  --task multilabel \
  --model tfidf_logreg \
  --output-dir results/tfidf_baseline
```

BERT-base fine-tuning:

```bash
python src/train_bert.py \
  --data data/processed/crisissense_converted.csv \
  --model-name bert-base-uncased \
  --output-dir results/bert_model \
  --eval-split dev \
  --test-split test \
  --save-predictions
```

The validation split is used for checkpoint selection. The test split is evaluated after training.

Default Transformer settings are seed `42`, maximum sequence length `128`, batch size `16`, learning rate `2e-5`, three epochs, and prediction threshold `0.5`. Checkpoint selection uses validation Macro-F1; the held-out test split is reserved for final reporting.

BART-MNLI zero-shot evaluation:

```bash
python src/evaluate_zeroshot.py \
  --data data/processed/crisissense_converted.csv \
  --model-name facebook/bart-large-mnli \
  --output-dir results/zeroshot
```

By default, `--sample-size 0` evaluates the full test split. Pass a positive `--sample-size` only for a quick smoke run.

## 4. External Validation

```bash
python src/cross_lingual_predict.py \
  --model-path results/bert_model/best_model \
  --data data/processed/humaid_converted_test.csv \
  --text-column text_clean \
  --label-column multilabel \
  --output-dir results/external_validation
```

## 5. Chinese Boundary Tests

Direct English BERT on Chinese input:

```bash
python src/cross_lingual_predict.py \
  --model-path results/bert_model/best_model \
  --data data/chinese_gold_177_reviewed.csv \
  --text-column text_clean \
  --label-column final_multilabel_gold \
  --output-dir results/chinese_direct_bert
```

mBERT trained on English CrisisSense-LLM and evaluated on Chinese input:

```bash
python src/train_bert.py \
  --data data/processed/crisissense_converted.csv \
  --model-name bert-base-multilingual-cased \
  --output-dir results/mbert_model \
  --eval-split dev \
  --test-split test

python src/cross_lingual_predict.py \
  --model-path results/mbert_model/best_model \
  --data data/chinese_gold_177_reviewed.csv \
  --text-column text_clean \
  --label-column final_multilabel_gold \
  --output-dir results/chinese_mbert
```

Translation bridge using the included translated column:

```bash
python src/translate_then_predict.py \
  --data data/chinese_gold_177_translated.csv \
  --translation-column text_en_translated \
  --label-column final_multilabel_gold \
  --model-path results/bert_model/best_model \
  --output-dir results/chinese_translate_bert
```

## 6. Selective Adjudication

```bash
python src/run_selective_adjudication.py \
  --data data/processed/crisissense_converted.csv \
  --external-data data/processed/humaid_converted_test.csv \
  --output-dir results/selective_adjudication \
  --seed-ratio 0.05 \
  --min-seed-per-label 20 \
  --rounds 3 \
  --batch-size 200
```

## 7. Figures

The published figure files are included. To regenerate figures from the provided metrics:

```bash
python src/generate_figures.py --output-dir figures
```
