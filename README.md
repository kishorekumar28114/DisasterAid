# Fine-Grained Semantic Classification of Disaster-Related Social Media Text

[![Paper DOI](https://img.shields.io/badge/DOI-10.3390%2Fapp16115621-blue.svg)](https://doi.org/10.3390/app16115621)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the public code, evaluation data, and result artifacts for the published paper:

> Xiaodong Wang and Tengfei Yang. 2026. Fine-Grained Semantic Classification of Disaster-Related Social Media Text for Emergency Management. Applied Sciences, 16(11), 5621. https://doi.org/10.3390/app16115621

The study evaluates six actionable semantic categories for disaster-related social media posts: `casualty`, `infrastructure_damage`, `request_for_help`, `rescue_or_aid`, `shelter_or_supply`, and `situation_update`.

## Repository Contents

```text
configs/                 Label definitions
data/                    Public evaluation files and sample input format
docs/                    Citation, data, and reproducibility notes
figures/                 Paper figures
results/metrics/         Machine-readable metrics
results/tables/          Paper result tables
src/                     Data conversion, training, evaluation, and figure scripts
```

Large model checkpoints, raw third-party datasets, local caches, and credentials are intentionally excluded.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_figures.py --output-dir figures
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\generate_figures.py --output-dir figures
```

Install a GPU-enabled PyTorch build separately if needed. Full reproduction commands are in `docs/REPRODUCIBILITY.md`.

## Data

The main English experiments use public third-party datasets that should be downloaded from their original sources:

- CrisisSense-LLM: https://github.com/KaiYin97/CrsisLLM
- HumAID: https://crisisnlp.qcri.org/

This repository includes:

- `data/sample/example_format.csv`: expected CSV format.
- `data/chinese_gold_177_reviewed.csv`: manually reviewed Chinese evaluation set used for boundary testing. It contains 160 informative examples with at least one semantic label and 17 non-informative examples with no positive semantic label.
- `data/chinese_gold_177_translated.csv`: the same Chinese evaluation set with English translations used in the translation-bridge experiment.

See `docs/DATA_STATEMENT.md` for field definitions and reuse notes.

## Main Results

| Method | Subset Acc | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.6641 | 0.6133 | 0.7784 |
| BERT-base | 0.9076 | 0.8824 | 0.9157 |
| BART-MNLI Zero-shot | 0.0570 | 0.3581 | 0.3623 |

External validation on HumAID gives `Macro-F1 = 0.8132`.

Chinese boundary-test results:

| Setting | Chinese Subset Acc | Chinese Macro-F1 | Chinese Micro-F1 |
|---|---:|---:|---:|
| English BERT-base, direct Chinese input | 0.0960 | 0.0522 | 0.0965 |
| mBERT, English training then Chinese transfer | 0.1921 | 0.2684 | 0.3632 |
| Chinese-to-English translation + English BERT-base | 0.2090 | 0.3603 | 0.4784 |

## Reproduction Workflow

1. Download CrisisSense-LLM and place `dialog.json` under `data/external/CrisisSense_LLM/data/`.
2. Download HumAID event files and place them under `data/external/humaid_hf_events/`.
3. Convert the source datasets.
4. Train baselines and Transformer models.
5. Run external validation, cross-lingual evaluation, translation-bridge evaluation, and selective adjudication.

Detailed commands are provided in `docs/REPRODUCIBILITY.md`.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{Wang2026FGSC,
  title = {Fine-Grained Semantic Classification of Disaster-Related Social Media Text for Emergency Management},
  author = {Wang, Xiaodong and Yang, Tengfei},
  journal = {Applied Sciences},
  volume = {16},
  number = {11},
  pages = {5621},
  year = {2026},
  doi = {10.3390/app16115621},
  url = {https://www.mdpi.com/2076-3417/16/11/5621}
}
```

GitHub citation metadata is provided in `CITATION.cff`; a standalone BibTeX entry is provided in `CITATION.bib`.

## License

Code is released under the MIT License. Third-party datasets remain under their original licenses and terms of use.
