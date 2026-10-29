# Data Statement

## Included Files

### `data/sample/example_format.csv`

Small synthetic file showing the expected input format. It is intended for script testing and column reference only.

### `data/chinese_gold_177_reviewed.csv`

Manually reviewed Chinese social media evaluation set used for cross-lingual boundary testing.

Columns:

| Column | Description |
|---|---|
| `doc_id` | Internal document identifier |
| `platform` | Source platform |
| `source_event` | Disaster event or collection topic |
| `publish_time` | Collection-time metadata where available |
| `text_clean` | Cleaned post text |
| `final_informative_label_gold` | Final informative/non-informative label |
| `final_multilabel_gold` | Pipe-separated final semantic labels |

The reviewed set contains 160 informative examples with at least one semantic label and 17 non-informative examples represented by an empty `final_multilabel_gold` field, meaning no positive semantic label among the six target categories.

### `data/chinese_gold_177_translated.csv`

Translation-bridge evaluation input. It contains the same reviewed Chinese examples plus:

| Column | Description |
|---|---|
| `text_en_translated` | English translation used for English BERT inference |

## Source Datasets Not Redistributed Here

The main English experiments rely on CrisisSense-LLM and HumAID. These datasets are not copied into this repository because they should be obtained from their original maintainers:

- CrisisSense-LLM: https://github.com/KaiYin97/CrsisLLM
- HumAID / CrisisNLP resources: https://crisisnlp.qcri.org/

After downloading, run the conversion scripts in `src/` to create project-format CSV files.

## Label Set

The six semantic labels are:

- `casualty`
- `infrastructure_damage`
- `request_for_help`
- `rescue_or_aid`
- `shelter_or_supply`
- `situation_update`

Detailed label descriptions are provided in `configs/label_schema.yaml`.

## Notes on Reuse

The included Chinese evaluation files are provided to support result checking for this paper. Users should follow the terms of the original platforms and avoid using the examples for user profiling or unrelated redistribution.
