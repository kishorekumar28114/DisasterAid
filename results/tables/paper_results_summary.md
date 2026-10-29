# 论文结果摘要

- 主实验最优模型为 `BERT-base`，`Macro-F1 = 0.8824`。
- 相比 `TF-IDF + LogReg`，主实验 `Macro-F1` 提升 `0.2691`。
- 外部验证 `HumAID test` 上，`BERT-base` 仍达到 `Macro-F1 = 0.8132`。
- 中文人工复核 gold set 上，`mBERT` 的 `Macro-F1 = 0.2684`，明显优于英文单语 `BERT-base` 的 `0.0522`。
- 中文翻译成英文后再预测的 `BERT-base` 达到 `Macro-F1 = 0.3603`，作为翻译桥接对照。
- 固定 `600` 条裁决预算下，`selective_adjudication` 将主测试集 `Macro-F1` 提升到 `0.7792`，外部验证提升到 `0.7153`。
