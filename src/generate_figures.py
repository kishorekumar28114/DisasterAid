"""
Generate paper-ready comparison figures from saved experiment results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


LABEL_ORDER = [
    "casualty",
    "infrastructure_damage",
    "request_for_help",
    "rescue_or_aid",
    "shelter_or_supply",
    "situation_update",
]

LABEL_DISPLAY = {
    "casualty": "Casualty",
    "infrastructure_damage": "Infrastructure\nDamage",
    "request_for_help": "Request\nfor Help",
    "rescue_or_aid": "Rescue/Aid",
    "shelter_or_supply": "Shelter/Supply",
    "situation_update": "Situation",
}

COLORS = ["#355C7D", "#C06C84", "#6C9A8B", "#F67280"]

STRATEGY_DISPLAY = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "disagreement": "Disagreement",
    "selective_adjudication": "Selective Adjudication",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison figures for the paper.")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory to store generated figures.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_first_existing(*candidates: str) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def extract_plain_metrics(data: dict) -> tuple[float, float, float]:
    return (
        float(data.get("subset_accuracy", 0.0)),
        float(data.get("f1_macro", 0.0)),
        float(data.get("f1_micro", 0.0)),
    )


def extract_bert_metrics(data: dict) -> tuple[float, float, float]:
    return (
        transformer_metric(data, "subset_accuracy"),
        transformer_metric(data, "f1_macro"),
        transformer_metric(data, "f1_micro"),
    )


def transformer_metric(data: dict, name: str) -> float:
    for prefix in ("test", "eval", "validation"):
        key = f"{prefix}_{name}"
        if key in data:
            return float(data[key])
    return float(data.get(name, 0.0))


def transformer_label_f1(data: dict, label: str) -> float:
    for prefix in ("test", "eval", "validation"):
        key = f"{prefix}_{label}_f1"
        if key in data:
            return float(data[key])
    return float(data.get(f"{label}_f1", 0.0))


def draw_grouped_bar_chart(
    title: str,
    categories: list[str],
    series: dict[str, list[float]],
    output_path: Path,
    ylabel: str = "Score",
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    x_positions = list(range(len(categories)))
    bar_count = len(series)
    width = 0.8 / max(bar_count, 1)

    for idx, (name, values) in enumerate(series.items()):
        offset = (idx - (bar_count - 1) / 2) * width
        positions = [x + offset for x in x_positions]
        bars = axis.bar(
            positions,
            values,
            width=width,
            label=name,
            color=COLORS[idx % len(COLORS)],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.015, 1.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(categories)
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False, ncol=min(3, bar_count))
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def add_box(
    axis,
    xy,
    width,
    height,
    title,
    body,
    facecolor,
    title_y=0.73,
    body_y=0.40,
    title_fontsize=12,
    body_fontsize=10,
    body_linespacing=1.4,
):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#2f3b52",
        facecolor=facecolor,
    )
    axis.add_patch(box)
    x0, y0 = xy
    axis.text(
        x0 + width / 2,
        y0 + height * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        color="#223046",
    )
    axis.text(
        x0 + width / 2,
        y0 + height * body_y,
        body,
        ha="center",
        va="center",
        fontsize=body_fontsize,
        color="#223046",
        linespacing=body_linespacing,
    )


def add_arrow(axis, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.5,
        color="#2f3b52",
    )
    axis.add_patch(arrow)


def draw_framework_figure(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        (
            (0.04, 0.26),
            "Input Data",
            "CrisisSense-LLM\nHumAID\nChinese Social Media",
            "#E8F1F2",
        ),
        (
            (0.26, 0.26),
            "Preprocess",
            "Cleaning\nLabel mapping\nEvent-group split",
            "#FDEBD3",
        ),
        (
            (0.48, 0.26),
            "Modeling",
            "TF-IDF + LogReg\nBERT-base\nmBERT / Zero-shot",
            "#E8F3E6",
        ),
        (
            (0.70, 0.26),
            "Evaluation",
            "Main task\nExternal validation\nCross-lingual test",
            "#F8E8EE",
        ),
    ]

    width = 0.18
    height = 0.34
    for xy, title, body, facecolor in boxes:
        add_box(axis, xy, width, height, title, body, facecolor)

    add_box(
        axis,
        (0.39, 0.68),
        0.22,
        0.22,
        "Human-in-the-Loop",
        "Seed model\nSelective adjudication\nIncremental retraining",
        "#EEF0FA",
        title_y=0.78,
        body_y=0.36,
        body_fontsize=9.4,
        body_linespacing=1.25,
    )

    add_arrow(axis, (0.22, 0.43), (0.26, 0.43))
    add_arrow(axis, (0.44, 0.43), (0.48, 0.43))
    add_arrow(axis, (0.66, 0.43), (0.70, 0.43))
    add_arrow(axis, (0.57, 0.60), (0.57, 0.68))
    add_arrow(axis, (0.79, 0.60), (0.61, 0.82))

    axis.text(
        0.5,
        0.95,
        "Research Framework and Experimental Workflow",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#223046",
    )
    axis.text(
        0.5,
        0.08,
        "Classification is evaluated first, then extended to cross-dataset, cross-lingual, and selective adjudication analysis.",
        ha="center",
        va="center",
        fontsize=10,
        color="#45556B",
    )

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def load_main_series() -> dict[str, list[float]]:
    tfidf_path = pick_first_existing(
        "results/metrics/tfidf_logreg_main_metrics.json",
    )
    bert_path = pick_first_existing(
        "results/metrics/bert_main_metrics.json",
    )
    llm_path = pick_first_existing("results/metrics/bart_mnli_zeroshot_metrics.json")

    series: dict[str, list[float]] = {}
    if tfidf_path:
        tfidf_data = load_json(tfidf_path).get("metrics", {})
        series["TF-IDF + LogReg"] = list(extract_plain_metrics(tfidf_data))
    if bert_path:
        series["BERT-base"] = list(extract_bert_metrics(load_json(bert_path)))
    if llm_path:
        series["LLM Zero-Shot"] = list(extract_plain_metrics(load_json(llm_path)))
    return series


def load_per_label_series() -> dict[str, list[float]]:
    tfidf_path = pick_first_existing(
        "results/metrics/tfidf_logreg_main_metrics.json",
    )
    bert_path = pick_first_existing(
        "results/metrics/bert_main_metrics.json",
    )

    series: dict[str, list[float]] = {}
    if bert_path:
        bert_data = load_json(bert_path)
        series["BERT-base"] = [
            transformer_label_f1(bert_data, label) for label in LABEL_ORDER
        ]
    if tfidf_path:
        tfidf_data = load_json(tfidf_path).get("metrics", {})
        label_metrics = tfidf_data.get("label_metrics", {})
        series["TF-IDF + LogReg"] = [
            float(label_metrics.get(label, {}).get("f1", 0.0)) for label in LABEL_ORDER
        ]
    return series


def load_cross_lingual_series() -> dict[str, list[float]]:
    english_zero_path = pick_first_existing(
        "results/metrics/chinese_direct_bert.json",
    )
    mbert_path = pick_first_existing(
        "results/metrics/chinese_mbert.json",
    )
    translate_bert_path = pick_first_existing(
        "results/metrics/chinese_translate_bert.json",
    )

    english_metrics = {"subset_accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}
    if english_zero_path:
        english_data = load_json(english_zero_path)
        english_metrics = english_data.get("metrics", english_metrics)

    mbert_metrics = {"subset_accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}
    if mbert_path:
        mbert_data = load_json(mbert_path)
        mbert_metrics = mbert_data.get("metrics", mbert_metrics)

    series = {
        "English BERT": list(extract_plain_metrics(english_metrics)),
        "mBERT": list(extract_plain_metrics(mbert_metrics)),
    }
    if translate_bert_path:
        translate_data = load_json(translate_bert_path)
        translate_metrics = translate_data.get(
            "metrics", {"subset_accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}
        )
        series["Translate+BERT"] = list(extract_plain_metrics(translate_metrics))
    return series


def load_external_validation_series() -> dict[str, list[float]]:
    crisissense_path = pick_first_existing(
        "results/metrics/bert_main_metrics.json",
    )
    humaid_path = pick_first_existing(
        "results/metrics/humaid_external_validation.json",
    )

    series: dict[str, list[float]] = {}
    if crisissense_path:
        series["CrisisSense test"] = list(extract_bert_metrics(load_json(crisissense_path)))
    if humaid_path:
        humaid_data = load_json(humaid_path)
        series["HumAID test"] = list(extract_plain_metrics(humaid_data.get("metrics", {})))
    return series


def load_selective_adjudication_rows() -> list[dict[str, str]]:
    csv_path = pick_first_existing(
        "results/metrics/selective_adjudication_round_metrics.csv"
    )
    if not csv_path:
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def draw_selective_adjudication_figure(output_path: Path) -> None:
    rows = load_selective_adjudication_rows()
    if not rows:
        return

    strategy_order = [
        strategy
        for strategy in ["random", "uncertainty", "disagreement", "selective_adjudication"]
        if any(row.get("strategy") == strategy for row in rows)
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    metric_specs = [
        ("eval_f1_macro", "CrisisSense Test Macro-F1"),
        ("external_f1_macro", "HumAID Test Macro-F1"),
    ]
    label_offsets = {
        ("eval_f1_macro", "uncertainty"): -0.004,
        ("eval_f1_macro", "disagreement"): -0.001,
        ("eval_f1_macro", "selective_adjudication"): 0.004,
        ("external_f1_macro", "uncertainty"): -0.004,
        ("external_f1_macro", "disagreement"): 0.001,
        ("external_f1_macro", "selective_adjudication"): 0.004,
    }

    for axis, (metric_key, title) in zip(axes, metric_specs):
        for idx, strategy in enumerate(strategy_order):
            strategy_rows = [
                row for row in rows if row.get("strategy") == strategy and row.get(metric_key)
            ]
            strategy_rows.sort(key=lambda row: int(row["round"]))
            x_values = [int(row["adjudicated_total"]) for row in strategy_rows]
            y_values = [float(row[metric_key]) for row in strategy_rows]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=COLORS[idx % len(COLORS)],
                label=STRATEGY_DISPLAY.get(strategy, strategy),
            )
            if x_values and y_values:
                label_y = y_values[-1] + label_offsets.get((metric_key, strategy), 0.0)
                axis.text(
                    x_values[-1] + 8,
                    label_y,
                    f"{y_values[-1]:.3f}",
                    fontsize=8,
                    color=COLORS[idx % len(COLORS)],
                    va="center",
                )

        axis.set_title(title)
        axis.set_xlabel("Adjudicated Samples")
        axis.set_ylabel("Macro-F1")
        axis.set_xlim(left=0)
        axis.set_ylim(0.60, 0.82)
        axis.grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_caption_file(output_dir: Path) -> None:
    caption_text = """# Paper Figure Captions

1. `framework_pipeline.png`
   图题建议：图 1 研究框架与实验流程图。展示数据输入、预处理、模型构建、主任务/外部验证/跨语言评测以及选择性增量修订之间的关系。

2. `main_results_comparison.png`
   图题建议：图 2 主实验结果对比。展示 TF-IDF + LogReg、BERT-base 和 LLM Zero-Shot 在代理事件组切分下的 Subset Accuracy、Macro-F1 和 Micro-F1。

3. `per_label_f1_comparison.png`
   图题建议：图 3 主任务逐标签 F1 对比。展示 BERT-base 与 TF-IDF + LogReg 在 6 个标签上的分类性能差异。

4. `cross_lingual_comparison.png`
   图题建议：图 4 中文跨语言迁移结果对比。展示英文单语 BERT、mBERT 以及翻译后英文 BERT 在中文社交媒体测试集上的整体性能差异。

5. `external_validation_comparison.png`
   图题建议：图 5 外部验证结果对比。展示主模型在 CrisisSense 测试集与 HumAID 测试集上的迁移表现。

6. `selective_adjudication_progress.png`
   图题建议：图 6 选择性裁决循环演化结果。展示不同裁决策略在固定预算下随新增裁决样本数量增长时的主测试集与外部验证集 Macro-F1 变化趋势。
"""
    caption_path = output_dir / "paper_figure_captions.md"
    caption_path.write_text(caption_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    metric_labels = ["Subset Acc", "Macro-F1", "Micro-F1"]
    label_names = [LABEL_DISPLAY[label] for label in LABEL_ORDER]

    draw_framework_figure(output_dir / "framework_pipeline.png")
    draw_grouped_bar_chart(
        title="Main Results Comparison",
        categories=metric_labels,
        series=load_main_series(),
        output_path=output_dir / "main_results_comparison.png",
    )
    draw_grouped_bar_chart(
        title="Per-Label F1 Comparison",
        categories=label_names,
        series=load_per_label_series(),
        output_path=output_dir / "per_label_f1_comparison.png",
    )
    draw_grouped_bar_chart(
        title="Cross-Lingual Transfer Comparison",
        categories=metric_labels,
        series=load_cross_lingual_series(),
        output_path=output_dir / "cross_lingual_comparison.png",
    )
    draw_grouped_bar_chart(
        title="External Validation Comparison",
        categories=metric_labels,
        series=load_external_validation_series(),
        output_path=output_dir / "external_validation_comparison.png",
    )
    draw_selective_adjudication_figure(
        output_path=output_dir / "selective_adjudication_progress.png"
    )
    write_caption_file(output_dir)

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
