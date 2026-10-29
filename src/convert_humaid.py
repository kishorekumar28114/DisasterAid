"""
Convert HumAID event-wise TSV or JSON files into the project CSV format.

Supported input formats:
  - Event-wise TSV files with columns: tweet_id, tweet_text, class_label
  - Event-wise JSON files from the Hugging Face mirror

File names or parent directories must contain train/dev/test to infer the split.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LABEL_MAPPING = {
    "caution_and_advice": "situation_update",
    "displaced_people_and_evacuations": "shelter_or_supply",
    "infrastructure_and_utility_damage": "infrastructure_damage",
    "injured_or_dead_people": "casualty",
    "missing_or_found_people": "casualty",
    "other_relevant_information": "situation_update",
    "requests_or_urgent_needs": "request_for_help",
    "rescue_volunteering_or_donation_effort": "rescue_or_aid",
}

IGNORED_LABELS = {
    "dont_know_cant_judge",
    "not_humanitarian",
    "sympathy_and_support",
}

DISASTER_HINTS = {
    "earthquake": "earthquake",
    "hurricane": "hurricane",
    "wildfire": "wildfire",
    "fire": "wildfire",
    "flood": "flood",
    "cyclone": "cyclone",
    "typhoon": "cyclone",
    "tornado": "tornado",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HumAID event-wise TSV or JSON files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/external/humaid_set1",
        help="Root directory of extracted HumAID event-wise files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/humaid_converted.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def infer_split(path):
    """Determine train/dev/test split from the file name."""
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "valid" in name or "val" in name:
        return "dev"
    if "test" in name:
        return "test"
    raise ValueError(f"Cannot infer split from file name: {path.name}")


def infer_event_name(path, root):
    """Extract the event name from the directory structure or filename."""
    if path.parent != root:
        return path.parent.name
    stem = path.stem
    for suffix in ("_train", "_dev", "_test", "-train", "-dev", "-test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def infer_disaster_type(event_name):
    """Guess the disaster type from the event name."""
    lowered = event_name.lower()
    for hint, disaster_type in DISASTER_HINTS.items():
        if hint in lowered:
            return disaster_type
    return "unknown"


def find_data_files(root):
    """Locate all TSV and JSON data files under the root directory."""
    files = sorted(root.rglob("*.tsv")) + sorted(root.rglob("*.json"))
    if not files:
        raise SystemExit(f"No TSV or JSON files found under: {root}")
    return files


def validate_rows(rows, path):
    """Check that rows contain the expected columns."""
    if not rows:
        return rows
    fieldnames = set(rows[0].keys())
    if {"tweet_id", "class_label"} <= fieldnames and "tweet_text" not in fieldnames:
        raise SystemExit(
            "Detected id-only HumAID files without tweet_text. "
            "Please provide the event-wise set with full text."
        )
    required = {"tweet_id", "tweet_text", "class_label"}
    if not required <= fieldnames:
        raise SystemExit(
            f"Unexpected columns in {path}: {sorted(fieldnames)}. "
            "Expected tweet_id, tweet_text, class_label."
        )
    return rows


def read_rows(path):
    """Read rows from a TSV or JSON file."""
    if path.suffix.lower() == ".tsv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        return validate_rows(rows, path)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows = data.get("data", [data]) if "data" in data else [data]
        else:
            rows = data
        return validate_rows(rows, path)

    raise SystemExit(f"Unsupported file format: {path}")


def normalize_event_group(event_name):
    """Normalize event name for use as a group identifier."""
    return re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_") or "unknown"


def convert_rows(data_path, root):
    """Convert all rows from a single data file into project records."""
    split = infer_split(data_path)
    event_name = infer_event_name(data_path, root)
    disaster_type = infer_disaster_type(event_name)
    grouped = defaultdict(lambda: {"text": "", "labels": set()})

    for row in read_rows(data_path):
        tweet_id = str(row["tweet_id"]).strip()
        tweet_text = str(row["tweet_text"]).strip()
        raw_label = str(row["class_label"]).strip()
        mapped_label = LABEL_MAPPING.get(raw_label)

        if raw_label in IGNORED_LABELS or mapped_label is None:
            continue
        if not tweet_id or not tweet_text:
            continue

        grouped[tweet_id]["text"] = tweet_text[:500]
        grouped[tweet_id]["labels"].add(mapped_label)

    converted = []
    for tweet_id, item in grouped.items():
        labels = sorted(item["labels"])
        if not labels:
            continue
        converted.append(
            {
                "doc_id": f"humaid_{tweet_id}",
                "platform": "twitter",
                "event_name": event_name,
                "disaster_type": disaster_type,
                "publish_time": "",
                "time_source": "not_provided_in_humaid_set1",
                "text_clean": item["text"],
                "lang": "en",
                "has_location": "",
                "has_number": "",
                "is_duplicate": "false",
                "split": split,
                "informative_label": "informative",
                "event_category": disaster_type,
                "humanitarian_aid": "|".join(labels),
                "locations": "",
                "multilabel": "|".join(labels),
                "event_group": normalize_event_group(event_name),
                "source_dataset": "HumAID",
            }
        )

    return converted


def main():
    args = parse_args()
    root = Path(args.input_dir)
    output_path = Path(args.output)

    data_files = find_data_files(root)
    converted = []
    for data_file in data_files:
        converted.extend(convert_rows(data_file, root))

    if not converted:
        raise SystemExit("No usable HumAID rows were converted.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=converted[0].keys())
        writer.writeheader()
        writer.writerows(converted)

    split_counts = Counter(row["split"] for row in converted)
    disaster_counts = Counter(row["disaster_type"] for row in converted)
    label_counts = Counter()
    for row in converted:
        for label in row["multilabel"].split("|"):
            if label:
                label_counts[label] += 1

    print(f"Converted rows: {len(converted)}")
    print(f"Output: {output_path}")
    print("\nSplit distribution:")
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count}")
    print("\nDisaster type distribution:")
    for disaster_type, count in disaster_counts.items():
        print(f"  {disaster_type}: {count}")
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
