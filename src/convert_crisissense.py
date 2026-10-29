"""
Convert CrisisSense-LLM dialog.json into the project CSV format.

The source data does not provide explicit event ids, so this script builds a
deterministic proxy event group from disaster type and primary location, then
performs group-based splitting instead of text-level random splitting.
"""

import csv
import json
import re
import uuid
from collections import Counter
from pathlib import Path


INPUT_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "external"
    / "CrisisSense_LLM"
    / "data"
    / "dialog.json"
)
OUTPUT_FILE = (
    Path(__file__).parent.parent / "data" / "processed" / "crisissense_converted.csv"
)

EVENT_MAPPING = {
    "wildfires": "wildfire",
    "cyclone": "cyclone",
    "earthquake": "earthquake",
    "hurricane": "hurricane",
    "floods": "flood",
    "flood": "flood",
    "typhoon": "cyclone",
    "tornado": "tornado",
}

HUMANITARIAN_AID_MAPPING = {
    "caution_and_advice": "situation_update",
    "displaced_people_and_evacuations": "shelter_or_supply",
    "infrastructure_and_utility_damage": "infrastructure_damage",
    "injured_or_dead_people": "casualty",
    "missing_or_found_people": "casualty",
    "requests_or_urgent_needs": "request_for_help",
    "rescue_volunteering_or_donation_effort": "rescue_or_aid",
}

SPLIT_RATIOS = {
    "train": 0.7,
    "dev": 0.1,
    "test": 0.2,
}


def parse_dict_string(content):
    """Parse a Python-style dict string into a dictionary."""
    try:
        return json.loads(content.replace("'", '"'))
    except Exception:
        return {}


def normalize_location(location):
    """Normalize a location string to a filesystem-safe identifier."""
    normalized = re.sub(r"[^a-z0-9]+", "_", location.lower()).strip("_")
    return normalized or "unknown"


def choose_primary_location(locations):
    """Deduplicate and filter location entries."""
    seen = set()
    unique_locations = []
    for location in locations:
        cleaned = str(location).strip()
        if not cleaned or cleaned.lower() == "yyy":
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique_locations.append(cleaned)
    return unique_locations


def proxy_event_group(event_type, locations):
    """Build a proxy event group identifier from type and primary location."""
    if locations:
        return f"{event_type}:{normalize_location(locations[0])}"
    return f"{event_type}:unknown"


def assign_group_splits(records):
    """Assign train/dev/test splits at the event-group level."""
    group_sizes = Counter(record["event_group"] for record in records)
    total = sum(group_sizes.values())
    target_counts = {
        split: total * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    assigned_counts = {split: 0 for split in SPLIT_RATIOS}
    group_split = {}

    ordered_groups = sorted(group_sizes.items(), key=lambda item: (-item[1], item[0]))
    for idx, (group_name, group_size) in enumerate(ordered_groups):
        remaining_groups = len(ordered_groups) - idx
        empty_splits = [split for split, count in assigned_counts.items() if count == 0]

        if empty_splits and len(empty_splits) == remaining_groups:
            chosen_split = empty_splits[0]
        else:
            deficits = {
                split: target_counts[split] - assigned_counts[split]
                for split in SPLIT_RATIOS
            }
            chosen_split = max(
                SPLIT_RATIOS,
                key=lambda split: (deficits[split], -assigned_counts[split], split == "train"),
            )

        group_split[group_name] = chosen_split
        assigned_counts[chosen_split] += group_size

    return group_split


def extract_record(item):
    """Extract a single record from a CrisisSense-LLM dialog item."""
    if item.get("type") != "domain":
        return None

    conversation = item.get("conversation", [])
    text = None
    event_type = None
    humanitarian_aid = None
    locations = []

    for message in conversation:
        if message.get("role") == "user" and str(message.get("task_id")) == "1":
            text = message.get("content", "")
        elif message.get("role") == "assistant":
            parsed = parse_dict_string(message.get("content", ""))
            if "EVENT_CATEGORY" in parsed:
                raw_event = str(parsed["EVENT_CATEGORY"]).lower()
                event_type = EVENT_MAPPING.get(raw_event, raw_event)
            if "HUMANITARIAN_AID" in parsed:
                raw_aid = str(parsed["HUMANITARIAN_AID"]).lower()
                humanitarian_aid = HUMANITARIAN_AID_MAPPING.get(raw_aid, raw_aid)
            if "LOCATIONS" in parsed:
                locs = parsed["LOCATIONS"]
                if isinstance(locs, list):
                    locations = choose_primary_location(locs)
                else:
                    locations = choose_primary_location([locs])

    if not text or not event_type:
        return None

    labels = [humanitarian_aid] if humanitarian_aid else []
    return {
        "doc_id": f"crisis_{uuid.uuid4().hex[:8]}",
        "platform": "twitter",
        "event_name": event_type,
        "disaster_type": event_type,
        "publish_time": "",
        "time_source": "missing_in_source_dataset",
        "text_clean": text[:500],
        "lang": "en",
        "has_location": "true" if locations else "false",
        "has_number": "false",
        "is_duplicate": "false",
        "informative_label": "informative",
        "event_category": event_type,
        "humanitarian_aid": humanitarian_aid or "",
        "locations": "|".join(locations),
        "multilabel": "|".join(labels),
        "event_group": proxy_event_group(event_type, locations),
        "split_strategy": "group_proxy_event",
    }


def convert_data():
    """Main conversion routine."""
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for item in data:
        record = extract_record(item)
        if record is not None:
            converted.append(record)

    group_split = assign_group_splits(converted)
    for record in converted:
        record["split"] = group_split[record["event_group"]]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=converted[0].keys())
        writer.writeheader()
        writer.writerows(converted)

    print(f"Conversion complete: {len(converted)} records")
    print(f"Output: {OUTPUT_FILE}")

    print("\nEvent type distribution:")
    for event_name, count in Counter(r["event_category"] for r in converted).most_common():
        print(f"  {event_name}: {count}")

    print("\nProxy event group distribution:")
    for group_name, count in Counter(r["event_group"] for r in converted).most_common(15):
        print(f"  {group_name}: {count}")

    print("\nSplit distribution:")
    for split_name, count in Counter(r["split"] for r in converted).most_common():
        print(f"  {split_name}: {count}")


if __name__ == "__main__":
    convert_data()
