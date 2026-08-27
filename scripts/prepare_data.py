from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REQUIRED = ("sample_id", "image_name", "image_path", "text", "language", "label", "split")


def normalize_language(value: object) -> str:
    key = str(value).strip().lower()
    if key in {"zh", "cn", "chinese", "中文"}:
        return "zh"
    if key in {"en", "eng", "english", "英文"}:
        return "en"
    raise ValueError(f"unsupported language: {value}")


def normalize_label(value: object) -> int:
    key = str(value).strip().lower()
    if key in {"0", "literal", "字面"}:
        return 0
    if key in {"1", "metaphor", "metaphorical", "隐喻"}:
        return 1
    raise ValueError(f"unsupported label: {value}")


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError("input must be .jsonl or .csv")


def validate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    seen_ids: set[str] = set()
    image_splits: dict[str, set[str]] = defaultdict(set)
    output: list[dict] = []
    for index, source in enumerate(rows, 1):
        missing = [name for name in REQUIRED if name not in source or source[name] in (None, "")]
        if missing:
            raise ValueError(f"row {index} missing fields: {', '.join(missing)}")
        row = dict(source)
        row["sample_id"] = str(row["sample_id"])
        if row["sample_id"] in seen_ids:
            raise ValueError(f"duplicate sample_id: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        row["language"] = normalize_language(row["language"])
        row["label"] = normalize_label(row["label"])
        row["split"] = str(row["split"]).strip().lower()
        image_splits[str(row["image_name"])].add(row["split"])
        output.append(row)
    leakage = {name: sorted(parts) for name, parts in image_splits.items() if len(parts) > 1}
    if leakage:
        raise ValueError(f"image_name appears across splits: {json.dumps(leakage, ensure_ascii=False)}")
    report = {
        "samples": len(output),
        "unique_sample_ids": len(seen_ids),
        "unique_images": len(image_splits),
        "languages": {lang: sum(r["language"] == lang for r in output) for lang in ("zh", "en")},
        "labels": {str(label): sum(r["label"] == label for r in output) for label in (0, 1)},
        "splits": {part: sum(r["split"] == part for r in output) for part in sorted({r["split"] for r in output})},
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and normalize existing data splits.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    rows, report = validate_rows(read_rows(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

