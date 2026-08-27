from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class GuardRecord:
    sample_id: str
    stage: str
    error_type: str | None
    retry_count: int
    fallback_used: bool
    final_status: str


def validate_output(text: str, required_text: str, forbidden: list[str]) -> tuple[bool, str | None]:
    if not text or not text.strip():
        return False, "empty_output"
    if required_text and required_text not in text:
        return False, "missing_required_field"
    if any(item and item.lower() in text.lower() for item in forbidden):
        return False, "forbidden_content"
    return True, None


def run_guarded(
    sample_id: str,
    stage: str,
    producer: Callable[[], str],
    required_text: str,
    forbidden: list[str],
    previous_valid: str | None = None,
    max_retries: int = 2,
) -> tuple[str | None, GuardRecord]:
    error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            value = producer()
            valid, error = validate_output(value, required_text, forbidden)
        except (TypeError, ValueError, json.JSONDecodeError):
            value, valid, error = "", False, "parse_error"
        if valid:
            return value, GuardRecord(sample_id, stage, None, attempt, False, "valid")
    if previous_valid is not None:
        return previous_valid, GuardRecord(sample_id, stage, error, max_retries, True, "fallback")
    return None, GuardRecord(sample_id, stage, error, max_retries, False, "excluded")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit already generated text with the paper's output rules.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--required-text", default="")
    parser.add_argument("--forbidden", action="append", default=[])
    args = parser.parse_args()
    records = []
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ok, error = validate_output(str(row.get(args.text_field, "")), args.required_text, args.forbidden)
            record = GuardRecord(
                sample_id=str(row.get("sample_id", "")),
                stage=args.text_field,
                error_type=error,
                retry_count=0,
                fallback_used=False,
                final_status="valid" if ok else "invalid",
            )
            records.append(asdict(record))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
