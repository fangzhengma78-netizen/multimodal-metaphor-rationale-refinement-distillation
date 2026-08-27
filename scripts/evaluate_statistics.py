from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

def parse_prediction(value: object) -> int:
    if isinstance(value, int) and value in (0, 1):
        return value
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    match = re.search(r"<label>\s*(literal|metaphor)\s*</label>", text, re.I)
    if not match:
        raise ValueError("unparseable prediction")
    return int(match.group(1).lower() == "metaphor")


def metrics(rows: list[dict]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    gold, pred, failures = [], [], 0
    for row in rows:
        try:
            pred.append(parse_prediction(row["prediction"]))
            gold.append(int(row["label"]))
        except (KeyError, TypeError, ValueError):
            failures += 1
    if not gold:
        raise ValueError("no parseable predictions")
    return {
        "valid_samples": len(gold),
        "parse_failures": failures,
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro")),
        "weighted_f1": float(f1_score(gold, pred, average="weighted")),
    }


def bootstrap_indices(n: int, repetitions: int, seed: int):
    import numpy as np

    if n <= 0:
        raise ValueError("bootstrap input must not be empty")
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    return np.random.default_rng(seed).integers(0, n, size=(repetitions, n))


def bootstrap_three_seed(groups: list[list[dict]], repetitions: int = 1000, seed: int = 42) -> dict:
    import numpy as np
    from sklearn.metrics import f1_score

    if len(groups) != 3:
        raise ValueError("exactly three seed files are required")
    if not groups[0]:
        raise ValueError("seed files must not be empty")

    ids = [[str(row["sample_id"]) for row in group] for group in groups]
    if ids[1:] != ids[:-1]:
        raise ValueError("three seed files must have identical sample_id order")

    labels = [[int(row["label"]) for row in group] for group in groups]
    if labels[1:] != labels[:-1]:
        raise ValueError("three seed files must have identical labels")

    gold = np.array(labels[0])
    predictions = [np.array([parse_prediction(row["prediction"]) for row in group]) for group in groups]
    values = []
    for index in bootstrap_indices(len(gold), repetitions, seed):
        seed_scores = [
            f1_score(gold[index], prediction[index], average="weighted")
            for prediction in predictions
        ]
        values.append(np.mean(seed_scores))
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "bootstrap_seed": seed,
        "resamples": repetitions,
        "weighted_f1_mean": float(np.mean(values)),
        "ci_95": [float(low), float(high)],
    }


def suff_comp(rows: list[dict]) -> dict:
    import numpy as np

    if not rows:
        raise ValueError("evidence confidence input must not be empty")

    sufficiency = []
    comprehensiveness = []
    for row in rows:
        full = float(row["p_full"])
        evidence_only = float(row["p_evidence_only"])
        evidence_removed = float(row["p_evidence_removed"])
        if any(value < 0.0 or value > 1.0 for value in (full, evidence_only, evidence_removed)):
            raise ValueError("confidence values must be normalized probabilities in [0, 1]")
        sufficiency.append(full - evidence_only)
        comprehensiveness.append(full - evidence_removed)

    return {
        "samples": len(rows),
        "sufficiency": float(np.mean(sufficiency)),
        "comprehensiveness": float(np.mean(comprehensiveness)),
    }


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics, three-seed bootstrap CI, Suff and Comp.")
    sub = parser.add_subparsers(dest="command", required=True)

    basic = sub.add_parser("metrics")
    basic.add_argument("--input", required=True)
    basic.add_argument("--output", required=True)

    boot = sub.add_parser("bootstrap")
    boot.add_argument("--inputs", nargs=3, required=True)
    boot.add_argument("--resamples", type=int, default=1000)
    boot.add_argument("--seed", type=int, default=42)
    boot.add_argument("--output", required=True)

    evidence = sub.add_parser("evidence")
    evidence.add_argument("--input", required=True)
    evidence.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "metrics":
        result = metrics(read_jsonl(args.input))
    elif args.command == "bootstrap":
        groups = [read_jsonl(path) for path in args.inputs]
        result = bootstrap_three_seed(groups, args.resamples, args.seed)
    else:
        result = suff_comp(read_jsonl(args.input))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
