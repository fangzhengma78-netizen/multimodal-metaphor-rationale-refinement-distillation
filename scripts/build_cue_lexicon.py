from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

def tokenize(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    if language == "zh":
        import jieba
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    if language == "en":
        return text.lower().split()
    raise ValueError("language must be zh or en")


def extract_ngrams(tokens: list[str]) -> list[str]:
    return tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def normalize_feature(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip().lower())


def collect_counts(rows: list[dict], language: str, text_field: str, label_field: str) -> dict[int, Counter]:
    counts = {0: Counter(), 1: Counter()}
    for row in rows:
        label = int(row[label_field])
        if label not in counts:
            raise ValueError("labels must be 0 or 1")
        counts[label].update(extract_ngrams(tokenize(str(row[text_field]), language)))
    return counts


def weighted_log_odds_scores(counts: dict[int, Counter]) -> dict[str, float]:
    """Monroe et al. informative-Dirichlet weighted log-odds z scores.

    The prior vector is the pooled 1--2 gram count vector of the input A0
    collection. Positive values indicate stronger association with label 1;
    negative values indicate stronger association with label 0.
    """
    import math

    prior = counts[0] + counts[1]
    prior_total = sum(prior.values())
    totals = {label: sum(counter.values()) for label, counter in counts.items()}
    scores: dict[str, float] = {}
    for feature, alpha_feature in prior.items():
        left = counts[1][feature] + alpha_feature
        right = counts[0][feature] + alpha_feature
        left_other = totals[1] + prior_total - left
        right_other = totals[0] + prior_total - right
        delta = math.log(left / left_other) - math.log(right / right_other)
        variance = (1.0 / left) + (1.0 / right)
        scores[feature] = delta / math.sqrt(variance)
    return scores


def read_lexicon(path: str | None) -> set[str]:
    if not path:
        return set()
    source = Path(path)
    if source.suffix.lower() == ".json":
        values = json.loads(source.read_text(encoding="utf-8"))
    else:
        values = re.split(r"[\n、]", source.read_text(encoding="utf-8-sig"))
    return {normalize_feature(str(value)) for value in values if str(value).strip()}


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1-2 gram weighted-log-odds cue lexicons.")
    parser.add_argument("--input", required=True, help="Training/validation JSONL containing A0 and labels")
    parser.add_argument("--language", required=True, choices=["zh", "en"])
    parser.add_argument("--text-field", default="A0")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--exclude-lexicon")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    counts = collect_counts(read_jsonl(args.input), args.language, args.text_field, args.label_field)
    excluded = read_lexicon(args.exclude_lexicon)
    scores = weighted_log_odds_scores(counts)
    ranked = sorted(
        ((feature, score) for feature, score in scores.items() if normalize_feature(feature) not in excluded),
        key=lambda item: (-abs(item[1]), item[0]),
    )[: args.top_k]
    records = [
        {
            "feature": feature,
            "language": args.language,
            "class_direction": "metaphor" if score > 0 else "literal",
            "score": score,
            "rank": rank,
            "frequency": counts[0][feature] + counts[1][feature],
        }
        for rank, (feature, score) in enumerate(ranked, 1)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
