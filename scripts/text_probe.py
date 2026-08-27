from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def is_train_image(image_name: str) -> bool:
    digest = hashlib.sha256(str(image_name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100 < 80


def grouped_split(rows: list[dict]) -> tuple[list[int], list[int]]:
    train = [i for i, row in enumerate(rows) if is_train_image(str(row["image_name"]))]
    test = [i for i, row in enumerate(rows) if not is_train_image(str(row["image_name"]))]
    if not train or not test:
        raise ValueError("deterministic image_name split produced an empty partition")
    if {rows[i]["image_name"] for i in train} & {rows[i]["image_name"] for i in test}:
        raise AssertionError("image leakage between probe partitions")
    return train, test


def fit_probe(train_text: list[str], train_y: list[int], test_text: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=20000)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    return LogisticRegression(max_iter=1000, random_state=42).fit(x_train, train_y).predict(x_test)


def run(rows: list[dict], field: str) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score

    train, test = grouped_split(rows)
    texts = [str(row[field]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    gold = [labels[i] for i in test]
    majority = Counter(labels[i] for i in train).most_common(1)[0][0]
    rng = np.random.default_rng(42)
    shuffled_texts = list(texts)
    rng.shuffle(shuffled_texts)
    shuffled_labels = [labels[i] for i in train]
    rng.shuffle(shuffled_labels)

    train_texts = [texts[i] for i in train]
    test_texts = [texts[i] for i in test]
    train_labels = [labels[i] for i in train]
    shuffled_train_texts = [shuffled_texts[i] for i in train]
    shuffled_test_texts = [shuffled_texts[i] for i in test]

    probe_prediction = fit_probe(train_texts, train_labels, test_texts)
    random_rationale_prediction = fit_probe(
        shuffled_train_texts,
        train_labels,
        shuffled_test_texts,
    )
    label_shuffle_prediction = fit_probe(train_texts, shuffled_labels, test_texts)

    return {
        "train_samples": len(train),
        "test_samples": len(test),
        "probe_accuracy": float(accuracy_score(gold, probe_prediction)),
        "majority_accuracy": float(accuracy_score(gold, [majority] * len(test))),
        "random_rationale_accuracy": float(accuracy_score(gold, random_rationale_prediction)),
        "label_shuffle_accuracy": float(accuracy_score(gold, label_shuffle_prediction)),
        "random_seed": 42,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rationale-only TF-IDF character n-gram probe.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--rationale-field", default="A1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run(rows, args.rationale_field), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
