from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalize(text: str, language: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    if language == "en":
        text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, language: str) -> list[str]:
    value = normalize(text, language)
    if language == "zh":
        import jieba
        sequence = jieba.lcut(value)
    else:
        sequence = value.split()
    return [x for x in sequence if x.strip()]


def feature_set(text: str, language: str) -> set[str]:
    seq = tokens(text, language)
    return set(seq) | {f"{seq[i]} {seq[i + 1]}" for i in range(len(seq) - 1)}


def read_lexicon(path: str | None) -> list[str]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8-sig")
    return [x.strip() for x in re.split(r"[\n、]", text) if x.strip()]


def lexical_hit(text: str, lexicon: list[str], language: str) -> bool:
    feats = feature_set(text, language)
    lexicon_features = set()
    for item in lexicon:
        item_tokens = tokens(item, language)
        if item_tokens:
            lexicon_features.add(" ".join(item_tokens))
    return bool(feats & lexicon_features)


def phrase_hit(text: str, phrases: list[str], language: str) -> bool:
    normalized_text = normalize(text, language)
    normalized_phrases = [normalize(phrase, language) for phrase in phrases]
    return any(phrase in normalized_text for phrase in normalized_phrases if phrase)


def diagnose(
    rows: list[dict],
    field: str,
    language: str,
    labels: list[str],
    train_cues: list[str],
    external_cues: list[str],
    train_templates: list[str],
    external_templates: list[str],
) -> dict:
    texts = [str(row.get(field, "")) for row in rows if str(row.get(field, "")).strip()]

    def rate(test) -> float:
        return sum(test(text) for text in texts) / len(texts) if texts else 0.0

    return {
        "field": field,
        "valid_samples": len(texts),
        "label_leakage_rate": rate(lambda x: lexical_hit(x, labels, language)),
        "train_cue_overlap_rate": rate(lambda x: lexical_hit(x, train_cues, language)),
        "external_cue_overlap_rate": rate(lambda x: lexical_hit(x, external_cues, language)),
        "train_template_hit_rate": rate(lambda x: phrase_hit(x, train_templates, language)),
        "external_template_hit_rate": rate(lambda x: phrase_hit(x, external_templates, language)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare A0 and A1 quality-risk indicators.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--language", required=True, choices=["zh", "en"])
    parser.add_argument("--label-lexicon", required=True)
    parser.add_argument("--train-cues", required=True)
    parser.add_argument("--train-templates", required=True)
    parser.add_argument("--external-cues")
    parser.add_argument("--external-templates")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    resource_paths = (
        args.label_lexicon,
        args.train_cues,
        args.external_cues,
        args.train_templates,
        args.external_templates,
    )
    resources = [read_lexicon(path) for path in resource_paths]
    report = {field: diagnose(rows, field, args.language, *resources) for field in ("A0", "A1")}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
