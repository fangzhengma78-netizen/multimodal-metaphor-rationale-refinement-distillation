import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_cue_lexicon import extract_ngrams, tokenize, weighted_log_odds_scores
from scripts.evaluate_statistics import bootstrap_indices, metrics, suff_comp
from scripts.output_guard import run_guarded
from scripts.prepare_data import normalize_label, normalize_language, validate_rows
from scripts.quality_diagnosis import lexical_hit, normalize, phrase_hit
from scripts.text_probe import grouped_split
from scripts.train_student import build_target


def row(i, image, split="train", language="en", label=0):
    return {"sample_id": str(i), "image_name": image, "image_path": f"{image}.jpg", "text": "x", "language": language, "label": label, "split": split}


def test_data_validation_and_normalization():
    output, report = validate_rows([row(1, "a", language="English", label="literal")])
    assert output[0]["language"] == "en" and output[0]["label"] == 0 and report["samples"] == 1
    assert normalize_language("中文") == "zh" and normalize_label("隐喻") == 1
    with pytest.raises(ValueError): validate_rows([row(1, "a", "train"), row(2, "a", "test")])


def test_ngrams_and_weighted_log_odds():
    assert extract_ngrams(["a", "b"]) == ["a", "b", "a b"]
    assert tokenize("A  B", "en") == ["a", "b"]
    scores = weighted_log_odds_scores({0: Counter({"literal": 1}), 1: Counter({"metaphor": 2})})
    assert scores["metaphor"] > 0 and scores["literal"] < 0


def test_normalization_and_hits():
    assert normalize("Ａ， B", "en") == "a b"
    assert lexical_hit("semantic mapping appears", ["semantic mapping"], "en")
    assert phrase_hit("This is direct support.", ["direct support"], "en")


def test_retry_and_fallback():
    value, record = run_guarded("1", "LIRN", lambda: "", "<rationale>", [], "previous")
    assert value == "previous" and record.retry_count == 2 and record.fallback_used


def test_group_split_has_no_image_leakage():
    rows = [{"image_name": f"image-{i}"} for i in range(100)]
    train, test = grouped_split(rows)
    assert train and test and not ({rows[i]["image_name"] for i in train} & {rows[i]["image_name"] for i in test})


def test_metrics_bootstrap_and_evidence():
    report = metrics([{"label": 0, "prediction": 0}, {"label": 1, "prediction": "<label>Metaphor</label>"}])
    assert report["accuracy"] == report["macro_f1"] == report["weighted_f1"] == 1.0
    assert (bootstrap_indices(4, 3, 42) == bootstrap_indices(4, 3, 42)).all()
    result = suff_comp([{"p_full": .8, "p_evidence_only": .6, "p_evidence_removed": .3}])
    assert result["sufficiency"] == pytest.approx(.2) and result["comprehensiveness"] == pytest.approx(.5)


def test_target_and_dry_run(tmp_path):
    assert build_target("evidence", 1) == "<rationale>evidence</rationale><label>Metaphor</label><eos>"
    root = Path(__file__).parents[1]
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"sample_id": "1", "image_path": "x.jpg", "text": "x", "A1": "evidence", "label": 1}) + "\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(root / "scripts/train_student.py"), "--config", str(root / "configs/public_example.yaml"), "--train-file", str(data), "--dry-run"], capture_output=True, text=True, check=True)
    assert "dry_run_ok" in result.stdout
