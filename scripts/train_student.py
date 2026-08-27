from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_NAMES = {0: "Literal", 1: "Metaphor"}
FIELDS = ("sample_id", "image_path", "text", "A1", "label")


def build_target(a1: str, label: int) -> str:
    if label not in LABEL_NAMES or not str(a1).strip():
        raise ValueError("A1 must be non-empty and label must be 0 or 1")
    return f"<rationale>{str(a1).strip()}</rationale><label>{LABEL_NAMES[label]}</label><eos>"


def validate_config(config: dict) -> None:
    expected = {"precision": "bf16", "optimizer": "AdamW", "learning_rate": 2e-4, "per_device_batch_size": 1, "gradient_accumulation_steps": 16, "epochs": 1, "max_sequence_length": 1024}
    for key, value in expected.items():
        if config["student_training"].get(key) != value:
            raise ValueError(f"paper setting mismatch: {key}")
    if config["student_training"].get("seed") not in {42, 43, 44}:
        raise ValueError("seed must be 42, 43, or 44")


def read_training_rows(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for index, row in enumerate(rows, 1):
        missing = [field for field in FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index} missing fields: {missing}")
        row["target"] = build_target(str(row["A1"]), int(row["label"]))
    return rows


def run_training(config: dict, rows: list[dict]) -> None:
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, default_data_collator, set_seed

    settings = config["student_training"]
    set_seed(settings["seed"])
    processor = AutoProcessor.from_pretrained(config["student_model"], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        config["student_model"], torch_dtype=torch.bfloat16, local_files_only=True
    )
    lora = settings["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules="all-linear", task_type="CAUSAL_LM"
    ))
    encoded = []
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(row["text"])}]}]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": row["target"]}]}]
        prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        batch = processor(text=[full_text], images=[image], return_tensors="pt", truncation=True, max_length=settings["max_sequence_length"])
        prompt = processor(text=[prompt_text], images=[image], return_tensors="pt", truncation=True, max_length=settings["max_sequence_length"])
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, : prompt["input_ids"].shape[1]] = -100
        encoded.append({key: value.squeeze(0) for key, value in batch.items()})
    arguments = TrainingArguments(
        output_dir=config["output_dir"], bf16=True, optim="adamw_torch",
        learning_rate=settings["learning_rate"], per_device_train_batch_size=settings["per_device_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"], num_train_epochs=settings["epochs"],
        seed=settings["seed"], save_strategy="no", report_to=[]
    )
    Trainer(model=model, args=arguments, train_dataset=encoded, data_collator=default_data_collator).train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal auditable Transformers/PEFT LoRA student-training entry.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    rows = read_training_rows(args.train_file)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "samples": len(rows), "model": config["student_model"], "seed": config["student_training"]["seed"], "target_format": "<rationale>A1</rationale><label>y</label><eos>"}, ensure_ascii=False))
        return
    run_training(config, rows)


if __name__ == "__main__":
    main()
