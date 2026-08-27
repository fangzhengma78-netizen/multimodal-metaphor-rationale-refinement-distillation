# Dual-view rationale reproducibility package

本仓库是论文“双视角判别依据精炼蒸馏框架”的可核查复现材料包。它公开定稿中英文 Prompt、训练词表、数据规范、论文参数、质量诊断、文本探针、统计评价和学生训练入口。

## 方法顺序

图像与文本 → 隐式关联视角和显式对应视角独立生成候选依据 → 训练标签引导选择、去冲突和整合得到 A0 → 质量诊断 → LIRN → NOPAD → A1 → 以 `<rationale>A1</rationale><label>y</label><eos>` 监督学生模型。学生推理只接收图像与文本。

```text
for each generation stage:
    reject empty, missing-field, unparseable, or forbidden output
    retry at most two times
    if still invalid, use the previous valid stage result
    if no fallback exists, exclude from subsequent processing and log status
```

## 环境

```bash
python -m venv .venv
pip install -r requirements.txt
```


## 命令

```bash
python scripts/prepare_data.py --input /path/to/data.jsonl --output /path/to/output/data.jsonl --report /path/to/output/report.json
python scripts/build_cue_lexicon.py --input /path/to/train.jsonl --language zh --exclude-lexicon lexicons/cues/train/zh.txt --output /path/to/output/cues.json
python scripts/quality_diagnosis.py --input /path/to/rationales.jsonl --language zh --label-lexicon lexicons/labels/zh.txt --train-cues lexicons/cues/train/zh.txt --train-templates lexicons/templates/train/zh.txt --output /path/to/output/diagnosis.json
python scripts/text_probe.py --input /path/to/rationales.jsonl --rationale-field A1 --output /path/to/output/probe.json
python scripts/train_student.py --config configs/public_example.yaml --train-file /path/to/student_train.jsonl --dry-run
python scripts/evaluate_statistics.py metrics --input /path/to/predictions.jsonl --output /path/to/output/metrics.json
python -m pytest
```
