# 数据字段

| 字段 | 用途 |
|---|---|
| `sample_id` | 全局唯一样本标识 |
| `image_name`、`image_path` | 图像标识与用户本地路径 |
| `text`、`language`、`label` | 文本、`zh/en`、`0字面/1隐喻` |
| `split` | 沿用原数据的 train/validation/test |
| `implicit_rationale`、`explicit_rationale` | 教师侧双视角候选 |
| `A0`、`A1` | 原始与精炼判别依据 |
| `status`、`retry_count`、`error_type` | 输出守卫记录 |

学生训练使用图像、文本、A1和标签。学生推理只能使用图像与文本，不能接收标签、候选依据、A0或A1。评价预测文件至少包含 `sample_id`、`label`、`prediction`。

