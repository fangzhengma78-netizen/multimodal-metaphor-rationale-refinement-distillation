# 数据处理

沿用 MetMeme 已有训练、验证、测试划分，不重新随机划分。先统一语言为 `zh/en`、标签为 `0/1`，检查 `sample_id` 唯一，并确保同一 `image_name` 不跨划分。中文答案线索构造使用 jieba，英文按空白分词并小写化，提取1–2 gram。

对特征 *w*，以两类 A0 的合并词频作为信息型 Dirichlet 先验 \(\alpha_w\)，\(\alpha_0=\sum_w\alpha_w\)。设两类中的特征频次和总特征数分别为 \(y_{1w},n_1\) 与 \(y_{0w},n_0\)，计算：

`delta_w = log((y_1w + alpha_w)/(n_1 + alpha_0 - y_1w - alpha_w)) - log((y_0w + alpha_w)/(n_0 + alpha_0 - y_0w - alpha_w))`

`z_w = delta_w / sqrt(1/(y_1w + alpha_w) + 1/(y_0w + alpha_w))`

按 `abs(z_w)` 降序选取 Top200；正值方向为 Metaphor，负值方向为 Literal。该实现采用 Monroe、Colaresi 和 Quinn 的 informative-Dirichlet weighted log-odds 统计量（DOI: 10.1093/pan/mpn018）。独立评价词表由验证集 A0 按相同规则构造，并在规范化后排除训练词表重合项；该词表不随仓库公开。

教师侧记录可包含双视角候选、A0、A1及输出状态。学生目标固定为 `<rationale>A1</rationale><label>y</label><eos>`。外部评价文件格式见 `data_schema.md`。
