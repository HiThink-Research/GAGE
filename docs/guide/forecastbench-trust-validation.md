# ForecastBench 接入可信度验证方案

本文档说明如何验证 GAGE 接入 ForecastBench Polymarket 静态评测后，框架本身和评测产物是否可信。

这里要验证两件事：

- **框架可信度**：GAGE 是否正确读取 ForecastBench 数据、join 题目和答案、计算 Brier 等指标。
- **产物可信度**：一次模型评测 run 的产物是否完整、可复查、可解释。

不建议直接用「GAGE 跑出来的模型分数」和 ForecastBench 官网 leaderboard 分数硬比。官网 leaderboard 使用 difficulty-adjusted Brier、跨 dataset / market 聚合、置信区间、提交规则等完整机制；当前 GAGE P0 接的是 `source=polymarket` 且 `resolved=true` 的静态子集，口径不同。

## 当前接入范围

当前 GAGE 只接 ForecastBench 中的 Polymarket 已结算 market questions：

```yaml
source_filter:
  - polymarket
resolved_only: true
question_type: market
```

本机数据目录：

```text
E:\Developing\GAGE\.gage_cache\forecastbench-datasets
```

本机已跑产物目录：

```text
E:\fb_runs
```

当前全量 Polymarket resolved 子集为 `964` 条，分布在 17 个 `question_set` 日期切片中。正式可用 run 目录为 `E:\fb_runs\fbm-*`。其中 `fbm-2025-03-30-r2` 是临时验证目录，不计入正式总量。

## 总体验证链路

```mermaid
flowchart TD
    A["第一层<br/>数据 join 正确性"] --> B["第二层<br/>官方 forecast_set 回放"]
    B --> C["第三层<br/>官方框架同模型对照"]
    C --> D["产物可信度检查<br/>summary / samples / trace"]

    A1["检查样本数<br/>question + resolution"] --> A
    B1["跳过模型调用<br/>直接回放官方预测"] --> B
    C1["同模型同题集<br/>比较 forecast 分布"] --> C
    D1["确认 run 完整<br/>可追溯到单 case"] --> D
```

## 第一层：数据 Join 正确性

这一层验证 GAGE 是否正确读取 ForecastBench 的题目和答案。

输入文件：

```text
datasets/question_sets/<DATE>-llm.json
datasets/resolution_sets/<DATE>_resolution_set.json
```

验证规则：

```text
1. 读取 question_set 中的 questions
2. 读取 resolution_set 中的 resolutions
3. 按 id join question 和 resolution
4. 只保留 source == "polymarket"
5. 只保留 resolved == true
6. 统计每个日期切片样本数
```

验收标准：

```text
每个 question_set 的 Polymarket resolved 样本数稳定
question.id 和 resolution.id join 无异常
resolved_to 字段存在且可用于评分
freeze_datetime_value 覆盖率可解释
```

当前本机统计结果：

```text
TOTAL 964
2024-07-21 119
2025-03-02 82
2025-03-16 73
2025-03-30 72
2025-04-13 79
2025-04-27 87
2025-05-11 60
2025-05-25 59
2025-06-08 67
2025-06-22 63
2025-08-03 29
2025-08-17 43
2025-08-31 44
2025-10-26 45
2025-11-09 16
2025-11-23 14
2025-12-07 12
```

注意：`freeze_datetime_value` 不是每条样本都有。例如 `2024-07-21` 有 119 条 resolved Polymarket 样本，但只有 59 条有可用的 freeze market value。因此 `average_market_baseline_brier` 只覆盖这 59 条，不能直接代表 119 条全量市场基线。

## 第二层：评分器正确性

这一层是最关键的框架校验。

目标是验证：

```text
同一个 question_set
同一个 resolution_set
同一个 forecast_set
```

GAGE 和官方框架算出来的分数是否一致。

这一步不调用模型。它验证的是 GAGE 的 loader、join、parser、metric、aggregator 是否对齐官方口径。

### 什么是 forecast_set

`forecast_set` 可以理解成某个模型交出的预测答案。它通常包含：

```json
{
  "organization": "...",
  "model": "...",
  "question_set": "2025-03-02-llm.json",
  "forecasts": [
    {
      "id": "0x...",
      "source": "polymarket",
      "forecast": 0.42,
      "resolution_date": null
    }
  ]
}
```

官方 ForecastBench 数据页公开 `forecast sets` 和 `processed forecast sets`。这类文件最适合做评分回放校验。

### Replay 模式

```mermaid
flowchart TD
    Q["官方题目<br/>Question Set"] --> J["按 id 关联"]
    R["官方答案<br/>Resolution Set"] --> J
    F["官方预测<br/>Forecast Set"] --> J
    J --> S["单样本评分<br/>Brier"]
    S --> A["聚合评分<br/>average_brier / Brier Index"]
    A --> O["GAGE replay 结果<br/>summary.json"]
    P["官方 processed forecast<br/>或官方结果"] --> C["逐项对比"]
    O --> C
```

Replay 模式要求：

```text
跳过 inference
直接读取 forecast_set.forecasts[*].forecast
按 id 找到 resolution.resolved_to
逐条计算 brier = (forecast - resolved_to)^2
聚合 average_brier
计算 brier_index_simple = (1 - sqrt(average_brier)) * 100
```

验收标准：

```text
sample_count 完全一致
单样本 forecast 完全一致
单样本 resolved_to 完全一致
单样本 brier 逐条一致
run 级 average_brier 误差 < 1e-6
run 级 brier_index_simple 误差 < 1e-6
```

如果这一层通过，就可以说明：

```text
GAGE 的 ForecastBench 静态评分链路可信
```

如果这一层不通过，优先排查：

```text
id join 是否错位
forecast 是否 parse 错
resolved_to 是否读反
缺失 forecast 是否按官方规则处理
聚合样本集合是否一致
```

## 第三层：官方框架同模型对照

这一层验证的是模型调用链路，而不是纯评分器。

做法是用 ForecastBench 官方框架和 GAGE 分别调用同一个模型、同一个 question_set，再比较输出差异。

```mermaid
flowchart TD
    Q["同一个 question_set"] --> OF["官方 ForecastBench 框架"]
    Q --> GF["GAGE 框架"]

    M["同一个模型<br/>同一个 API endpoint"] --> OF
    M --> GF

    OF --> O1["官方生成<br/>forecast_set"]
    GF --> G1["GAGE 生成<br/>forecast_set / samples.jsonl"]

    O1 --> R1["GAGE replay 官方 forecast_set"]
    R1 --> S1["验证评分器一致"]

    O1 --> C["比较模型输出"]
    G1 --> C
    C --> D["forecast 分布<br/>parse_error_rate<br/>average_brier"]
```

需要尽量对齐的参数：

```text
官方 market prompt
输出格式 *0.xxx*
是否带 freeze value
是否启用 reformat prompt
temperature = 0
max tokens
thinking 是否关闭
模型名称和模型版本
API base URL
```

这一层不要强求逐条预测完全一致。即使 `temperature=0`，不同框架在 prompt 包装、stop 参数、reformat、token 限制、供应商实现上有细微差异，也可能导致预测不同。

合理验收标准：

```text
parse_error_rate 接近
forecast 分布大体一致
average_brier 同数量级
关键 case 差异可解释
```

如果官方框架生成的 forecast_set 被 GAGE replay 后分数一致，但 GAGE 自己调用模型生成的 forecast_set 差异较大，说明问题主要在：

```text
prompt / inference 参数 / 输出解析 / 模型服务行为
```

而不是 GAGE 的 scorer。

## 产物可信度检查

每个正式 run 至少检查这些文件：

```text
E:\fb_runs\<RUN_ID>\summary.json
E:\fb_runs\<RUN_ID>\samples.jsonl
E:\fb_runs\<RUN_ID>\samples\task_forecastbench_polymarket_static_full\*.json
```

`summary.json` 重点检查：

```text
tasks[0].execution.status == "completed"
sample_count == tasks[0].sample_count
samples_valid == samples_total
parse_error_rate 是否可接受
metrics[0].count 是否合理
market_baseline_samples 覆盖率是否解释清楚
```

`samples.jsonl` 重点检查：

```text
每条样本有 sample_id
每条样本有 model_output
每条样本有 metrics.forecastbench_probability
forecast / resolved_to / brier 可追溯
```

单样本 JSON 重点检查：

```text
sample.messages[0] 是否为实际发给模型的 prompt
model_output.answer 是否为模型原始输出
raw_response 是否保存服务商返回内容
metrics.forecastbench_probability.values 是否包含单样本评分
```

单样本指标解释：

```text
forecast: 模型预测概率
resolved_to: 最终结算结果，通常 0 或 1
brier: (forecast - resolved_to)^2
brier_index_simple_case: 单 case 简化 Brier Index
accuracy_at_0_5: 概率按 0.5 阈值转 Yes/No 后是否命中
parse_error: 输出是否解析失败
clamp_applied: 预测值是否被截断到 [0, 1]
market_baseline_brier: freeze market value 的 Brier，仅有 freeze value 时存在
model_minus_market_brier: 模型 Brier 减市场基线 Brier
```

## 当前产物的可信度边界

当前本机 `E:\fb_runs\fbm-*` 产物可以用于：

```text
验证 GAGE 能跑通 ForecastBench Polymarket resolved 子集
分析 Xiaomi Mimo 在该子集上的 simple Brier 表现
逐 case 查看 prompt、模型输出和评分
比较模型预测和 freeze market baseline 的差异
```

当前产物不能直接用于：

```text
声称复现 ForecastBench 官方 leaderboard
声称模型在完整 ForecastBench 上的官方排名
声称 difficulty-adjusted Brier / CI / p-value 已经对齐官方
```

## 推荐落地顺序

1. 固化第一层数据统计脚本，作为 loader 回归测试。
2. 实现 `forecast_set replay` 模式，跳过 inference 直接评分官方 forecast_set。
3. 用官方 processed forecast set 做 parity test，确认 scorer 误差在 `1e-6` 内。
4. 用官方框架和 GAGE 跑同一个模型、同一个 question_set，比较 forecast 分布和 parse_error_rate。
5. 再决定是否补 difficulty-adjusted leaderboard、bootstrap CI、p-value 等官方高阶指标。

## 参考链接

- ForecastBench 官网数据页：https://www.forecastbench.org/datasets/
- ForecastBench Leaderboards：https://www.forecastbench.org/leaderboards/
- ForecastBench GitHub：https://github.com/forecastingresearch/forecastbench
