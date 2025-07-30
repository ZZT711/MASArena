# BBH (Big-Bench Hard) 多智能体辩论测试

## 简介

这个目录包含了针对BBH数据集的多智能体辩论测试实现，参考了MMLU的实现形式，保持了高度一致的逻辑结构。

## 文件说明

- `gen_bbh.py`: 生成BBH数据集的多智能体辩论结果
- `eval_bbh.py`: 评估BBH辩论结果的准确率
- `README.md`: 本说明文件

## 使用方法

### 1. 生成BBH辩论结果

```bash
cd llm_multiagent_debate/bbh
python gen_bbh.py
```

这将：
- 从 `../dataset/bbh_test.jsonl` 加载BBH测试数据
- 选择前100个问题进行测试
- 使用3个智能体进行2轮辩论
- 异步并行处理以提高效率
- 将结果保存到 `bbh_3_2.json`

### 2. 评估结果

```bash
python eval_bbh.py bbh_3_2.json
```

这将：
- 加载生成的辩论结果
- 使用BBH evaluator的逻辑进行评估
- 按任务类型统计准确率
- 保存详细评估结果到 `bbh_3_2_evaluation_results.json`

## 配置参数

### 在 `gen_bbh.py` 中可以调整的参数：

- `num_questions = 100`: 测试问题数量
- `agents = 3`: 智能体数量
- `rounds = 2`: 辩论轮数
- `batch_size = 5`: 批处理大小

### API 配置：

请确保在项目根目录有 `.env` 文件，包含必要的API配置。

## BBH任务类型支持

该实现支持以下BBH任务类型：
- `boolean_expressions`: 布尔表达式计算
- `causal_judgement`: 因果判断
- `date_understanding`: 日期理解
- `disambiguation_qa`: 歧义消解
- `dyck_languages`: Dyck语言（括号匹配）
- `formal_fallacies`: 形式逻辑谬误
- `geometric_shapes`: 几何形状识别
- `hyperbaton`: 形容词语序
- `logical_deduction`: 逻辑推理

## 评估逻辑

评估使用了 `evaluators/bbh_evaluator.py` 中的逻辑：
- 优先查找 `<answer></answer>` 标签中的答案
- 支持多种答案格式的提取和标准化
- 针对不同任务类型使用相应的评估策略
- 使用多智能体答案的最频繁值作为最终答案

## 输出文件

- `bbh_3_2.json`: 包含所有问题的辩论过程和答案
- `bbh_3_2_evaluation_results.json`: 详细的评估结果，包括总体准确率和分任务统计

## 异步并发优化

- 使用 `asyncio` 实现异步处理
- 批处理机制避免API限制
- 进度条显示处理进度
- 错误重试机制保证稳定性

## 注意事项

1. 确保有足够的API配额进行100个问题的测试
2. 根据API限制调整 `batch_size` 参数
3. BBH数据集需要在 `../dataset/bbh_test.jsonl` 路径下
4. 评估需要项目根目录下的 `evaluators` 模块 