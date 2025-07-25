# HuggingGPT Evaluation Framework

A complete evaluation framework for HuggingGPT that supports unified evaluation across multiple datasets, including math, reasoning, code generation, and more.

## 🌟 Features

- **Unified Evaluation Interface**: Consistent evaluation for multiple datasets
- **Prompt Formatting System**: Automatically apply the correct output format for each dataset
- **Modular Design**: Easily extendable to new datasets and evaluators
- **Detailed Reports**: Includes accuracy, processing time, and detailed results
- **Command-Line Interface**: Convenient CLI tools for evaluation
- **Batch Evaluation**: Support for evaluating multiple datasets in one run

## 📊 Supported Datasets

| Dataset     | Type            | Description                      | Samples |
|-------------|-----------------|----------------------------------|---------|
| MATH        | Math            | Competition math problems        | 487     |
| AIME        | Math            | American Invitational Math Exam  | 31      |
| BBH         | Reasoning       | Big-Bench Hard reasoning tasks   | 541     |
| DROP        | Reading/QA      | Discrete reading comprehension   | 401     |
| HumanEval   | Code Generation | Python code generation tasks     | 132     |
| MMLU-Pro    | Knowledge       | Multidisciplinary QA             | 401     |

## 🚀 Quick Start

### Install Dependencies

```bash
pip install langchain openai transformers torch
# Optional dependencies
pip install wolframalpha duckduckgo-search
```

### Configure Environment Variables

```bash
# OpenAI API
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"

# Azure OpenAI (optional)
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_VERSION="2023-05-15"

# Wolfram Alpha (optional, for math computing)
export WOLFRAM_ALPHA_APPID="your-app-id"
```

### Basic Usage

#### 1. List Available Datasets

```bash
python run_evaluation.py --list-datasets
```

#### 2. Get Dataset Information

```bash
python run_evaluation.py --info --dataset math
```

#### 3. Evaluate a Single Dataset

```bash
# Evaluate the Math dataset (first 10 samples)
python run_evaluation.py --dataset math --max-samples 10

# Evaluate BBH dataset and save to a custom directory
python run_evaluation.py --dataset bbh --max-samples 20 --output-dir results/bbh

# Evaluate DROP dataset with detailed logging
python run_evaluation.py --dataset drop --max-samples 15 --log-level DEBUG
```

## 🛠️ Programmatic Usage

### Example

```python
from evaluation_framework import EvaluationManager, EvaluationConfig
from config import create_llm_from_config, create_default_tools, DEFAULT_CONFIG

# Create the LLM and tools
llm = create_llm_from_config(DEFAULT_CONFIG.llm_config, "openai")
tools = create_default_tools()

# Initialize the evaluation manager
manager = EvaluationManager(llm, tools)

# Prepare configuration
config = EvaluationConfig(
    dataset_name="math",
    data_path="hugginggpt/dataset/math_test.jsonl",
    max_samples=10,
    save_intermediate=True,
    output_dir="results"
)

# Run evaluation
result = manager.evaluate_dataset(config)
print(f"Accuracy: {result.accuracy:.3f}")
print(f"Correct samples: {result.correct_samples}/{result.total_samples}")
```

### Batch Evaluation

```python
# Evaluate all supported datasets
datasets = manager.get_supported_datasets()

for name in datasets:
    cfg = EvaluationConfig(
        dataset_name=name,
        data_path=f"hugginggpt/dataset/{name}_test.jsonl",
        max_samples=5,
        output_dir=f"results/{name}"
    )
    res = manager.evaluate_dataset(cfg)
    print(f"{name}: {res.accuracy:.3f}")
```

## 🔧 Custom Evaluators

Extend the framework to support new datasets:

```python
from evaluation_framework import BaseUnifiedEvaluator

class CustomEvaluator(BaseUnifiedEvaluator):
    def _get_base_input(self, item):
        # Define how to format the input
        return f"Question: {item['question']}"

    def extract_answer(self, response):
        # Define how to extract the answer
        import re
        match = re.search(r'Answer[: ](.+)', response)
        return match.group(1).strip() if match else ""

    def check_correctness(self, prediction, ground_truth):
        # Define correctness check
        return prediction.strip() == ground_truth.strip()

    def evaluate_single(self, item):
        # Implement evaluation logic
        pass

# Register the custom evaluator
EvaluationManager.EVALUATOR_MAPPING['custom_dataset'] = CustomEvaluator
```

## 📁 Project Structure

```
hugginggpt/
├── evaluation_framework.py   # Core evaluation framework
├── run_evaluation.py        # CLI entry point
├── config.py                # Configuration management
├── format_prompts.py        # Prompt formatting
├── examples.py              # Usage examples
├── hugginggpt.py            # Core HuggingGPT class
├── dataset/                 # Dataset files
│   ├── math_test.jsonl
│   ├── bbh_test.jsonl
│   └── ...
├── evaluators/              # Evaluator modules
└── results/                 # Evaluation results
```

## 🎯 Evaluation Metrics

The framework provides the following metrics:

- **Accuracy**: Proportion of correct answers
- **Average Time**: Average processing time per sample
- **Detailed Results**: Prediction, ground truth, and full response for each sample
- **Error Analysis**: Error details for failed samples

## 📋 Dataset Formats

### MATH Format

```json
{
  "problem": "Find the smallest positive integer...",
  "level": "Level 5",
  "type": "Prealgebra",
  "solution": "Two numbers are relatively prime if..."
}
```

### BBH Format

```json
{
  "id": "task_1",
  "input": "Question about...",
  "target": "(A)"
}
```

### DROP Format

```json
{
  "query_id": "query_1",
  "passage": "Text passage...",
  "question": "Based on the passage...",
  "answers_spans": {"spans": ["answer1", "answer2"]}
}
```

## 🔧 Configuration Options

### LLM Config

```python
from config import LLMConfig

llm_config = LLMConfig(
    model_name="gpt-4",
    temperature=0.0,
    max_tokens=4096,
    timeout=120,
    api_key="your-key",
    api_base="your-base-url"
)
```

### Evaluation Config

```python
from evaluation_framework import EvaluationConfig

eval_config = EvaluationConfig(
    dataset_name="math",
    data_path="path/to/data.jsonl",
    max_samples=100,
    batch_size=1,
    save_intermediate=True,
    output_dir="results",
    log_level="INFO",
    timeout_seconds=60
)
```

## 📈 Output Results

Evaluation outputs include:

1. **Console Output**: Real-time progress and summary
2. **Detailed Results File**: JSON with full results
3. **Summary File**: JSONL summary

### Example Output

```json
{
  "dataset_name": "math",
  "total_samples": 10,
  "correct_samples": 7,
  "accuracy": 0.7,
  "average_time": 2.5,
  "timestamp": "2024-01-01 12:00:00",
  "detailed_results": [
    {
      "item_id": "1",
      "problem": "Find the smallest positive integer...",
      "predicted_answer": "42",
      "ground_truth": "42",
      "correct": true,
      "evaluation_time": 2.1
    }
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## ❓ FAQ

**Q: How to add a new dataset?**

A:
1. Add dataset mapping to `DatasetLoader.DATASET_PATHS`
2. Implement an evaluator extending `BaseUnifiedEvaluator`
3. Register it in `EvaluationManager.EVALUATOR_MAPPING`
4. Add a prompt format in `format_prompts.py` if needed

**Q: How to use different LLMs?**

A: Modify `config.py` or use `create_llm_from_config` for custom LLMs

**Q: Where are results saved?**

A: By default in `results/`, configurable via `--output-dir` or `EvaluationConfig.output_dir`

## 📞 Support

If you need help:

1. Check documentation and examples
2. Search GitHub Issues
3. Open a new Issue with details

---

**Happy Evaluating! 🎉** 