# ChatDev HumanEval Evaluation System

This system evaluates ChatDev's performance on the HumanEval benchmark.

## System Architecture

### Core Components

1. **data_structures.py** - Defines data structures
   - `HumanevalTask`: Single test task
   - `EvaluationResult`: Single task evaluation result
   - `BenchmarkSummary`: Overall evaluation summary

2. **run_benchmark.py** - Main evaluation script
   - Randomly selects a specified number of HumanEval problems
   - Generates code for each problem using ChatDev
   - Immediately evaluates and aggregates results

3. **humaneval_evaluator.py** - Evaluation logic (existing, not modified)
   - Executes generated code
   - Runs test cases
   - Returns evaluation results

4. **test_benchmark.py** - Test script
   - Small-scale test to verify the system is working

## Data Flow

```
HumanEval dataset → Randomly select tasks → ChatDev generates code → Evaluator runs tests → Collect results
     ↓                ↓                ↓              ↓           ↓
humaneval_test.jsonl → HumanevalTask → codebooks → HumanEvalEvaluator → BenchmarkSummary
```

## Usage

### 1. Environment Setup

Ensure necessary dependencies are installed:
```bash
pip install -r requirements.txt
```

Set the OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Test the System

First run a small-scale test to ensure the system works:
```bash
cd ChatDev-main/Humaneval_benchmark
python test_benchmark.py
```

### 3. Run Full Evaluation

Run the full evaluation with 100 tasks:
```bash
python run_benchmark.py --num_tasks 100
```

Or specify a different number:
```bash
python run_benchmark.py --num_tasks 50  # Test 50 tasks
```

### 4. View Results

After completion, the system outputs:
- Real-time progress and final statistics in the console
- Detailed results saved in `benchmark_results.json`

## Output Format

### Console Output (simplified):

```
Processing task 1/100: HumanEval_84
  Generating code...
  Evaluating...
  1. HumanEval_84: Pass
     Pass rate: 100.0%
     Time: 45.2s

=== HumanEval Evaluation Results ===
Total tasks: 100
Tasks fully passed: 65
Task pass rate: 65.00%
Average test-case pass rate: 65.00%
Total execution time: 4520.50s
Average time per task: 45.21s
```

### JSON Results File

```json
{
  "summary": {
    "total_tasks": 100,
    "passed_tasks": 65,
    "accuracy": 0.65,
    "avg_test_accuracy": 0.65,
    "total_execution_time": 4520.5
  },
  "details": [
    {
      "task_id": "HumanEval_84",
      "success": true,
      "passed_count": 1,
      "total_count": 1,
      "accuracy": 1.0,
      "execution_time": 45.2,
      "code_generated": "def solve(N):\n    ..."
    }
  ]
}
```

## Features

1. **Random Selection**: Uses a fixed random seed (42) to ensure reproducibility
2. **Isolated Processing**: Each task is run independently by ChatDev to avoid state interference
3. **Immediate Evaluation**: Code is evaluated immediately after generation to save memory
4. **Detailed Logging**: Each task's ChatDev execution process is logged separately
5. **Error Handling**: Both generation and evaluation failures are handled gracefully

## API Interfaces

### ChatDev Interface

```python
# Input: HumanevalTask.prompt (problem description)
# Output: chat_env.codes.codebooks (code dictionary)
chat_chain = ChatChain(task_prompt=task.prompt, ...)
chat_chain.execute_chain()
codebooks = chat_chain.chat_env.codes.codebooks
```

### Evaluator Interface

```python
# Input: problem dict + run_result dict
# Output: evaluation result dict
problem = {
    "id": task.task_id,
    "problem": task.prompt,
    "test": task.test,
    "entry_point": task.entry_point
}
run_result = {
    "final_answer": generated_code,
    "extracted": True
}
result = evaluator.evaluate(problem, run_result)
```

## Notes

1. **API Rate Limit**: ChatDev frequently calls the OpenAI API, watch your quota
2. **Execution Time**: 100 tasks take approximately 1-2 hours, please be patient
3. **Resource Usage**: Each task creates temporary directories and log files
4. **Error Recovery**: The system continues processing remaining tasks after a failure
5. **Log Cleanup**: Temporary files in the `WareHouse/` directory can be cleaned up after evaluation

## Troubleshooting

### Common Issues

1. **ImportError**: Check path settings and dependencies
2. **API Error**: Check OpenAI API key and network connectivity
3. **Timeout**: Some complex tasks may timeout; this is expected
4. **Permission Error**: Ensure write permissions in the working directory

### Debug Mode

Enable detailed logs by modifying:
```bash
# In run_benchmark.py
logging.basicConfig(level=logging.DEBUG)
``` 