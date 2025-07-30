"""
HumanEval Benchmark for ChatDev

Randomly selects 100 HumanEval problems, generates code for each problem using ChatDev,
immediately evaluates the code, and computes the overall accuracy.

The code is stored in the chat_env.codes.codebooks field.
"""
import argparse
import logging
import os
import sys
import json
import random
import time
from typing import List, Dict

from camel.typing import ModelType

# 添加必要的路径
root = os.path.dirname(__file__)
chatdev_root = os.path.dirname(root)
# 确保chatdev_root在前面，避免utils模块冲突
sys.path.insert(0, chatdev_root)
sys.path.append(root)

from chatdev.chat_chain import ChatChain
from data_structures import HumanevalTask, EvaluationResult, BenchmarkSummary
from humaneval_evaluator import HumanEvalEvaluator

try:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message import FunctionCall
    openai_new_api = True
except ImportError:
    openai_new_api = False
    print("Warning: Your OpenAI version is outdated.")


def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(chatdev_root, "CompanyConfig", company)
    default_config_dir = os.path.join(chatdev_root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json", 
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)


def load_humaneval_tasks(jsonl_path: str) -> List[HumanevalTask]:
    """加载HumanEval测试集"""
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tasks.append(HumanevalTask.from_dict(data))
    return tasks


def run_single_task(task: HumanevalTask, config_name: str, org_name: str, 
                   model_name: str, code_path: str) -> Dict[str, str]:
    """
    为单个任务运行ChatDev，返回生成的代码
    
    Args:
        task: HumanEval任务
        config_name: 配置名称
        org_name: 组织名称
        model_name: 模型名称
        code_path: 代码路径
        
    Returns:
        生成的代码字典 {filename: code_content}
    """
    # 获取配置文件路径
    config_path, config_phase_path, config_role_path = get_config(config_name)

    # 模型映射表
    args2type = {
        'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
        'GPT_4': ModelType.GPT_4,
        'GPT_4_TURBO': ModelType.GPT_4_TURBO,
        'GPT_4O': ModelType.GPT_4O,
        'GPT_4O_MINI': ModelType.GPT_4O_MINI,
    }
    if openai_new_api:
        args2type['GPT_3_5_TURBO'] = ModelType.GPT_3_5_TURBO_NEW

    # 初始化 ChatChain
    chat_chain = ChatChain(
        config_path=config_path,
        config_phase_path=config_phase_path,
        config_role_path=config_role_path,
        task_prompt=task.prompt,
        project_name=f"Humaneval_{task.task_id}",
        org_name=org_name,
        model_type=args2type[model_name],
        code_path=code_path
    )

    # 设置日志
    logging.basicConfig(
        filename=chat_chain.log_filepath, 
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%d-%m %H:%M:%S', 
        encoding="utf-8"
    )

    try:
        # 执行完整流程并获取生成的代码
        codebooks = chat_chain.run_all_and_get_codebooks()
        print("=============================================================")
        print("codebooks")
        print("=============================================================")
        print(codebooks)
        return codebooks
    except Exception as e:
        print(f"Error running ChatDev for {task.task_id}: {e}")
        return {}


def evaluate_single_task(task: HumanevalTask, codebooks: Dict[str, str], 
                        evaluator: HumanEvalEvaluator) -> (EvaluationResult, str):
    """
    评测单个任务
    
    Args:
        task: HumanEval任务
        codebooks: 生成的代码
        evaluator: 评测器
        
    Returns:
        评测结果
    """
    start_time = time.time()
    
    # 提取主要的Python代码
    main_code = ""
    for filename, code in codebooks.items():
        if filename.endswith('.py'):
            main_code += code + "\n"
    
    try:
        # 按照evaluator的接口要求构造问题和结果格式
        problem = {
            "id": task.task_id,
            "problem": task.prompt,
            "test": task.test,
            "entry_point": task.entry_point
        }
        
        run_result = {
            "final_answer": main_code,
            "extracted": True  # 表示代码已经提取过了
        }
        
        # 调用evaluator的evaluate方法
        result = evaluator.evaluate(problem, run_result)
        execution_time = time.time() - start_time
        
        # 从结果中提取信息
        score = result.get('score', 0.0)
        success = (score == 1.0)
        message = result.get('message', '')
        
        # 构造EvaluationResult
        return EvaluationResult(
            task_id=task.task_id,
            results=[],  # evaluator不返回详细的测试用例结果
            passed_count=1 if success else 0,
            total_count=1,
            success=success,
            code_generated=main_code,
            execution_time=execution_time
        ), message
        
    except Exception as e:
        execution_time = time.time() - start_time
        err_msg = f"Error evaluating {task.task_id}: {e}"
        print(err_msg)
        
        return EvaluationResult(
            task_id=task.task_id,
            results=[],
            passed_count=0,
            total_count=1,
            success=False,
            code_generated=main_code,
            execution_time=execution_time
        ), err_msg


def run_benchmark(num_tasks: int = 100):
    """
    Run the HumanEval benchmark

    Args:
        num_tasks: Number of tasks to evaluate (default: 100)
    """
    # Configuration parameters
    config_name = "Human"
    org_name = "DefaultOrganization"
    model_name = "GPT_4O_MINI"
    code_path = ""
    
    print(f"Starting HumanEval benchmark: {num_tasks} tasks in total...")
    
    # Load the HumanEval dataset
    humaneval_path = os.path.join(root, "data", "humaneval_test.jsonl")
    all_tasks = load_humaneval_tasks(humaneval_path)
    
    # Randomly select the specified number of tasks
    random.seed(42)  # Set random seed for reproducibility
    selected_tasks = random.sample(all_tasks, min(num_tasks, len(all_tasks)))
    
    print(f"Selected {len(selected_tasks)} tasks")
    
    # Initialize the evaluator
    evaluator = HumanEvalEvaluator("humaneval", {})
    
    # Store all evaluation results
    all_results = []
    total_start_time = time.time()
    
    # Process tasks one by one
    for i, task in enumerate(selected_tasks, 1):
        # print(f"\n处理任务 {i}/{len(selected_tasks)}: {task.task_id}")
        
        # 运行ChatDev生成代码
        # print(f"  生成代码中...")
        codebooks = run_single_task(task, config_name, org_name, model_name, code_path)
        
        message = ""
        if not codebooks:
            # print(f"  {task.task_id} 代码生成失败")
            # Create a failed evaluation result
            result = EvaluationResult(
                task_id=task.task_id,
                results=[],
                passed_count=0,
                total_count=1,
                success=False,
                code_generated="",
                execution_time=0.0
            )
            message = "代码生成失败"
        else:
            # print(f"  评测中...")
            # 立即评测
            result, message = evaluate_single_task(task, codebooks, evaluator)
        
        all_results.append(result)
        
        # Output current task result [[memory:2678902]]
        status = "Pass" if result.success else "Fail"
        print("-" * 50)
        print(f"Problem {i}: {task.task_id}")
        # print(f"题目:\n{task.prompt}")
        # print(f"\n预期行为 (测试代码):\n{task.test}")
        print(f"Model output:\n{result.code_generated or 'No code generated'}")
        print(f"Evaluation result: {status}")
        if not result.success:
            print(f"Error message: {message}")
        print("-" * 50)
    
    # Compute overall statistics
    total_execution_time = time.time() - total_start_time
    passed_tasks = sum(1 for r in all_results if r.success)
    total_test_cases = sum(r.total_count for r in all_results)
    passed_test_cases = sum(r.passed_count for r in all_results)
    
    avg_test_accuracy = passed_test_cases / total_test_cases if total_test_cases > 0 else 0.0
    
    # Create summary results object
    summary = BenchmarkSummary(
        total_tasks=len(selected_tasks),
        passed_tasks=passed_tasks,
        accuracy=passed_tasks / len(selected_tasks),
        avg_test_accuracy=avg_test_accuracy,
        total_execution_time=total_execution_time,
        details=all_results
    )
    
    # Print summary
    print(summary)
    
    # Save detailed results to file
    results_path = os.path.join(root, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        # 将结果转换为可序列化的格式
        results_dict = {
            "summary": {
                "total_tasks": summary.total_tasks,
                "passed_tasks": summary.passed_tasks,
                "accuracy": summary.accuracy,
                "avg_test_accuracy": summary.avg_test_accuracy,
                "total_execution_time": summary.total_execution_time
            },
            "details": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "passed_count": r.passed_count,
                    "total_count": r.total_count,
                    "accuracy": r.accuracy,
                    "execution_time": r.execution_time,
                    "code_generated": r.code_generated
                }
                for r in all_results
            ]
        }
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HumanEval Benchmark for ChatDev')
    parser.add_argument('--num_tasks', type=int, default=100,
                        help="Number of tasks to test (default: 100)")
    args = parser.parse_args()
    
    # Run the benchmark
    summary = run_benchmark(args.num_tasks)


