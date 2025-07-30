import json
import sys
import os
import asyncio
import openai
from typing import Dict, Any, List
import numpy as np

# Add the root directory to path to find evaluators module
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

from evaluators.humaneval_evaluator import HumanEvalEvaluator
from evaluators.mmlu_pro_evaluator import MMLU_ProEvaluator
from evaluators.math_evaluator import MathEvaluator
from evaluators.drop_evaluator import DROPEvaluator
from evaluators.bbh_evaluator import BBHEvaluator
from evaluators.aime_evaluator import AIMEEvaluator
from llm_multiagent_debate.format_prompts import get_format_prompt


class MultiAgentEvaluator:
    """
    Multi-agent BBH evaluator using official BBHEvaluator with answer aggregation.
    """
    
    def __init__(self, benchmark_name: str, verbose: bool = False):
        self.model_name = "gpt-4o-mini"
        self.verbose = verbose
        self.benchmark_name = benchmark_name
        
        # Initialize the official evaluator
        config = {
            "verbose": verbose,
            "timeout": 60
        }

        if benchmark_name == "bbh":
            self.evaluator = BBHEvaluator(name="bbh", config=config)
        elif benchmark_name == "humaneval":
            self.evaluator = HumanEvalEvaluator(name="humaneval", config=config)
        elif benchmark_name == "mmlu_pro":
            self.evaluator = MMLU_ProEvaluator(name="mmlu_pro", config=config)
        elif benchmark_name == "math":
            self.evaluator = MathEvaluator(name="math", config=config)
        elif benchmark_name == "drop":
            self.evaluator = DROPEvaluator(name="drop", config=config)
        elif benchmark_name == "aime":
            self.evaluator = AIMEEvaluator(name="aime", config=config)
        else:
            raise ValueError(f"Invalid benchmark name: {benchmark_name}")

    def compute_accuracy_for_agents(self, responses: List[Any], test_data: Dict[str, Any]) -> tuple:
        """
        Compute accuracy for multi-agent responses using the official evaluator.
        
        Args:
            responses: List of agent responses
            test_data: Test data containing problem information
            
        Returns:
            Tuple of (score, extracted_code, message, extra_info)
        """
        if not responses:
            return 0, "", "No responses provided", {}
        
        # Extract final answers from responses
        final_answers = []
        for response in responses:
            if isinstance(response, list) and len(response) > 0:
                # Get the last response
                last_response = response[-1]
                if isinstance(last_response, dict) and 'content' in last_response:
                    content = last_response['content']
                    if content:  # 确保content不为None或空字符串
                        final_answers.append(content)
                    else:
                        final_answers.append("No content available")
                else:
                    final_answers.append(str(last_response) if last_response else "No response")
            else:
                final_answers.append(str(response) if response else "No response")
        
        # Use the last agent's answer as aggregated answer
        aggregated_answer = final_answers[-1] if final_answers else "No answer available"
        
        # Create run_result format expected by the evaluator
        run_result = {"final_answer": aggregated_answer}
        
        # Use the official evaluator
        try:
            # 检查test_data是否为字典类型，如果不是，返回错误
            if not isinstance(test_data, dict):
                return 0.0, "", f"Error: test_data is not a dictionary (got {type(test_data).__name__})", {}

            # The official evaluator expects a dictionary containing specific keys.
            # Create the proper format for each benchmark type.
            if self.benchmark_name == "bbh":
                problem_data = {
                    "id": test_data.get("task_id"),
                    "problem": test_data.get("input"),
                    "solution": test_data.get("target"),
                }
            elif self.benchmark_name == "humaneval":
                problem_data = {
                    "id": test_data.get("task_id"),
                    "problem": test_data.get("prompt"),
                    "solution": test_data.get("canonical_solution"),
                    "test": test_data.get("test"),
                    "entry_point": test_data.get("entry_point"),
                }
            elif self.benchmark_name == "mmlu_pro":
                problem_data = {
                    "id": test_data.get("id"),
                    "problem": test_data.get("question"),
                    "solution": test_data.get("answer"),
                }
            elif self.benchmark_name == "math":
                problem_data = {
                    "id": test_data.get("id", "unknown"),
                    "problem": test_data.get("problem"),
                    "solution": test_data.get("solution"),
                }
            elif self.benchmark_name == "drop":
                problem_data = {
                    "id": test_data.get("id"),
                    "problem": test_data.get("context"),
                    "solution": test_data.get("ref_text"),
                }
            elif self.benchmark_name == "aime":
                problem_data = {
                    "id": test_data.get("id", "unknown"),
                    "problem": test_data.get("question"),
                    "solution": test_data.get("answer"),
                }
            else:
                # Generic fallback
                problem_data = test_data

            eval_result = self.evaluator.evaluate(problem_data, run_result)
            
            score = eval_result.get("score", 0.0) if isinstance(eval_result, dict) else 0.0
            extracted_code = eval_result.get("extracted_answer", "") if isinstance(eval_result, dict) else str(eval_result)
            message = eval_result.get("message", "") if isinstance(eval_result, dict) else ""
            
            extra_info = {
                "agent_count": len(final_answers),
                "aggregated_answer": aggregated_answer,
                "individual_answers": final_answers,
                "run_evaluation": eval_result.get("run_evaluation") if isinstance(eval_result, dict) else None
            }
            
            return score, extracted_code, message, extra_info
            
        except Exception as e:
            error_msg = f"Evaluation error: {e}"
            if self.verbose:
                import traceback
                error_msg += f"\n{traceback.format_exc()}"
            return 0.0, "", error_msg, {}

    def evaluate_results(self, response_file: str):
        """
        Evaluate results using the official evaluator with agent aggregation.
        
        Args:
            response_file: Path to the response file containing multi-agent results
        """
        print(f"Using official {self.benchmark_name.upper()} evaluator with multi-agent aggregation")
        print(f"Model: {self.model_name}")
        print(f"Format prompt source: format_prompts.get_format_prompt('{self.benchmark_name}')")
        
        # Load results
        try:
            with open(response_file, "r", encoding="utf-8") as f:
                response_dict = json.load(f)
        except FileNotFoundError:
            print(f"Error: Response file not found {response_file}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file {response_file}")
            return
        
        print(f"Loaded results for {len(response_dict)} questions")
        
        questions = list(response_dict.keys())
        scores = []
        correct_count = 0
        total_count = 0
        
        detailed_results = []
        evaluation_reports = []
        
        for i, question in enumerate(questions):
            try:
                responses, test_data, task_id = response_dict[question]
                
                # test_data 
                if not isinstance(test_data, dict):
                    print(f"Warning: Test data format is incorrect for question {i+1} (is {type(test_data).__name__}).")
                    scores.append(0)
                    total_count += 1
                    detailed_results.append({
                        "task_id": task_id,
                        "question": question[:200] + "..." if len(question) > 200 else question,
                        "score": 0,
                        "extracted_answer": "",
                        "message": "Data format error: test_data is not a dictionary type",
                        "is_correct": False,
                        "agent_count": 0,
                        "aggregated_answer": ""
                    })
                    continue
                
                # Compute accuracy using official evaluator with aggregation
                score, extracted_code, message, extra_info = self.compute_accuracy_for_agents(
                    responses, test_data
                )
                
                # Update statistics
                scores.append(score)
                total_count += 1
                
                if score >= 1.0:
                    correct_count += 1
                
                # Record detailed results
                detailed_result = {
                    "task_id": task_id,
                    "question": question[:200] + "..." if len(question) > 200 else question,
                    "score": score,
                    "extracted_answer": extracted_code[:500] + "..." if len(extracted_code) > 500 else extracted_code,
                    "message": message,
                    "is_correct": score >= 1.0,
                    "agent_count": extra_info.get("agent_count", 0),
                    "aggregated_answer": extra_info.get("aggregated_answer", "")[:300] + "..." if len(extra_info.get("aggregated_answer", "")) > 300 else extra_info.get("aggregated_answer", "")
                }
                detailed_results.append(detailed_result)
                
                # Save evaluation report
                if extra_info.get("run_evaluation"):
                    evaluation_reports.append({
                        "task_id": task_id,
                        "question_index": i,
                        "run_evaluation": extra_info["run_evaluation"]
                    })
                
                # Print progress for first 10 examples
                if i < 10:
                    print(f"\nQuestion {i+1}: {task_id}")
                    print(f"Question: {question[:100]}...")
                    print(f"Expected answer: {test_data.get('target', 'N/A')}")
                    print(f"Output answer: {extracted_code}")
                    print(f"Correctness: {'✓' if score >= 1.0 else '✗'}")
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                
                detailed_results.append({
                    "task_id": "error",
                    "question": question[:200] if isinstance(question, str) else str(question)[:200],
                    "score": 0,
                    "extracted_answer": "",
                    "message": str(e),
                    "is_correct": False,
                    "agent_count": 0,
                    "aggregated_answer": ""
                })
                scores.append(0)
                total_count += 1
                continue
        
        # Calculate overall accuracy
        if scores:
            overall_accuracy = np.mean(scores)
            overall_std = np.std(scores) / (len(scores) ** 0.5)
            pass_rate = correct_count / total_count
            
            print(f"\n=== Official {self.benchmark_name.upper()} Evaluation Results ===")
            print(f"Total questions: {total_count}")
            print(f"Correct answers: {correct_count}")
            print(f"Pass rate: {pass_rate:.4f} ({pass_rate*100:.2f}%)")
            print(f"Average score: {overall_accuracy:.4f} ± {overall_std:.4f}")
            print(f"Average score: {overall_accuracy*100:.2f}%")
            
            # Score distribution
            score_distribution = {}
            for score in scores:
                score_range = f"{score:.1f}"
                score_distribution[score_range] = score_distribution.get(score_range, 0) + 1
            
            print(f"\n=== Score Distribution ===")
            for score_range in sorted(score_distribution.keys()):
                count = score_distribution[score_range]
                percentage = count / total_count * 100
                print(f"Score {score_range}: {count} questions ({percentage:.2f}%)")
            
            # Save evaluation results
            eval_results = {
                "evaluation_type": f"official_{self.benchmark_name}_with_aggregation",
                "model_used_for_aggregation": self.model_name,
                "total_questions": total_count,
                "correct_count": correct_count,
                "pass_rate": pass_rate,
                "overall_accuracy": overall_accuracy,
                "overall_std": overall_std,
                "score_distribution": score_distribution,
                "detailed_results": detailed_results,
                "evaluation_reports": evaluation_reports
            }
            
            # Save detailed results
            eval_output_file = response_file.replace('.json', '_official_evaluation_results.json')
            with open(eval_output_file, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)
            
            print(f"\nDetailed evaluation results saved to: {eval_output_file}")
            
            # Save evaluation report
            report_file = response_file.replace('.json', '_official_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"{self.benchmark_name.upper()} Official Evaluator Report\n")
                f.write("===================================\n\n")
                f.write("Evaluator Features:\n")
                f.write(f"- Official {self.benchmark_name.upper()} evaluator from evaluators module\n")
                f.write("- Multi-agent answer aggregation using GPT-4o-mini\n")
                f.write(f"- Format prompt from format_prompts.get_format_prompt('{self.benchmark_name}')\n")
                f.write("- LangSmith integration for evaluation execution\n")
                f.write("- Execution safety based on timeout\n\n")
                f.write(f"Model used for aggregation: {self.model_name}\n")
                f.write(f"Total questions: {total_count}\n")
                f.write(f"Correct answers: {correct_count}\n")
                f.write(f"Pass rate: {pass_rate:.4f} ({pass_rate*100:.2f}%)\n")
                f.write(f"Average score: {overall_accuracy:.4f} ± {overall_std:.4f}\n")
                f.write(f"Average score: {overall_accuracy*100:.2f}%\n\n")
                
                f.write("Score distribution:\n")
                for score_range in sorted(score_distribution.keys()):
                    count = score_distribution[score_range]
                    percentage = count / total_count * 100
                    f.write(f"Score {score_range}: {count} questions ({percentage:.2f}%)\n")
            
            print(f"Official evaluation report saved to: {report_file}")
            
        else:
            print("No valid evaluation results")


if __name__ == "__main__":
    # Known benchmarks for auto-detection
    known_benchmarks = ["bbh", "humaneval", "mmlu_pro", "math", "drop", "aime"]
    
    # Default benchmark
    benchmark_name = "bbh"
    
    # Check command line arguments for a response file
    response_file_arg = next((arg for arg in sys.argv[1:] if not arg.startswith('-')), None)
    
    if response_file_arg:
        response_file = response_file_arg
        # Try to infer benchmark name from filename
        inferred_benchmark = None
        base_name = os.path.basename(response_file)
        for b in known_benchmarks:
            if base_name.startswith(b):
                inferred_benchmark = b
                break
        if inferred_benchmark:
            benchmark_name = inferred_benchmark
        else:
            print(f"Warning: Could not infer benchmark name from filename '{response_file}'. Defaulting to 'bbh'.")
            benchmark_name = "bbh"
    else:
        # Default file name
        response_file = "bbh_2_3.json"
    
    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    print(f"Starting official {benchmark_name.upper()} evaluation: {response_file}")
    
    # Initialize and run evaluator
    evaluator = MultiAgentEvaluator(benchmark_name=benchmark_name, verbose=verbose)
    evaluator.evaluate_results(response_file) 

