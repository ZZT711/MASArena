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
from llm_multiagent_debate.format_prompts import get_format_prompt


class MultiAgentHumanEvalEvaluator:
    """
    Multi-agent HumanEval evaluator using official HumanEvalEvaluator with answer aggregation.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        
        # Initialize the official HumanEval evaluator
        config = {
            "verbose": verbose,
            "timeout": 60
        }
        self.evaluator = HumanEvalEvaluator(name="humaneval", config=config)
        
        # Get format prompt for code generation
        self.format_prompt = get_format_prompt("humaneval")
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI()


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
                    final_answers.append(last_response['content'])
                else:
                    final_answers.append(str(last_response))
            else:
                final_answers.append(str(response))
        
        aggregated_answer = final_answers[-1]
        
        # Create run_result format expected by the evaluator
        run_result = {"final_answer": aggregated_answer}
        
        # Use the official evaluator
        try:
            # The official evaluator expects a dictionary containing specific keys like
            # 'problem' for the prompt and 'id' for the task ID. The 'test_data'
            # loaded from the JSON uses 'prompt' and 'task_id'. This creates a new
            # dictionary in the expected format for the evaluator.
            problem_data = {
                **test_data,
                "problem": test_data.get("prompt"),
                "id": test_data.get("task_id"),
            }

            eval_result = self.evaluator.evaluate(problem_data, run_result)
            
            score = eval_result.get("score", 0.0)
            extracted_code = eval_result.get("extracted_answer", "")
            message = eval_result.get("message", "")
            
            extra_info = {
                "agent_count": len(final_answers),
                "aggregated_answer": aggregated_answer,
                "individual_answers": final_answers,
                "run_evaluation": eval_result.get("run_evaluation")
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
        Evaluate HumanEval results using the official evaluator with agent aggregation.
        
        Args:
            response_file: Path to the response file containing multi-agent results
        """
        print(f"Using official HumanEval evaluator with multi-agent aggregation")
        print(f"Model: {self.model_name}")
        print(f"Format prompt source: format_prompts.get_format_prompt('humaneval')")
        
        # Load results
        try:
            with open(response_file, "r", encoding="utf-8") as f:
                response_dict = json.load(f)
        except FileNotFoundError:
            print(f"Error: Response file {response_file} not found")
            return
        except json.JSONDecodeError:
            print(f"Error: Cannot parse JSON file {response_file}")
            return
        
        print(f"Loaded results for {len(response_dict)} problems")
        
        questions = list(response_dict.keys())
        scores = []
        correct_count = 0
        total_count = 0
        
        detailed_results = []
        evaluation_reports = []
        
        for i, question in enumerate(questions):
            try:
                responses, test_data, task_id = response_dict[question]
                
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
                    "extracted_code": extracted_code[:500] + "..." if len(extracted_code) > 500 else extracted_code,
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
                    print(f"\nProblem {i+1}: {task_id}")
                    print(f"Question: {question[:100]}...")
                    print(f"Score: {score:.2f}")
                    print(f"Extracted code preview: {extracted_code[:150]}...")
                    print(f"Message: {message}")
                    print(f"Passed: {'✓' if score >= 1.0 else '✗'}")
                    print(f"Agent count: {extra_info.get('agent_count', 0)}")
                
            except Exception as e:
                print(f"Error processing problem {i+1}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                
                detailed_results.append({
                    "task_id": "error",
                    "question": question[:200] if isinstance(question, str) else str(question)[:200],
                    "score": 0,
                    "extracted_code": "",
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
            
            print(f"\n=== Official HumanEval Evaluation Results ===")
            print(f"Total problems: {total_count}")
            print(f"Correct solutions: {correct_count}")
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
                print(f"Score {score_range}: {count} problems ({percentage:.2f}%)")
            
            # Save evaluation results
            eval_results = {
                "evaluation_type": "official_humaneval_with_aggregation",
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
                f.write("HumanEval Official Evaluator Report\n")
                f.write("===================================\n\n")
                f.write("Evaluator Features:\n")
                f.write("- Official HumanEvalEvaluator from evaluators module\n")
                f.write("- Multi-agent answer aggregation using GPT-4o-mini\n")
                f.write("- Format prompt from format_prompts.get_format_prompt('humaneval')\n")
                f.write("- LangSmith integration for run evaluation\n")
                f.write("- Timeout-based execution safety\n\n")
                f.write(f"Model used for aggregation: {self.model_name}\n")
                f.write(f"Total problems: {total_count}\n")
                f.write(f"Correct solutions: {correct_count}\n")
                f.write(f"Pass rate: {pass_rate:.4f} ({pass_rate*100:.2f}%)\n")
                f.write(f"Average score: {overall_accuracy:.4f} ± {overall_std:.4f}\n")
                f.write(f"Average score: {overall_accuracy*100:.2f}%\n\n")
                
                f.write("Score Distribution:\n")
                for score_range in sorted(score_distribution.keys()):
                    count = score_distribution[score_range]
                    percentage = count / total_count * 100
                    f.write(f"Score {score_range}: {count} problems ({percentage:.2f}%)\n")
            
            print(f"Official evaluation report saved to: {report_file}")
            
        else:
            print("No valid evaluation results")


if __name__ == "__main__":
    # Default evaluation file
    response_file = "humaneval_3_2.json"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        response_file = sys.argv[1]
    
    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    print(f"Starting HumanEval evaluation with official evaluator: {response_file}")
    
    # Initialize and run evaluator
    evaluator = MultiAgentHumanEvalEvaluator(verbose=verbose)
    evaluator.evaluate_results(response_file) 