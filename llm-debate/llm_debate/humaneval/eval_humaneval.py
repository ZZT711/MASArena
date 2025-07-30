import json
import re
import numpy as np
import time
import sys
import os
import traceback
import ast
from threading import Thread
from typing import Dict, Any, Tuple, Optional, List
import random


class EnhancedHumanEvalEvaluator:
    """Enhanced HumanEval evaluator, aiming to replicate the original evaluation logic as closely as possible"""
    
    class TimeoutError(Exception):
        """Execution timeout exception"""
        pass

    def __init__(self, timeout: int = 60, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose

    def log(self, message: str, level: str = "INFO"):
        """Simple logging function"""
        if self.verbose:
            print(f"[{level}] {message}")

    def run_with_timeout(self, func, args, timeout: int = None):
        """
        Execute a function within a specified time, raise an exception if it times out
        """
        timeout = timeout or self.timeout
        result = []
        exception = []

        def target():
            try:
                result.append(func(*args))
            except BaseException as e:
                exception.append(e)

        thread = Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise self.TimeoutError("Execution timed out")

        if exception:
            raise exception[0]

        return result[0] if result else None

    def syntax_check(self, code: str) -> bool:
        """Check if the code syntax is correct"""
        try:
            ast.parse(code)
            return True
        except (SyntaxError, MemoryError):
            if self.verbose:
                traceback.print_exc()
            return False

    def code_extract_fallback(self, text: str) -> str:
        """
        Code extraction fallback method, similar to the original code_extract
        Tries different line combinations to find the longest syntactically correct code block
        """
        lines = text.split("\n")
        longest_code = ""
        longest_length = 0

        # Try to find the longest syntactically correct consecutive code block
        for i in range(len(lines)):
            for j in range(i + 1, len(lines) + 1):
                current_lines = "\n".join(lines[i:j])
                if self.syntax_check(current_lines):
                    # Calculate the number of non-empty lines as length
                    current_length = sum(1 for line in lines[i:j] if line.strip())
                    if current_length > longest_length:
                        longest_length = current_length
                        longest_code = current_lines

        return longest_code if longest_code else text

    def extract_imports_and_functions(self, code: str) -> str:
        """
        Simplified code cleaning, extracts import statements and function definitions
        """
        try:
            tree = ast.parse(code)
            
            # Collect import statements and function definitions
            imports = []
            functions = []
            classes = []
            assignments = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.get_source_segment(code, node))
                elif isinstance(node, ast.FunctionDef):
                    functions.append(ast.get_source_segment(code, node))
                elif isinstance(node, ast.ClassDef):
                    classes.append(ast.get_source_segment(code, node))
                elif isinstance(node, ast.Assign):
                    assignments.append(ast.get_source_segment(code, node))
            
            # Recombine code
            sanitized_parts = []
            if imports:
                sanitized_parts.extend(imports)
            if classes:
                sanitized_parts.extend(classes)
            if functions:
                sanitized_parts.extend(functions)
            if assignments:
                sanitized_parts.extend(assignments)
            
            if sanitized_parts:
                return "\n\n".join(filter(None, sanitized_parts))
            else:
                return code
                
        except Exception as e:
            self.log(f"AST parsing failed, using original code: {e}", "WARNING")
            return code

    def extract_code(self, text: str) -> str:
        """
        Enhanced code extraction, emulates the multi-level fallback strategy of the original version
        """
        text = text.strip()
        self.log(f"Starting code extraction... Text snippet: {text[:100]}")

        # ① "## Validated Code" marker block (original feature)
        qa_match = re.search(r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if qa_match:
            code = qa_match.group(1).strip()
            self.log("Code found in 'Validated Code' section")
            return code

        # ② Content within <answer>...</answer> tags
        answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            # Remove code block markers
            code = re.sub(r'^```python\s*', '', code, flags=re.IGNORECASE)
            code = re.sub(r'^```\s*', '', code)
            code = re.sub(r'\s*```$', '', code)
            self.log("Code found in <answer> tag")
            return code.strip()

        # ③ ```python``` code block
        python_block_pattern = r"```python\s*([\s\S]*?)\s*```"
        match = re.search(python_block_pattern, text, re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            self.log("Code found in python code block")
            return code

        # ④ Generic ``` code block
        code_block_pattern = r"```\s*([\s\S]*?)\s*```"
        match = re.search(code_block_pattern, text)
        if match:
            code = match.group(1).strip()
            self.log("Code found in generic code block")
            return code

        # ⑤ Function definition pattern matching
        func_pattern = r"(def\s+\w+\s*\(.*?\):[\s\S]*?)(?=\n{2,}|\n\s*def|\Z)"
        match = re.search(func_pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            self.log("Code found by function pattern matching")
            return code

        # ⑥ Similar to original sanitize (simplified)
        try:
            cleaned_code = self.extract_imports_and_functions(text)
            if cleaned_code != text and self.syntax_check(cleaned_code):
                self.log("Code extracted by AST cleaning")
                return cleaned_code
        except Exception as e:
            self.log(f"AST cleaning failed: {e}")

        # ⑦ Similar to original code_extract fallback
        try:
            extracted_code = self.code_extract_fallback(text)
            if extracted_code and self.syntax_check(extracted_code):
                self.log("Code extracted by fallback")
                return extracted_code
        except Exception as e:
            self.log(f"Fallback extraction failed: {e}")

        # ⑧ Final fallback: return the entire text
        self.log("Using original text as final fallback")
        return text.strip()

    def check_solution(self, code: str, test: str, entry_point: str) -> Tuple[bool, str]:
        """
        Check if the solution is correct, exactly matching the original logic
        """
        try:
            # Create an isolated namespace
            env = {}

            # Inject candidate implementation
            exec(code, env)
            
            # Check if entry_point exists
            if entry_point not in env:
                msg = f"Function '{entry_point}' not found"
                self.log(f"Check failed: {msg}", "ERROR")
                return False, msg
            
            candidate_fn = env[entry_point]

            # Inject and get check function
            exec(test, env)
            
            if "check" not in env:
                msg = "Test function 'check' not found"
                self.log(f"Check failed: {msg}", "ERROR")
                return False, msg
            
            check_fn = env["check"]

            # Execute tests (with timeout)
            self.run_with_timeout(check_fn, (candidate_fn,), timeout=60)
            return True, "All tests passed"

        except self.TimeoutError as te:
            msg = str(te)
        except AssertionError as ae:
            msg = f"Test failed: {ae}"
        except Exception as exc:
            msg = f"Execution error: {exc}"
            if self.verbose:
                self.log(traceback.format_exc(), "ERROR")

        self.log(f"Check failed: {msg}", "ERROR")
        return False, msg

    def calculate_score(self, test_code: str, prediction: str, entry_point: str) -> Tuple[float, str, str]:
        """
        Calculate score, consistent with original logic
        """
        passed, message = self.check_solution(prediction, test_code, entry_point)
        return (1.0 if passed else 0.0), prediction, message

    def evaluate_single(self, text: str, test_data: Dict[str, Any]) -> Tuple[float, str, str]:
        """
        Evaluate a single code solution, fully emulating the original process
        """
        try:
            # Use enhanced code extraction
            extracted_code = self.extract_code(text)
            
            if not extracted_code.strip():
                return 0.0, "", "No valid code extracted"
            
            # Basic syntax check
            if not self.syntax_check(extracted_code):
                return 0.0, extracted_code, "Syntax error"
            
            # Execute full tests
            score, code_used, message = self.calculate_score(
                test_data.get("test", ""), 
                extracted_code, 
                test_data.get("entry_point", "")
            )
            
            return score, code_used, message
            
        except Exception as e:
            error_msg = f"Evaluation error: {e}"
            if self.verbose:
                error_msg += f"\n{traceback.format_exc()}"
            return 0.0, "", error_msg

    def create_evaluation_report(self, problem: Dict[str, Any], text: str, score: float, code_used: str, message: str) -> Dict[str, Any]:
        """
        Create detailed evaluation report, similar to original create_run
        """
        return {
            "task_id": problem.get("id", "unknown"),
            "problem": problem.get("problem", ""),
            "final_answer": text,
            "extracted_answer": code_used,
            "expected_test": problem.get("test", ""),
            "entry_point": problem.get("entry_point", ""),
            "score": score,
            "message": message,
            "passed": score == 1.0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


def compute_accuracy_for_agents_enhanced(responses, test_data, evaluator):
    """
    Enhanced multi-agent accuracy calculation
    """
    if not responses:
        return 0, "", "No responses", {}
    
    eval_results = []
    detailed_reports = []
    
    for i, response in enumerate(responses):
        if isinstance(response, list) and len(response) > 0:
            # Get the last response
            last_response = response[-1]
            if isinstance(last_response, dict) and 'content' in last_response:
                pred_solution = last_response['content']
            else:
                pred_solution = str(last_response)
        else:
            pred_solution = str(response)
        
        # Evaluate this solution
        score, code_used, message = evaluator.evaluate_single(pred_solution, test_data)
        eval_results.append((score, code_used, message, pred_solution))
        
        # Create detailed report
        report = evaluator.create_evaluation_report(test_data, pred_solution, score, code_used, message)
        report["agent_id"] = i
        detailed_reports.append(report)
    
    if not eval_results:
        return 0, "", "No valid code extraction", {}
    
    # Select the solution with the highest score
    best_result = max(eval_results, key=lambda x: x[0])
    best_score, best_code, best_message, best_original = best_result
    
    # Statistics
    best_count = sum(1 for result in eval_results if result[0] == best_score)
    avg_score = np.mean([result[0] for result in eval_results])
    
    enhanced_message = f"{best_message} (Best score {best_score:.2f}, {len(eval_results)} solutions with {best_count} achieving the highest score, average score: {avg_score:.2f})"
    
    return best_score, best_code, enhanced_message, {
        "detailed_reports": detailed_reports,
        "best_agent_id": eval_results.index(best_result),
        "average_score": avg_score,
        "score_distribution": [result[0] for result in eval_results]
    }


def evaluate_humaneval_results_enhanced(response_file):
    """
    Enhanced HumanEval result evaluation
    """
    # Initialize enhanced evaluator
    evaluator = EnhancedHumanEvalEvaluator(verbose=True)
    print("Using enhanced HumanEval evaluator (replicating original logic)")
    
    # Load results
    try:
        with open(response_file, "r", encoding="utf-8") as f:
            response_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Result file not found {response_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file {response_file}")
        return
    
    print(f"Loaded results for {len(response_dict)} problems")
    
    questions = list(response_dict.keys())
    scores = []
    correct_count = 0
    total_count = 0
    
    # Detailed result recording
    detailed_results = []
    all_agent_reports = []
    
    # Randomize question order
    random.shuffle(questions)
    
    for i, question in enumerate(questions):
        try:
            responses, test_data, task_id = response_dict[question]
            
            # Calculate accuracy (enhanced)
            score, best_code, message, extra_info = compute_accuracy_for_agents_enhanced(
                responses, test_data, evaluator
            )
            
            # Update statistics
            scores.append(score)
            total_count += 1
            
            if score >= 1.0:  # Fully correct
                correct_count += 1
            
            # Record detailed results
            detailed_result = {
                "task_id": task_id,
                "question": question[:200] + "..." if len(question) > 200 else question,
                "score": score,
                "best_code": best_code[:500] + "..." if len(best_code) > 500 else best_code,
                "message": message,
                "is_correct": score >= 1.0,
                "agent_count": len(responses),
                "average_agent_score": extra_info.get("average_score", 0),
                "best_agent_id": extra_info.get("best_agent_id", -1)
            }
            detailed_results.append(detailed_result)
            
            # Save detailed reports for all agents
            if "detailed_reports" in extra_info:
                for report in extra_info["detailed_reports"]:
                    report["question_id"] = i
                    report["question_task_id"] = task_id
                all_agent_reports.extend(extra_info["detailed_reports"])
            
            # Print detailed information (first 10 examples)
            if i < 10:
                print(f"\nProblem {i+1}: {task_id}")
                print(f"Question: {question[:150]}...")
                print(f"Score: {score:.2f}")
                print(f"Best code preview: {best_code[:200]}...")
                print(f"Evaluation info: {message}")
                print(f"Passed: {'✓' if score >= 1.0 else '✗'}")
                print(f"Agent count: {len(responses)}, Average score: {extra_info.get('average_score', 0):.2f}")
            
        except Exception as e:
            print(f"Error processing problem {i+1}: {e}")
            if evaluator.verbose:
                traceback.print_exc()
            detailed_results.append({
                "task_id": "error",
                "question": question[:200] if isinstance(question, str) else str(question)[:200],
                "score": 0,
                "best_code": "",
                "message": str(e),
                "is_correct": False,
                "agent_count": 0,
                "average_agent_score": 0,
                "best_agent_id": -1
            })
            scores.append(0)
            total_count += 1
            continue
    
    # Calculate overall accuracy
    if scores:
        overall_accuracy = np.mean(scores)
        overall_std = np.std(scores) / (len(scores) ** 0.5)
        pass_rate = correct_count / total_count  # Proportion passing
        
        print(f"\n=== Enhanced HumanEval Evaluation Results ===")
        print(f"Total questions: {total_count}")
        print(f"Fully correct count: {correct_count}")
        print(f"Pass rate: {pass_rate:.4f} ({pass_rate*100:.2f}%)")
        print(f"Average score: {overall_accuracy:.4f} ± {overall_std:.4f}")
        print(f"Average score: {overall_accuracy*100:.2f}%")
        
        # Analyze score distribution
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
            "evaluation_type": "enhanced_humaneval",
            "total_questions": total_count,
            "correct_count": correct_count,
            "pass_rate": pass_rate,
            "overall_accuracy": overall_accuracy,
            "overall_std": overall_std,
            "score_distribution": score_distribution,
            "detailed_results": detailed_results,
            "all_agent_reports": all_agent_reports
        }
        
        eval_output_file = response_file.replace('.json', '_enhanced_evaluation_results.json')
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed evaluation results saved to: {eval_output_file}")
        
        # Save enhanced evaluation report
        report_file = response_file.replace('.json', '_enhanced_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("HumanEval Enhanced Evaluation Report\n")
            f.write("==========================\n\n")
            f.write("Evaluator Features:\n")
            f.write("- Multi-level code extraction strategy (including '## Validated Code' marker support)\n")
            f.write("- AST syntax analysis and code cleaning\n")
            f.write("- Intelligent code fallback extraction\n")
            f.write("- Detailed error classification and logging\n")
            f.write("- Full agent-level analysis\n\n")
            f.write(f"Total questions: {total_count}\n")
            f.write(f"Fully correct count: {correct_count}\n")
            f.write(f"Pass rate: {pass_rate:.4f} ({pass_rate*100:.2f}%)\n")
            f.write(f"Average score: {overall_accuracy:.4f} ± {overall_std:.4f}\n")
            f.write(f"Average score: {overall_accuracy*100:.2f}%\n\n")
            
            f.write("Score distribution:\n")
            for score_range in sorted(score_distribution.keys()):
                count = score_distribution[score_range]
                percentage = count / total_count * 100
                f.write(f"Score {score_range}: {count} questions ({percentage:.2f}%)\n")
        
        print(f"Enhanced evaluation report saved to: {report_file}")
        
    else:
        print("No valid evaluation results")


if __name__ == "__main__":
    # Default evaluation file
    response_file = "humaneval_3_2.json"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        response_file = sys.argv[1]
    
    print(f"Starting evaluation of HumanEval results file: {response_file}")
    evaluate_humaneval_results_enhanced(response_file) 