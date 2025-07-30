import json
import re
import numpy as np
import time
import sys
import os
import traceback
from typing import Dict, Any, Tuple, Optional, List
import random


class EnhancedMMLUEvaluator:
    """Enhanced MMLU evaluator, using the original logic of MMLU_ProEvaluator"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Weight for exact match score is always 1.0 as it's the only metric
        self.exact_match_weight = 1.0

    def log(self, message: str, level: str = "INFO"):
        """Simple logging function"""
        if self.verbose:
            print(f"[{level}] {message}")

    def check_exact_match(self, reference: str, candidate: str) -> float:
        """
        Check if the candidate answer exactly matches the reference answer (case-insensitive)
        Using the original logic of MMLU_ProEvaluator
        
        Args:
            reference: Reference answer (e.g., 'A', 'B', 'C', etc.)
            candidate: Candidate answer
            
        Returns:
            1.0 if fully matched, otherwise 0.0
        """
        # Clean and standardize both answers
        ref_clean = reference.strip().upper()
        cand_clean = candidate.strip().upper()
        
        # Check for exact match
        if cand_clean == ref_clean:
            return 1.0
        
        # Check if the candidate answer is a number index (e.g., "1", "2", "3") converted to letters
        try:
            if cand_clean.isdigit():
                cand_index = int(cand_clean) - 1
                cand_letter = chr(ord('A') + cand_index)
                if cand_letter == ref_clean:
                    return 1.0
        except Exception:
            pass
            
        return 0.0

    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract the answer from the agent's response
        Using the original logic of MMLU_ProEvaluator
        
        Args:
            response: The full response text from the agent
            
        Returns:
            The extracted answer letter
        """
        # 1. Best case: Find <answer>X</answer>
        match = re.search(r'<answer>\s*([A-Z])\s*</answer>', response, re.IGNORECASE)
        if match:
            self.log(f"Found answer in <answer> tag: {match.group(1)}")
            return match.group(1).strip().upper()

        # 2. Alternative: Find (X) at the end of the response
        match = re.search(r'\(?([A-Z])\)?$', response.strip())
        if match:
            self.log(f"Found answer at the end of the response: {match.group(1)}")
            return match.group(1).strip().upper()

        # 3. Alternative: Find "Answer: X" or similar pattern
        match = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
        if match:
            self.log(f"Found answer in 'Answer:' pattern: {match.group(1)}")
            return match.group(1).strip().upper()

        # 4. Final try: Return the last uppercase letter found in the response
        found = re.findall(r'[A-Z]', response)
        if found:
            self.log(f"Using the last found uppercase letter: {found[-1]}")
            return found[-1]

        # If no tag or pattern is found, return the original response
        self.log(f"No clear answer found, returning original response: {response.strip()}")
        return response.strip()

    def get_correct_answer_text(self, problem: Dict[str, Any]) -> str:
        """
        Get the correct answer text from the problem
        Using the original logic of MMLU_ProEvaluator
        
        Args:
            problem: The problem dictionary
            
        Returns:
            The correct answer text
        """
        options = problem.get("options", [])
        answer_index = problem.get("answer_index")
        answer_letter = problem.get("answer")
        
        # If options and a valid answer index exist
        if options and isinstance(answer_index, int) and 0 <= answer_index < len(options):
            return options[answer_index]
        
        # If answer letter and options exist
        if answer_letter and options:
            try:
                # Convert letter to index (A=0, B=1, etc.)
                idx = ord(answer_letter.upper()) - ord('A')
                if 0 <= idx < len(options):
                    return options[idx]
            except Exception:
                pass
        
        # If correct answer text cannot be obtained, return the answer letter/index
        return str(answer_letter if answer_letter else answer_index)

    def evaluate_single(self, text: str, test_data: Dict[str, Any]) -> Tuple[float, str, str]:
        """
        Evaluate a single MMLU solution
        
        Args:
            text: The response text from the agent
            test_data: The test data dictionary
            
        Returns:
            (score, extracted_answer, message)
        """
        try:
            # Extract answer
            extracted_answer = self.extract_answer_from_response(text)
            
            if not extracted_answer.strip():
                return 0.0, "", "No valid answer extracted"
            
            # Get reference answer
            reference_letter = test_data.get("answer", "")
            
            # Calculate exact match score (based on letter)
            score = self.check_exact_match(reference_letter, extracted_answer)
            
            # Get answer text for detailed information
            correct_answer_text = self.get_correct_answer_text(test_data)
            
            if score >= 1.0:
                message = f"Correct! Answer: {extracted_answer} (Reference: {reference_letter} - {correct_answer_text})"
            else:
                message = f"Incorrect! Answer: {extracted_answer}, Expected: {reference_letter} - {correct_answer_text}"
            
            return score, extracted_answer, message
            
        except Exception as e:
            error_msg = f"Evaluation error: {e}"
            if self.verbose:
                error_msg += f"\n{traceback.format_exc()}"
            return 0.0, "", error_msg

    def create_evaluation_report(self, problem: Dict[str, Any], text: str, score: float, extracted_answer: str, message: str) -> Dict[str, Any]:
        """
        Create a detailed evaluation report
        """
        return {
            "task_id": problem.get("question_id", "unknown"),
            "question": problem.get("question", ""),
            "final_answer": text,
            "extracted_answer": extracted_answer,
            "reference_answer": problem.get("answer", ""),
            "correct_answer_text": self.get_correct_answer_text(problem),
            "options": problem.get("options", []),
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
        score, extracted_answer, message = evaluator.evaluate_single(pred_solution, test_data)
        eval_results.append((score, extracted_answer, message, pred_solution))
        
        # Create detailed report
        report = evaluator.create_evaluation_report(test_data, pred_solution, score, extracted_answer, message)
        report["agent_id"] = i
        detailed_reports.append(report)
    
    if not eval_results:
        return 0, "", "No valid answer extraction", {}
    
    # Select the solution with the highest score
    best_result = max(eval_results, key=lambda x: x[0])
    best_score, best_answer, best_message, best_original = best_result
    
    # Statistics
    best_count = sum(1 for result in eval_results if result[0] == best_score)
    avg_score = np.mean([result[0] for result in eval_results])
    
    enhanced_message = f"{best_message} (Best score {best_score:.2f}, {len(eval_results)} solutions have {best_count} with the highest score, average score: {avg_score:.2f})"
    
    return best_score, best_answer, enhanced_message, {
        "detailed_reports": detailed_reports,
        "best_agent_id": eval_results.index(best_result),
        "average_score": avg_score,
        "score_distribution": [result[0] for result in eval_results]
    }


def evaluate_mmlu_results_enhanced(response_file):
    """
    Enhanced MMLU result evaluation
    """
    # Initialize enhanced evaluator
    evaluator = EnhancedMMLUEvaluator(verbose=True)
    print("Using enhanced MMLU evaluator (based on MMLU_ProEvaluator original logic)")
    
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
    
    print(f"Loaded results for {len(response_dict)} questions")
    
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
            score, best_answer, message, extra_info = compute_accuracy_for_agents_enhanced(
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
                "best_answer": best_answer,
                "message": message,
                "is_correct": score >= 1.0,
                "agent_count": len(responses),
                "average_agent_score": extra_info.get("average_score", 0),
                "best_agent_id": extra_info.get("best_agent_id", -1),
                "reference_answer": test_data.get("answer", ""),
                "correct_answer_text": evaluator.get_correct_answer_text(test_data)
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
                print(f"\nQuestion {i+1}: {task_id}")
                print(f"Question: {question[:150]}...")
                print(f"Options: {test_data.get('options', [])[:3]}...")  # Show first 3 options
                print(f"Correct answer: {test_data.get('answer', '')} - {evaluator.get_correct_answer_text(test_data)}")
                print(f"Best answer: {best_answer}")
                print(f"Score: {score:.2f}")
                print(f"Passed: {'✓' if score >= 1.0 else '✗'}")
                print(f"Agent count: {len(responses)}, Average score: {extra_info.get('average_score', 0):.2f}")
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            if evaluator.verbose:
                traceback.print_exc()
            detailed_results.append({
                "task_id": "error",
                "question": question[:200] if isinstance(question, str) else str(question)[:200],
                "score": 0,
                "best_answer": "",
                "message": str(e),
                "is_correct": False,
                "agent_count": 0,
                "average_agent_score": 0,
                "best_agent_id": -1,
                "reference_answer": "",
                "correct_answer_text": ""
            })
            scores.append(0)
            total_count += 1
            continue
    
    # Calculate overall accuracy
    if scores:
        overall_accuracy = np.mean(scores)
        overall_std = np.std(scores) / (len(scores) ** 0.5)
        pass_rate = correct_count / total_count  # Proportion passing
        
        print(f"\n=== Enhanced MMLU Evaluation Results ===")
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
        
        # Statistics for answer distribution
        answer_distribution = {}
        for result in detailed_results:
            answer = result.get("best_answer", "")
            answer_distribution[answer] = answer_distribution.get(answer, 0) + 1
        
        print(f"\n=== Answer Distribution ===")
        for answer in sorted(answer_distribution.keys()):
            count = answer_distribution[answer]
            percentage = count / total_count * 100
            print(f"Answer {answer}: {count} questions ({percentage:.2f}%)")
        
        # Save evaluation results
        eval_results = {
            "evaluation_type": "enhanced_mmlu",
            "total_questions": total_count,
            "correct_count": correct_count,
            "pass_rate": pass_rate,
            "overall_accuracy": overall_accuracy,
            "overall_std": overall_std,
            "score_distribution": score_distribution,
            "answer_distribution": answer_distribution,
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
            f.write("MMLU Enhanced Evaluation Report\n")
            f.write("=============================\n\n")
            f.write("Evaluator Features:\n")
            f.write("- Based on MMLU_ProEvaluator original logic\n")
            f.write("- Multi-level answer extraction strategy (<answer> tag, Answer: pattern, last letter, etc.)\n")
            f.write("- Supports letter-to-number conversion matching\n")
            f.write("- Detailed error classification and logging\n")
            f.write("- Full agent-level analysis\n")
            f.write("- Answer distribution statistics\n\n")
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
            
            f.write("\nAnswer distribution:\n")
            for answer in sorted(answer_distribution.keys()):
                count = answer_distribution[answer]
                percentage = count / total_count * 100
                f.write(f"Answer {answer}: {count} questions ({percentage:.2f}%)\n")
        
        print(f"Enhanced evaluation report saved to: {report_file}")
        
    else:
        print("No valid evaluation results")


if __name__ == "__main__":
    # Default evaluation file
    response_file = "mmlu_pro_3_2.json"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        response_file = sys.argv[1]
    
    print(f"Starting MMLU result file evaluation: {response_file}")
    evaluate_mmlu_results_enhanced(response_file)
