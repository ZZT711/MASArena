import pickle
import json
import re
import os
import sys
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

# Add evaluators module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.append(project_root)

from evaluators.math_evaluator import MathEvaluator

# Load environment variables from .env file
load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=40,
    base_url=os.getenv("OPENAI_API_BASE")
)
# Set OpenAI API



def read_jsonl(path: str):
    """Read JSONL file"""
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def evaluate_responses(agent_contexts, ground_truth, math_evaluator):
    """Evaluate multiple agent responses, using aggregated answers as the final answer"""
    if not agent_contexts:
        return False, None, None, None
    
    # Extract the last response from each agent
    agent_answers = []
    for agent_context in agent_contexts:
        if agent_context:
            last_response = agent_context[-1]['content']
            agent_answers.append(last_response)
        else:
            agent_answers.append(None)
    
    # Filter out empty answers
    valid_answers = [ans for ans in agent_answers if ans]
    
    if not valid_answers:
        return False, None, None, None

    aggregated_answer = valid_answers[-1]
    
    # Use MathEvaluator to evaluate the aggregated answer
    score, extracted_prediction = math_evaluator.calculate_score(ground_truth, aggregated_answer)
    extracted_ground_truth = math_evaluator.extract_answer(ground_truth)
    
    is_correct = score == 1
    
    return is_correct, extracted_prediction, extracted_ground_truth, {
        'agent_answers': agent_answers,
        'aggregated_answer': aggregated_answer
    }


def main():
    # Initialize MathEvaluator with config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "dataset", "math_test.jsonl")
    math_evaluator = MathEvaluator(config={
        "data_path": dataset_path,
        "log_path": os.path.join(current_dir, "logs", "math_evaluation")
    })
    
    # Load generated results
    try:
        generated_data = pickle.load(open("math_agents2_rounds3.p", "rb"))
    except FileNotFoundError:
        print("Error: Generated results file math_agents2_rounds3.p not found")
        print("Please run gen_math.py to generate results first")
        return
    
    # Load original dataset
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "..", "dataset", "math_test.jsonl")
        math_questions = read_jsonl(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file {dataset_path} not found")
        return
    
    correct_count = 0
    total_count = 0
    detailed_results = []
    
    print("Starting evaluation of MAS on the math test set...")
    print("Using MathEvaluator for evaluation, aggregating answers from multiple agents via LLM")
    print("=" * 70)
    
    # Evaluate each question
    for question, (agent_contexts, solution) in tqdm(generated_data.items()):
        is_correct, pred_answer, true_answer, answer_details = evaluate_responses(
            agent_contexts, solution, math_evaluator
        )
        
        total_count += 1
        if is_correct:
            correct_count += 1
        
        detailed_results.append({
            'question': question[:100] + "..." if len(question) > 100 else question,
            'predicted': pred_answer,
            'ground_truth': true_answer,
            'correct': is_correct,
            'agent_answers': [math_evaluator.extract_answer(ans) if ans else None 
                            for ans in answer_details['agent_answers']],
            'aggregated_answer': answer_details['aggregated_answer'],
            'final_prediction_source': 'llm_aggregated_answer'
        })
        
        # Display progress in real-time
        accuracy = correct_count / total_count
        print(f"\rCurrent progress: {correct_count}/{total_count} ({accuracy:.1%})", end="", flush=True)
    
    print()  # Newline
    print("=" * 70)
    
    # Final results
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Evaluation complete!")
    print(f"Total questions: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {correct_count}/{total_count} = {final_accuracy:.1%}")
    
    # Display some specific examples
    print("\nTop 5 detailed results:")
    print("-" * 70)
    for i, result in enumerate(detailed_results[:5]):
        status = "✓" if result['correct'] else "✗"
        print(f"{i+1}. {status} Question: {result['question']}")
        print(f"   Predicted answer: {result['predicted']}")
        print(f"   Ground truth: {result['ground_truth']}")
        print(f"   Agent answers: {result['agent_answers']}")
        print(f"   Aggregated answer: {result['aggregated_answer'][:200]}..." if len(result['aggregated_answer']) > 200 else f"   Aggregated answer: {result['aggregated_answer']}")
        print(f"   Evaluation method: {result['final_prediction_source']}")
        print()
    
    # Save detailed results
    with open("math_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": total_count,
            "correct_answers": correct_count,
            "accuracy": final_accuracy,
            "evaluation_method": "MathEvaluator with LLM aggregated answers",
            "detailed_results": detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Detailed results saved to math_evaluation_results.json")


if __name__ == "__main__":
    main() 