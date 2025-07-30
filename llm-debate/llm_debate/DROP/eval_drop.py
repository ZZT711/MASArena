import json
import openai
import numpy as np
import time
import re
import string

def normalize_answer(s):
    """Standardize the answer by removing punctuation, extra spaces, and converting to lowercase."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parse_answer_drop(input_str):
    """Parse DROP answers - extract content within parentheses or key information."""
    # First, try to extract content within parentheses
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, input_str)
    
    if matches:
        # Return the content within the last parenthesis
        return matches[-1].strip()
    
    # If no parentheses, try to extract numbers or keywords
    # Prioritize numbers
    numbers = re.findall(r'\b\d+\b', input_str)
    if numbers:
        return numbers[-1]
    
    # If no numbers, try to extract the answer from the last few words
    words = input_str.strip().split()
    if words:
        # Return the last few words, but no more than 3 words
        return ' '.join(words[-3:]).strip()
    
    return input_str.strip()


def exact_match_score(prediction, ground_truth):
    """Calculate exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    """Calculate F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    
    if len(common_tokens) == 0:
        return 0
    
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


def compute_accuracy_drop(gt, pred_solutions):
    """Calculate DROP accuracy."""
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer_drop(pred_solution)
            
            if pred_answer is not None and pred_answer.strip():
                pred_answers.append(pred_answer)

        if len(pred_answers) == 0:
            return 0, 0
        
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer_drop(pred_solutions)

    if pred_answer is None or not pred_answer.strip():
        return 0, 0
    
    # DROP dataset may have multiple correct answers, separated by |
    ground_truths = gt.split('|') if '|' in gt else [gt]
    
    exact_match = 0
    f1 = 0
    
    for ground_truth in ground_truths:
        exact_match = max(exact_match, exact_match_score(pred_answer, ground_truth.strip()))
        f1 = max(f1, f1_score(pred_answer, ground_truth.strip()))
    
    return exact_match, f1


def most_frequent(List):
    if not List:
        return None
    
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":
    response_dict = json.load(open("drop_3_2.json", "r", encoding="utf-8"))
    questions = list(response_dict.keys())  

    exact_matches = []
    f1_scores = []
    correct_count = 0
    total_count = 0

    for question in questions:
        responses, gt = response_dict[question]

        pred_solutions = []
        for response in responses:
            pred_solution = response[-1]['content']
            pred_solutions.append(pred_solution)

        exact_match, f1 = compute_accuracy_drop(gt, pred_solutions)

        if exact_match is not None and f1 is not None:
            exact_matches.append(float(exact_match))
            f1_scores.append(float(f1))
            total_count += 1
            if exact_match == 1:
                correct_count += 1
        else:
            print(f"Could not parse answer: {gt}")

        print("Exact Match Rate:", np.mean(exact_matches) if exact_matches else 0, 
              "F1 Score:", np.mean(f1_scores) if f1_scores else 0)

    print("=" * 50)
    print("DROP Test Summary:")
    print(f"Total Questions: {total_count}")
    print(f"Correct Exact Matches: {correct_count}")
    print(f"Incorrect Exact Matches: {total_count - correct_count}")
    print(f"Exact Match Rate: {correct_count/total_count:.4f}" if total_count > 0 else "Exact Match Rate: 0.0000")
    print(f"Average Exact Match Rate: {np.mean(exact_matches):.4f}" if exact_matches else "Average Exact Match Rate: 0.0000")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}" if f1_scores else "Average F1 Score: 0.0000")
    print(f"F1 Score Standard Deviation: {np.std(f1_scores):.4f}" if f1_scores else "F1 Score Standard Deviation: 0.0000") 