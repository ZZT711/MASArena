import json
import openai
import numpy as np
import time
import re

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def solve_math_problems(input_str):
    """Extract numbers from a string."""
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]
    return None


def parse_answer_aime(input_str):
    """Parse AIME answer - primarily extract numbers within parentheses or the last number."""
    # First, try to extract content within parentheses
    pattern = r'\((\d+\.?\d*)\)'
    matches = re.findall(pattern, input_str)
    
    if matches:
        # Return the last number within the parentheses
        return matches[-1]
    
    # If no parentheses, try to extract all numbers, return the last one
    return solve_math_problems(input_str)


def compute_accuracy_aime(gt, pred_solutions):
    """Calculate AIME accuracy."""
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer_aime(pred_solution)
            
            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if len(pred_answers) == 0:
            return 0
        
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer_aime(pred_solutions)

    if pred_answer is None:
        return 0
    
    # AIME answers are integers, standardize before string comparison
    try:
        gt_num = str(int(float(gt)))
        pred_num = str(int(float(pred_answer)))
        if gt_num == pred_num:
            return 1
        else:
            return 0
    except:
        # If conversion fails, compare strings directly
        if str(gt).strip() == str(pred_answer).strip():
            return 1
        else:
            return 0


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
    response_dict = json.load(open("aime_3_2.json", "r", encoding="utf-8"))
    questions = list(response_dict.keys())

    accuracies = []
    correct_count = 0
    total_count = 0

    for question in questions:
        responses, gt = response_dict[question]

        pred_solutions = []
        for response in responses:
            pred_solution = response[-1]['content']
            pred_solutions.append(pred_solution)

        accurate = compute_accuracy_aime(gt, pred_solutions)

        if accurate is not None:
            accuracies.append(float(accurate))
            total_count += 1
            if accurate == 1:
                correct_count += 1
        else:
            print(f"Could not parse answer: {gt}")

        print("Accuracy:", np.mean(accuracies), "Standard Error:", np.std(accuracies) / (len(accuracies) ** 0.5))

    print("=" * 50)
    print("AIME Test Summary:")
    print(f"Total Questions: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {total_count - correct_count}")
    print(f"Accuracy: {correct_count/total_count:.4f}" if total_count > 0 else "Accuracy: 0.0000")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}" if accuracies else "Average Accuracy: 0.0000")
    print(f"Standard Deviation: {np.std(accuracies):.4f}" if accuracies else "Standard Deviation: 0.0000") 