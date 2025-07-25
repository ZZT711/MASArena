"""
AIME Evaluator

Standalone evaluator for AIME-style math problems.
"""

import re
import time
from typing import Dict, Any, Tuple
from pathlib import Path
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from .math_evaluator import MathEvaluator

class AIMEEvaluator(MathEvaluator):
    """
    Evaluator for AIME-style math problems.
    Extracts answers and compares with expected answers (numeric/string match).
    """
    def __init__(self, name: str = "aime", config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.data_path = config.get("data_path", f"benchmark/data/{name}_test.jsonl")
        self.log_path = config.get("log_path", f"benchmark/data/results/{name.upper()}")
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        self.run_evaluator = RunEvaluator()

    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: int) -> Run:
        problem["problem"] = problem["question"]
        problem["solution"] = problem["answer"]
        return super().create_run(problem, final_answer, extracted_answer, score)
    

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        
        # Extract the final answer from messages

        all_messages = run_result.get("messages", [])
        final_answer = self.extract_final_answer(all_messages)

        score, extracted_answer = super().calculate_score(problem["answer"], final_answer)
        run = self.create_run(problem, final_answer, extracted_answer, score)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "run_evaluation": run_evaluation,
        } 