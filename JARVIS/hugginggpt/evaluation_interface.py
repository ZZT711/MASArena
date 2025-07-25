from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import json
import re

from langchain.base_language import BaseLanguageModel
from langchain_core.tools import BaseTool

from hugginggpt import HuggingGPT


class BaseEvaluator(ABC):
    """Base class for dataset evaluators."""
    
    def __init__(self, hugginggpt_agent: HuggingGPT):
        self.agent = hugginggpt_agent
    
    @abstractmethod
    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single item from the dataset."""
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """Extract the final answer from the agent's response."""
        pass
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]], 
                        max_items: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate the entire dataset."""
        if max_items:
            dataset = dataset[:max_items]
        
        results = []
        correct = 0
        total = 0
        
        for i, item in enumerate(dataset):
            print(f"Evaluating item {i+1}/{len(dataset)}")
            result = self.evaluate_single(item)
            results.append(result)
            
            if result.get('correct', False):
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }


class MathEvaluator(BaseEvaluator):
    """Evaluator for math datasets (GSM8K, MATH, etc.)."""
    
    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item.get('problem', item.get('question', ''))
        ground_truth = item.get('solution', item.get('answer', ''))
        
        # Use HuggingGPT to solve the math problem
        response = self.agent.run(f"Solve this math problem: {question}")
        predicted_answer = self.extract_answer(response)
        
        # Compare answers
        is_correct = self._compare_math_answers(predicted_answer, ground_truth)
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': predicted_answer,
            'full_response': response,
            'correct': is_correct
        }
    
    def extract_answer(self, response: str) -> str:
        """Extract numerical answer from response."""
        # Look for final answer patterns
        patterns = [
            r'答案是[：:]?\s*([0-9]+(?:\.[0-9]+)?)',
            r'final answer[：:]?\s*([0-9]+(?:\.[0-9]+)?)',
            r'结果为[：:]?\s*([0-9]+(?:\.[0-9]+)?)',
            r'(?:=\s*)?([0-9]+(?:\.[0-9]+)?)(?:\s*$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: extract last number
        numbers = re.findall(r'[0-9]+(?:\.[0-9]+)?', response)
        return numbers[-1] if numbers else ""
    
    def _compare_math_answers(self, pred: str, truth: str) -> bool:
        """Compare mathematical answers with tolerance."""
        try:
            pred_val = float(pred) if pred else None
            truth_val = float(re.findall(r'[0-9]+(?:\.[0-9]+)?', str(truth))[-1]) if truth else None
            
            if pred_val is None or truth_val is None:
                return False
            
            return abs(pred_val - truth_val) < 1e-6
        except:
            return pred.strip() == str(truth).strip()


class DropEvaluator(BaseEvaluator):
    """Evaluator for DROP (Reading Comprehension) dataset."""
    
    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        passage = item.get('passage', '')
        question = item.get('question', '')
        answers = item.get('answers_spans', {}).get('spans', [])
        
        # Format input for HuggingGPT
        input_text = f"Answer the following question based on the passage:\nPassage: {passage}\nQuestion: {question}"
        response = self.agent.run(input_text)
        predicted_answer = self.extract_answer(response)
        
        # Check if prediction matches any ground truth answer
        is_correct = any(self._normalize_text(predicted_answer) == self._normalize_text(ans) 
                        for ans in answers)
        
        return {
            'passage': passage,
            'question': question,
            'ground_truth_answers': answers,
            'predicted_answer': predicted_answer,
            'full_response': response,
            'correct': is_correct
        }
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from response (English prompts)."""
        # Look for answer patterns in English
        patterns = [
            r'Answer is[: ]?\s*(.+?)(?:\n|$)',
            r'Answer[: ]?\s*(.+?)(?:\n|$)',
            r'Response[: ]?\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use last sentence
        sentences = response.split('。')
        return sentences[-2].strip() if len(sentences) > 1 else response.strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'\s+', ' ', text.lower().strip())


class MMLUEvaluator(BaseEvaluator):
    """Evaluator for MMLU (Multiple Choice) dataset."""
    
    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item.get('question', '')
        choices = item.get('choices', [])
        answer = item.get('answer', 0)  # Usually index of correct answer
        
        # Format choices
        choice_text = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        input_text = f"Please answer the following multiple-choice question:\n{question}\n{choice_text}\nSelect the correct answer (A/B/C/D):"
        
        response = self.agent.run(input_text)
        predicted_choice = self.extract_answer(response)
        
        # Convert to index
        pred_index = ord(predicted_choice.upper()) - ord('A') if predicted_choice else -1
        is_correct = pred_index == answer
        
        return {
            'question': question,
            'choices': choices,
            'ground_truth_index': answer,
            'predicted_choice': predicted_choice,
            'predicted_index': pred_index,
            'full_response': response,
            'correct': is_correct
        }
    
    def extract_answer(self, response: str) -> str:
        """Extract choice (A/B/C/D) from response (English prompts)."""
        # Look for choice patterns in English
        match = re.search(r'Answer[: ]?\s*([ABCD])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Look for standalone choice
        match = re.search(r'\b([ABCD])\b', response)
        if match:
            return match.group(1).upper()
        
        return ""


class HumanEvalEvaluator(BaseEvaluator):
    """Evaluator for HumanEval (Code Generation) dataset."""
    
    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        prompt = item.get('prompt', '')
        canonical_solution = item.get('canonical_solution', '')
        test = item.get('test', '')
        
        # Ask HuggingGPT to generate code
        input_text = f"Please complete the following Python function:\n{prompt}"
        response = self.agent.run(input_text)
        generated_code = self.extract_answer(response)
        
        # Test the generated code
        is_correct = self._test_code(generated_code, test, prompt)
        
        return {
            'prompt': prompt,
            'canonical_solution': canonical_solution,
            'generated_code': generated_code,
            'full_response': response,
            'test_passed': is_correct,
            'correct': is_correct
        }
    
    def extract_answer(self, response: str) -> str:
        """Extract Python code from response."""
        # Look for code blocks
        code_pattern = r'```python\n(.*?)\n```'
        match = re.search(code_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Look for function definitions
        func_pattern = r'(def\s+\w+.*?)(?=\n\ndef|\n\n[A-Z]|\Z)'
        match = re.search(func_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        return response
    
    def _test_code(self, code: str, test: str, prompt: str) -> bool:
        """Test if generated code passes the test cases."""
        try:
            # Combine prompt + generated code + test
            full_code = prompt + code + "\n" + test
            
            # Execute in isolated environment
            exec_globals = {}
            exec(full_code, exec_globals)
            return True
        except Exception as e:
            print(f"Code execution failed: {e}")
            return False


def create_evaluator(dataset_type: str, hugginggpt_agent: HuggingGPT) -> BaseEvaluator:
    """Factory function to create appropriate evaluator."""
    evaluators = {
        'math': MathEvaluator,
        'drop': DropEvaluator, 
        'mmlu': MMLUEvaluator,
        'humaneval': HumanEvalEvaluator
    }
    
    if dataset_type not in evaluators:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return evaluators[dataset_type](hugginggpt_agent) 