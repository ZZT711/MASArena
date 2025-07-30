#!/usr/bin/env python3
"""
Unified Evaluation Framework

This framework integrates HuggingGPT with dataset evaluators and supports:
 - A unified evaluation interface for six datasets
 - Individual dataset testing
 - Automatic response format adaptation
 - Result aggregation and analysis
"""

import json
import time
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

import numpy as np
from langchain.base_language import BaseLanguageModel
from langchain_core.tools import BaseTool

from hugginggpt import HuggingGPT
from format_prompts import get_format_prompt, DatasetType
from evaluators.math_evaluator import MathEvaluator
from evaluators.aime_evaluator import AIMEEvaluator  
from evaluators.bbh_evaluator import BBHEvaluator
from evaluators.drop_evaluator import DROPEvaluator
from evaluators.humaneval_evaluator import HumanEvalEvaluator
from evaluators.mmlu_pro_evaluator import MMLU_ProEvaluator


@dataclass
class EvaluationConfig:
    """Evaluation Configuration"""
    dataset_name: str
    data_path: str
    max_samples: Optional[int] = None
    output_dir: Optional[str] = None
    save_intermediate: bool = False
    random_sample: bool = False


@dataclass 
class EvaluationResult:
    """Evaluation Result"""
    dataset_name: str
    accuracy: float
    total_samples: int
    correct_samples: int
    avg_time_per_sample: float
    failed_samples: int
    detailed_results: List[Dict[str, Any]]
    config: EvaluationConfig


class DatasetLoader:
    """Unified Dataset Loader"""
    
    BASE_DIR = Path(__file__).parent
    DATASET_PATHS = {
        'math': str(BASE_DIR / 'dataset' / 'math_test.jsonl'),
        'aime': str(BASE_DIR / 'dataset' / 'aime_test.jsonl'),
        'bbh': str(BASE_DIR / 'dataset' / 'bbh_test.jsonl'),
        'drop': str(BASE_DIR / 'dataset' / 'drop_test.jsonl'),
        'humaneval': str(BASE_DIR / 'dataset' / 'humaneval_test.jsonl'),
        'mmlu_pro': str(BASE_DIR / 'dataset' / 'mmlu_pro_test.jsonl'),
    }
    
    @classmethod
    def load_dataset(cls, dataset_name: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load a specific dataset"""
        if dataset_name not in cls.DATASET_PATHS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(cls.DATASET_PATHS.keys())}")
        
        dataset_path = cls.DATASET_PATHS[dataset_name]
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    if max_samples and len(data) >= max_samples:
                        break
        
        return data
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get dataset information"""
        dataset_path = cls.DATASET_PATHS[dataset_name]
        if not os.path.exists(dataset_path):
            return {"error": "File not found"}
        
        total_lines = 0
        sample_data = None
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    total_lines += 1
                    if i == 0:  # Save the first sample as an example
                        sample_data = json.loads(line)
        
        return {
            "name": dataset_name,
            "path": dataset_path,
            "total_samples": total_lines,
            "sample_keys": list(sample_data.keys()) if sample_data else [],
            "sample_data": sample_data
        }


class MathUnifiedEvaluator(MathEvaluator):
    """Math Problem Evaluator"""

    def __init__(self, hugginggpt_agent: HuggingGPT, config: EvaluationConfig, dataset_name: str):
        super().__init__(name=dataset_name, config=asdict(config))
        self.agent = hugginggpt_agent
        self.format_prompt = get_format_prompt(dataset_name)
    
    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete evaluation process for a single item"""
        start_time = time.time()
        input_text = item.get("problem", "")

        if not input_text:
            return {'correct': False, 'error': 'Input question is empty.', 'evaluation_time': time.time() - start_time}

        try:
            print('=' * 50)
            # Run agent, its return value may be a string or a dictionary, pass the formatted prompt
            agent_output = self.agent.run_with_trace(input_text, format_prompt=self.format_prompt)

            print("agent_output:", agent_output)
            print('=' * 50)
            # Safety check: ensure the run_result passed to super().evaluate is a dictionary
            if isinstance(agent_output, str):
                run_result = {"messages": [("ai", agent_output)]}
            elif isinstance(agent_output, dict):
                run_result = agent_output
                print("run_result:", run_result)
                print('=' * 50)
            else:
                run_result = {"messages": [("ai", str(agent_output))]}

            # Call the core evaluation logic from `math_evaluator.py`
            evaluation_details = super().evaluate(problem=item, run_result=run_result)
            print("evaluation_details:", evaluation_details)
            print('=' * 50)
            # Convert specific results (math_score) to a general format (correct)
            is_correct = evaluation_details.get("math_score", 0) == 1
            
            print('item_id:', item.get('id', ''),
                'question:', input_text,
                'ground_truth:', self.extract_answer(item.get("solution", "")),
                'extracted_answer:', evaluation_details.get('extracted_answer', ''),
                'full_response:', run_result.get("response", ""),
                'correct:', is_correct,
                'evaluation_time:', time.time() - start_time,
                'dataset:', self.name)
            return {
                'item_id': item.get('id', ''),
                'question': input_text,
                'ground_truth': self.extract_answer(item.get("solution", "")),
                'extracted_answer': evaluation_details.get('extracted_answer', ''),
                'full_response': run_result.get("response", ""),
                'correct': is_correct,
                'evaluation_time': time.time() - start_time,
                'dataset': self.name
            }
        except Exception as e:
            logging.error(f"Error evaluating item {item.get('id', 'N/A')}: {e}", exc_info=True)
            return {
                'item_id': item.get('id', ''),
                'error': str(e),
                'correct': False,
                'evaluation_time': time.time() - start_time,
                'dataset': self.name
            }


class BBHUnifiedEvaluator(BBHEvaluator):
    """BBH Dataset Evaluator"""
    
    def __init__(self, hugginggpt_agent: HuggingGPT, config: EvaluationConfig, dataset_name: str):
        super().__init__(name=dataset_name, config=asdict(config))
        self.agent = hugginggpt_agent
        self.format_prompt = get_format_prompt(dataset_name)

    def _preprocess_bbh_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess BBH data item to match the expected format for the evaluator"""
        processed_item = item.copy()
        
        # BBH evaluator expects field mapping
        processed_item["problem"] = item.get("input", "")
        processed_item["solution"] = item.get("target", "")
        processed_item["id"] = item.get("task_id", "")
        
        return processed_item

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete evaluation process for a single item"""
        # Preprocess data
        processed_item = self._preprocess_bbh_item(item)
        print("processed_item:", processed_item)
        
        # Extract question from data item
        question = processed_item["problem"]
        print("question:", question)
        # Format prompt
        # formatted_question = self.format_prompt.format(question=question)
        
        # Call HuggingGPT agent
        result = self.agent.run_with_trace(question, format_prompt=self.format_prompt)
        
        # Ensure result has the correct format, including final_answer field
        if isinstance(result, str):
            run_result = {"final_answer": result}
        elif isinstance(result, dict):
            run_result = result
            # Extract final_answer from response field
            if "final_answer" not in run_result:
                run_result["final_answer"] = result.get("response", result.get("output", result.get("content", str(result))))
        else:
            run_result = {"final_answer": str(result)}
        
        # Use parent class's evaluate method
        evaluation_details = super().evaluate(processed_item, run_result)
        evaluation_details['correct'] = evaluation_details.get('score', 0) > 0.7
        return evaluation_details


class DROPUnifiedEvaluator(DROPEvaluator):
    """DROP Dataset Evaluator"""
    
    def __init__(self, hugginggpt_agent: HuggingGPT, config: EvaluationConfig, dataset_name: str):
        super().__init__(name=dataset_name, config=asdict(config))
        self.agent = hugginggpt_agent
        self.format_prompt = get_format_prompt(dataset_name)

    def _preprocess_drop_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess DROP data item to match the expected format for the evaluator"""
        processed_item = item.copy()
        
        # Treat context as problem field
        processed_item["problem"] = item.get("context", "")
        
        # Treat ref_text as solution field
        processed_item["solution"] = item.get("ref_text", "") or item.get("completion", "").strip()
        
        return processed_item

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete evaluation process for a single item"""
        start_time = time.time()
        
        # Preprocess data item
        processed_item = self._preprocess_drop_item(item)
        
        input_text = item.get("context", "")
        

        if not input_text.strip():
            return {'correct': False, 'error': 'Input question is empty.', 'evaluation_time': time.time() - start_time}

        try:
            # Run agent, pass the formatted prompt
            agent_output = self.agent.run_with_trace(input_text, format_prompt=self.format_prompt)
            
            # Safety check: ensure the run_result passed to super().evaluate is a dictionary
            if isinstance(agent_output, str):
                run_result = {"final_answer": agent_output}
            elif isinstance(agent_output, dict):
                # Ensure final_answer field is present
                if "final_answer" not in agent_output:
                    # Try to extract the last AI response from messages
                    messages = agent_output.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, tuple) and len(last_message) >= 2:
                            agent_output["final_answer"] = last_message[1]
                        else:
                            agent_output["final_answer"] = str(last_message)
                    else:
                        agent_output["final_answer"] = str(agent_output)
                run_result = agent_output
            else:
                run_result = {"final_answer": str(agent_output)}

            # Call parent class's core evaluation logic, using preprocessed data
            evaluation_details = super().evaluate(problem=processed_item, run_result=run_result)
            print("evaluation_details:",  evaluation_details)
            
            # Convert specific results to a general format
            is_correct = evaluation_details.get("score", 0) >= 0.3  # DROP uses F1 score, >=0.3 is considered correct
            
            return {
                'item_id': item.get('id', ''),
                'question': input_text,
                'ground_truth': processed_item.get("solution", ""),
                'extracted_answer': evaluation_details.get('extracted_answer', ''),
                'full_response': run_result.get("final_answer", ""),
                'correct': is_correct,
                'evaluation_time': time.time() - start_time,
                'dataset': self.name
            }
        except Exception as e:
            logging.error(f"Error evaluating item {item.get('id', 'N/A')}: {e}", exc_info=True)
            return {
                'item_id': item.get('id', ''),
                'error': str(e),
                'correct': False,
                'evaluation_time': time.time() - start_time,
                'dataset': self.name
            }


class HumanEvalUnifiedEvaluator(HumanEvalEvaluator):
    """HumanEval Dataset Evaluator"""
    
    def __init__(self, hugginggpt_agent: HuggingGPT, config: EvaluationConfig, dataset_name: str):
        super().__init__(name=dataset_name, config=asdict(config))
        self.agent = hugginggpt_agent
        self.format_prompt = get_format_prompt(dataset_name)

    def _preprocess_humaneval_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess HumanEval data item to match the expected format for the evaluator"""
        processed_item = item.copy()
        
        # HumanEval evaluator expects field mapping
        processed_item["problem"] = item.get("prompt", "")
        processed_item["id"] = item.get("task_id", "")
        processed_item["test"] = item.get("test", "")
        processed_item["entry_point"] = item.get("entry_point", "")
        
        return processed_item

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete evaluation process for a single item"""
        # Preprocess data
        processed_item = self._preprocess_humaneval_item(item)
        # Extract question from data item
        question = processed_item["problem"]
        
        # Call HuggingGPT agent
        result = self.agent.run_with_trace(question, format_prompt=self.format_prompt)

        # Ensure result has the correct format, including final_answer field
        if isinstance(result, str):
            run_result = {"final_answer": result}
        elif isinstance(result, dict):
            run_result = result
            # Extract final_answer from response field
            if "final_answer" not in run_result:
                run_result["final_answer"] = result.get("response", result.get("output", result.get("content", str(result))))
        else:
            run_result = {"final_answer": str(result)}
        
        # Use parent class's evaluate method
        evaluation_details = super().evaluate(processed_item, run_result)
        evaluation_details['correct'] = evaluation_details.get('score', 0) > 0.7
        return evaluation_details


class MMLU_ProUnifiedEvaluator(MMLU_ProEvaluator):
    """MMLU Pro Dataset Evaluator"""
    
    def __init__(self, hugginggpt_agent: HuggingGPT, config: EvaluationConfig, dataset_name: str):
        # Do not call parent initializer to avoid loading dataset
        # Directly initialize BaseEvaluator
        from evaluators.base_evaluator import BaseEvaluator
        BaseEvaluator.__init__(self, name=dataset_name, config=asdict(config))
        self.agent = hugginggpt_agent
        self.format_prompt = get_format_prompt(dataset_name)
        # Set weights
        self.exact_match_weight = 1.0
        # Set empty dataset to avoid loading errors
        self.dataset = []

    def _preprocess_mmlu_pro_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess MMLU Pro data item to match the expected format for the evaluator"""
        processed_item = item.copy()
        
        # MMLU Pro evaluator expects field mapping
        processed_item["problem"] = item.get("question", "")
        processed_item["solution"] = item.get("answer", "")
        processed_item["id"] = item.get("id", "")
        processed_item["options"] = item.get("options", [])
        
        return processed_item

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete evaluation process for a single item"""
        # Preprocess data
        processed_item = self._preprocess_mmlu_pro_item(item)
        
        # Extract question and options from data item
        question = processed_item["problem"]
        options = processed_item.get("options", [])
        
        # Format question, including options
        if options:
            formatted_options = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
            full_question = f"{question}\n\nOptions:\n{formatted_options}"
        else:
            full_question = question
        
        # Format prompt
        print("format_prompt:", self.format_prompt)
        # Call HuggingGPT agent
        result = self.agent.run_with_trace(full_question, format_prompt=self.format_prompt)
        print("result:", result)
        
        
        # Ensure result has the correct format, including final_answer field
        if isinstance(result, str):
            run_result = {"final_answer": result}
        elif isinstance(result, dict):
            run_result = result
            # Extract final_answer from response field
            if "final_answer" not in run_result:
                run_result["final_answer"] = result.get("response", "")
        else:
            run_result = {"final_answer": result.get("response", "")}
        
        # Use parent class's evaluate method
        evaluation_details = super().evaluate(processed_item, run_result)
        evaluation_details['correct'] = evaluation_details.get('score', 0) > 0.7
        return evaluation_details


class AIMEUnifiedEvaluator(AIMEEvaluator):
    """AIME Math Competition Evaluator"""
    
    def __init__(self, hugginggpt_agent: HuggingGPT, config: EvaluationConfig, dataset_name: str):
        super().__init__(name=dataset_name, config=asdict(config))
        self.agent = hugginggpt_agent
        self.format_prompt = get_format_prompt(dataset_name)

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete evaluation process for a single item"""
        start_time = time.time()
        input_text = item.get("question", "") or item.get("problem", "")

        if not input_text:
            return {'correct': False, 'error': 'Input question is empty.', 'evaluation_time': time.time() - start_time}

        try:
            # Run agent, pass the formatted prompt
            agent_output = self.agent.run_with_trace(input_text, format_prompt=self.format_prompt)
            
            # Safety check: ensure the run_result passed to super().evaluate is a dictionary
            if isinstance(agent_output, str):
                run_result = {"messages": [("ai", agent_output)]}
            elif isinstance(agent_output, dict):
                run_result = agent_output
            else:
                run_result = {"messages": [("ai", str(agent_output))]}

            # Call parent class's core evaluation logic
            evaluation_details = self.evaluate(problem=item, run_result=run_result)
            
            # Convert specific results to a general format
            is_correct = evaluation_details.get("score", 0) == 1
            
            return {
                'item_id': item.get('id', ''),
                'question': input_text,
                'ground_truth': item.get("answer", ""),
                'extracted_answer': evaluation_details.get('extracted_answer', ''),
                'full_response': run_result.get("response", ""),
                'correct': is_correct,
                'evaluation_time': time.time() - start_time,
                'dataset': self.name
            }
        except Exception as e:
            logging.error(f"Error evaluating item {item.get('id', 'N/A')}: {e}", exc_info=True)
            return {
                'item_id': item.get('id', ''),
                'error': str(e),
                'correct': False,
                'evaluation_time': time.time() - start_time,
                'dataset': self.name
            }


class EvaluationManager:
    """Evaluation Manager, coordinates all evaluation components"""
    
    def __init__(self, hugginggpt_agent: HuggingGPT):
        self.agent = hugginggpt_agent
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger("EvaluationManager")
        logger.setLevel(logging.WARNING)  # Reduce log output
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _extract_question(self, item: Dict[str, Any], dataset_name: str) -> str:
        """Extract question text"""
        if dataset_name == "math":
            return item.get("problem", "")
        elif dataset_name == "aime":
            return item.get("question", "") or item.get("problem", "")
        elif dataset_name == "bbh":
            return item.get("input", "")
        elif dataset_name == "drop":
            # Extract question from context field
            context = item.get("context", "")
            if "Question: " in context:
                question_part = context.split("Question: ")[1].split("Answer:")[0].strip()
                return question_part
            return context
        elif dataset_name == "humaneval":
            return item.get("prompt", "")
        elif dataset_name == "mmlu_pro":
            question = item.get("question", "")
            options = item.get("options", [])
            if options:
                formatted_options = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                return f"{question}\n\nOptions:\n{formatted_options}"
            return question
        return item.get("question", "") or item.get("problem", "") or item.get("prompt", "") or item.get("input", "")

    def _extract_expected_answer(self, item: Dict[str, Any], dataset_name: str) -> str:
        """Extract expected answer"""
        if dataset_name == "math":
            return item.get("solution", "")
        elif dataset_name == "aime":
            return item.get("answer", "") or item.get("solution", "")
        elif dataset_name == "bbh":
            return item.get("target", "")
        elif dataset_name == "drop":
            return item.get("ref_text", "") or item.get("completion", "").strip()
        elif dataset_name == "humaneval":
            # HumanEval's answer is code, display canonical_solution
            return item.get("canonical_solution", "")
        elif dataset_name == "mmlu_pro":
            answer = item.get("answer", "")
            if answer:
                return f"({answer})"  # Format as (A), (B) etc.
            return answer
        return item.get("solution", "") or item.get("answer", "") or item.get("target", "") or item.get("ref_text", "")
    
    def evaluate_dataset(self, config: EvaluationConfig) -> EvaluationResult:
        """Evaluate a dataset"""
        print(f"\n🔍 Starting evaluation for dataset: {config.dataset_name}")
        start_time = time.time()
        
        # Check if dataset is supported
        evaluator = self._get_evaluator(config.dataset_name, config)
        
        # Load data
        try:
            dataset = DatasetLoader.load_dataset(config.dataset_name, config.max_samples)
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {config.data_path}")
            raise

        if config.max_samples is not None:
            if config.random_sample:
                import random
                if len(dataset) > config.max_samples:
                    dataset = random.sample(dataset, config.max_samples)
            else:
                dataset = dataset[:config.max_samples]

        if not dataset:
            raise ValueError(f"Dataset {config.dataset_name} failed to load or is empty")
        
        # Record evaluation results
        results = []
        correct_count = 0
        failed_count = 0
        total_time = 0
        
        for i, item in enumerate(dataset):
            start_time = time.time()
            
            # Extract question and answer
            question = self._extract_question(item, config.dataset_name)
            expected_answer = self._extract_expected_answer(item, config.dataset_name)
            
            print(f"\n📝 Question {i+1}/{len(dataset)}")
            print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"Expected answer: {expected_answer}")
            
            try:
                # Use evaluate_single method, which will internally call agent.run_with_trace
                result = evaluator.evaluate_single(item)
                evaluation_time = time.time() - start_time
                total_time += evaluation_time
                
                # Display output and results
                print('=' * 50)
                print("result:", result)
                print('=' * 50)
                predicted_answer = result.get('extracted_answer', 'N/A')
               
                is_correct = result.get('correct', False)

                   
                
                print(f"Output answer: {predicted_answer}")
                print(f"✅ Correct" if is_correct else "❌ Incorrect")
                print(f"⏱️   Time: {evaluation_time:.2f} seconds")
                
                if is_correct:
                    correct_count += 1
                else:
                    failed_count += 1
                
                # Ensure result includes evaluation time
                result['evaluation_time'] = evaluation_time
                results.append(result)
                
                # Save intermediate results
                if config.save_intermediate and (i + 1) % 10 == 0:
                    self._save_intermediate_results(results, config, i + 1)
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                failed_count += 1
                evaluation_time = time.time() - start_time
                total_time += evaluation_time
                results.append({
                    'sample_index': i,
                    'correct': False,
                    'error': str(e),
                    'evaluation_time': evaluation_time
                })
        
        # Calculate final results
        accuracy = correct_count / len(dataset) if dataset else 0.0
        avg_time = total_time / len(dataset) if dataset else 0.0
        
        evaluation_result = EvaluationResult(
            dataset_name=config.dataset_name,
            accuracy=accuracy,
            total_samples=len(dataset),
            correct_samples=correct_count,
            avg_time_per_sample=avg_time,
            failed_samples=failed_count,
            detailed_results=results,
            config=config
        )
        
        # Save final results
        self._save_final_results(evaluation_result)
        
        self.logger.info(f"Evaluation complete! Accuracy: {accuracy:.2%}, Average time: {avg_time:.2f} seconds/sample")
        
        return evaluation_result
    
    def evaluate_all_datasets(self, base_config: EvaluationConfig) -> Dict[str, EvaluationResult]:
        """Evaluate all datasets"""
        results = {}
        
        for dataset_name in DatasetLoader.DATASET_PATHS.keys():
            config = EvaluationConfig(
                dataset_name=dataset_name,
                max_samples=base_config.max_samples,
                output_dir=base_config.output_dir,
                verbose=base_config.verbose,
                save_intermediate=base_config.save_intermediate,
                use_format_prompt=base_config.use_format_prompt
            )
            
            try:
                result = self.evaluate_dataset(config)
                results[dataset_name] = result
            except Exception as e:
                self.logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
                results[dataset_name] = None
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], config: EvaluationConfig, current_count: int):
        """Save intermediate results"""
        output_path = Path(config.output_dir) / f"{config.dataset_name}_intermediate_{current_count}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_final_results(self, result: EvaluationResult):
        """Save final results"""
        output_dir = Path(result.config.output_dir)
        
        # Save detailed results
        detailed_path = output_dir / f"{result.dataset_name}_detailed.json"
        # with open(detailed_path, 'w', encoding='utf-8') as f:
        # json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)
        
        # Save summary results
        summary_path = output_dir / f"{result.dataset_name}_summary.json"
        summary = {
            'dataset_name': result.dataset_name,
            'accuracy': result.accuracy,
            'total_samples': result.total_samples,
            'correct_samples': result.correct_samples,
            'failed_samples': result.failed_samples,
            'avg_time_per_sample': result.avg_time_per_sample
        }
        # with open(summary_path, 'w', encoding='utf-8') as f:
        #     json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def _get_evaluator(self, dataset_name: str, config: EvaluationConfig):
        """Create appropriate evaluator"""
        evaluator_classes = {
            'math': MathUnifiedEvaluator,
            'aime': AIMEUnifiedEvaluator,
            'bbh': BBHUnifiedEvaluator,
            'drop': DROPUnifiedEvaluator,
            'humaneval': HumanEvalUnifiedEvaluator,
            'mmlu_pro': MMLU_ProUnifiedEvaluator
        }
        
        evaluator_class = evaluator_classes.get(dataset_name)
        if not evaluator_class:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return evaluator_class(self.agent, config, dataset_name)
    
    @classmethod
    def get_supported_datasets(cls) -> List[str]:
        """Get list of supported datasets"""
        return list(DatasetLoader.DATASET_PATHS.keys())
    
    @classmethod
    def get_dataset_info_all(cls) -> Dict[str, Dict[str, Any]]:
        """Get information for all datasets"""
        info = {}
        for dataset_name in DatasetLoader.DATASET_PATHS.keys():
            try:
                info[dataset_name] = DatasetLoader.get_dataset_info(dataset_name)
            except Exception as e:
                info[dataset_name] = {"error": str(e)}
        return info 