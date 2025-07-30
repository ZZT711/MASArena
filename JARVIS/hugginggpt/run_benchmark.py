"""
Example Usage of the Evaluation Framework

Demonstrates how to use the HuggingGPT evaluation framework to evaluate different datasets.
"""

import logging
from config import (
    DEFAULT_CONFIG, MATH_OPTIMIZED_CONFIG, CODING_OPTIMIZED_CONFIG,
    create_llm_from_config
)
from evaluation_framework import EvaluationManager, EvaluationConfig
from hugginggpt import HuggingGPT



def math_evaluation():
    """Math Problem Evaluation"""
    print("\n=== Math Problem Evaluation ===")
    
    llm = create_llm_from_config(MATH_OPTIMIZED_CONFIG.llm_config, "openai")
    tools = []
    
    hugginggpt = HuggingGPT(llm, tools)
    manager = EvaluationManager(hugginggpt)
    
    config = EvaluationConfig(
        dataset_name="math",
        data_path="hugginggpt/dataset/math_test.jsonl",
        max_samples=1,
        save_intermediate=True,
        output_dir="results/math/math"
    )
    
    result = manager.evaluate_dataset(config)
    
    print(f"Math evaluation accuracy: {result.accuracy:.3f}")
    return result

def aime_evaluation():
    """AIME Problem Evaluation"""
    print("\n=== AIME Problem Evaluation ===")
    
    llm = create_llm_from_config(MATH_OPTIMIZED_CONFIG.llm_config, "openai")
    tools = []
    
    hugginggpt = HuggingGPT(llm, tools)
    manager = EvaluationManager(hugginggpt)
    
    config = EvaluationConfig(
        dataset_name="aime",
        data_path="hugginggpt/dataset/aime_test.jsonl", 
        max_samples=1,
        save_intermediate=True,
        output_dir="results/aime/aime"
    )
    
    result = manager.evaluate_dataset(config)
    
    print(f"AIME evaluation accuracy: {result.accuracy:.3f}")
    return result

def drop_evaluation():
    """DROP Problem Evaluation"""
    print("\n=== DROP Problem Evaluation ===")
    
    llm = create_llm_from_config(MATH_OPTIMIZED_CONFIG.llm_config, "openai")
    tools = []

    hugginggpt = HuggingGPT(llm, tools)
    manager = EvaluationManager(hugginggpt)
    
    config = EvaluationConfig(
        dataset_name="drop",
        data_path="hugginggpt/dataset/drop_test.jsonl",
        max_samples=1,
        save_intermediate=True,
        output_dir="results/drop/drop"
    )
    
    result = manager.evaluate_dataset(config)
    
    print(f"DROP evaluation accuracy: {result.accuracy:.4f}")
    return result

def mmlu_evaluation():
    """MMLU Problem Evaluation"""
    print("\n=== MMLU Problem Evaluation ===")
    
    llm = create_llm_from_config(DEFAULT_CONFIG.llm_config, "openai")
    tools = []
    
    hugginggpt = HuggingGPT(llm, tools)
    manager = EvaluationManager(hugginggpt)
    
    config = EvaluationConfig(
        dataset_name="mmlu_pro",
        data_path="hugginggpt/dataset/mmlu_pro_test.jsonl",
        max_samples=1,
        save_intermediate=True,
        output_dir="results/mmlu/mmlu"
    )
    
    result = manager.evaluate_dataset(config)
    print(f"MMLU evaluation accuracy: {result.accuracy:.3f}")
    return result
    
def bbh_evaluation():
    """BBH Problem Evaluation"""
    print("\n=== BBH Problem Evaluation ===")
    
    llm = create_llm_from_config(DEFAULT_CONFIG.llm_config, "openai")
    tools = []
    
    hugginggpt = HuggingGPT(llm, tools)
    manager = EvaluationManager(hugginggpt)
    
    config = EvaluationConfig(
        dataset_name="bbh",
        data_path="hugginggpt/dataset/bbh_test.jsonl",
        max_samples=1,
        save_intermediate=True,
        output_dir="results/bbh/bbh"
    )
    
    result = manager.evaluate_dataset(config)
    print(f"BBH evaluation accuracy: {result.accuracy:.3f}")
    return result

def humaneval_evaluation():
    """HumanEval Problem Evaluation"""
    print("\n=== HumanEval Problem Evaluation ===")
    
    llm = create_llm_from_config(CODING_OPTIMIZED_CONFIG.llm_config, "openai")
    tools = []
    
    hugginggpt = HuggingGPT(llm, tools)
    manager = EvaluationManager(hugginggpt)
    
    config = EvaluationConfig(
        dataset_name="humaneval",
        data_path="hugginggpt/dataset/humaneval_test.jsonl",
        max_samples=1,
        save_intermediate=True,
        output_dir="results/humaneval/humaneval",
        random_sample=True
    )
    
    result = manager.evaluate_dataset(config)
    print(f"HumanEval evaluation accuracy: {result.accuracy:.3f}") 
    return result

def main():
    """Run example evaluations"""
    logging.basicConfig(level=logging.INFO)
    
    try:
         math_evaluation()
        # aime_evaluation()
        # drop_evaluation()
        # mmlu_evaluation()
        # bbh_evaluation()
        # humaneval_evaluation()
    except Exception as e:
        print(f"Example run failed: {e}")
        logging.error(f"Example run failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 