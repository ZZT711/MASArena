#!/usr/bin/env python3
"""
Simplified execution script: directly run the full evaluation of 100 HumanEval tasks

Usage:
python run_100_tasks.py
"""

import os
import sys

# Add paths
root = os.path.dirname(__file__)
chatdev_root = os.path.dirname(root)
sys.path.insert(0, chatdev_root)
sys.path.append(root)

from run_benchmark import run_benchmark

def main():
    """Run full evaluation of 100 tasks"""
    print("=" * 50)
    print("ChatDev HumanEval full evaluation")
    print("Number of tasks: 100")
    print("Estimated time: 1-2 hours")
    print("=" * 50)
    
    # Confirm start
    response = input("Start evaluation? (y/N): ").strip().lower()
    if response not in ['y', 'yes', '是']:
        print("Evaluation cancelled.")
        return
    
    print("\nStarting evaluation...")
    
    try:
        # Run benchmark for 100 tasks
        summary = run_benchmark(num_tasks=100)
        
        print("\n" + "=" * 50)
        print("Evaluation completed!")
        print(f"Final result: {summary.passed_tasks}/{summary.total_tasks} tasks passed")
        print(f"Accuracy: {summary.accuracy:.1%}")
        print(f"Total time: {summary.total_execution_time/60:.1f} minutes")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 