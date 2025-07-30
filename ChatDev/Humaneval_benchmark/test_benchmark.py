#!/usr/bin/env python3
"""
Test script: Verify that the HumanEval evaluation system works correctly.
Only tests 2 tasks to ensure the entire process runs through.
"""

import os
import sys

# Add paths
root = os.path.dirname(__file__)
chatdev_root = os.path.dirname(root)
sys.path.insert(0, chatdev_root)
sys.path.append(root)

from run_benchmark import run_benchmark

def test_small_benchmark():
    """Test small-scale benchmark (1 task)"""
    print("Starting test of ChatDev HumanEval evaluation system...")
    print("Test size: 1 task")
    
    try:
        # Run benchmark for 2 tasks
        summary = run_benchmark(num_tasks=3)
        
        print(f"\nTest completed!")
        print(f"Task success rate: {summary.accuracy:.1%}")
        print(f"Total execution time: {summary.total_execution_time:.1f}s")
        
        # Output detailed results for each task
        print("\nDetailed results:")
        for detail in summary.details:
            status = "✓ Pass" if detail.success else "✗ Fail"
            print(f"  {detail.task_id}: {status} ({detail.execution_time:.1f}s)")
            if detail.code_generated:
                print(f"    Generated code length: {len(detail.code_generated)} characters")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_benchmark()
    
    if success:
        print("\n✓ Test succeeded! The system is functioning properly.")
        print("You can now run the full 100-task benchmark:")
        print("python run_benchmark.py --num_tasks 100")
    else:
        print("\n✗ Test failed! Please check system configuration.")
        sys.exit(1) 