from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class HumanevalTask:
    """Data structure for a single HumanEval test task"""
    task_id: str               # Unique identifier in JSONL, e.g., "HumanEval_84"
    prompt: str                # Problem description, including function signature and docstring
    entry_point: str           # Function entry point name
    canonical_solution: str    # Reference solution code
    test: str                  # Test case code block
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HumanevalTask':
        """Create a HumanevalTask instance from a dictionary"""
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            entry_point=data["entry_point"],
            canonical_solution=data["canonical_solution"],
            test=data["test"]
        )


@dataclass
class TestCaseResult:
    """Execution result of a single test case"""
    index: int                 # Test case index
    passed: bool               # Whether passed
    expected: Any              # Expected value
    actual: Any                # Actual value
    error: Optional[str] = ""  # Error message (if any)


@dataclass
class EvaluationResult:
    """Evaluation result for a single task"""
    task_id: str
    results: List[TestCaseResult]
    passed_count: int
    total_count: int
    success: bool              # Whether all test cases passed
    code_generated: str        # Generated code
    execution_time: float      # Execution time (seconds)
    
    @property
    def accuracy(self) -> float:
        """Compute accuracy"""
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0


@dataclass
class BenchmarkSummary:
    """Summary of overall evaluation results"""
    total_tasks: int
    passed_tasks: int          # Number of tasks completely passed
    accuracy: float            # Task-level accuracy (passed_tasks / total_tasks)
    avg_test_accuracy: float   # Average test-case-level accuracy
    total_execution_time: float # Total execution time
    details: List[EvaluationResult]  # Detailed results for each task
    
    def __str__(self) -> str:
        """Format output results"""
        return f"""
=== HumanEval Evaluation Results ===
Total tasks: {self.total_tasks}
Tasks fully passed: {self.passed_tasks}
Task pass rate: {self.accuracy:.2%}
Average test-case pass rate: {self.avg_test_accuracy:.2%}
Total execution time: {self.total_execution_time:.2f}s
Average time per task: {self.total_execution_time/self.total_tasks:.2f}s
""" 