'''
This module provides a function to check if any two numbers in a given list
are closer to each other than a specified threshold.
Function:
- has_close_elements(numbers: List[float], threshold: float) -> bool:
    Checks if any two numbers in the list are closer than the given threshold.
'''
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    Args:
        numbers (List[float]): A list of floating-point numbers.
        threshold (float): The threshold distance to check against.
    Returns:
        bool: True if any two numbers are closer than the threshold, False otherwise.
    Examples:
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
    """
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False