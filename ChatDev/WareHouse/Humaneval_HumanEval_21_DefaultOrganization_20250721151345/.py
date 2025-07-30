from typing import List
def find_min(numbers: List[float]) -> float:
    """Return the minimum value from the list of numbers."""
    min_value = numbers[0]
    for number in numbers:
        if number < min_value:
            min_value = number
    return min_value
def find_max(numbers: List[float]) -> float:
    """Return the maximum value from the list of numbers."""
    max_value = numbers[0]
    for number in numbers:
        if number > max_value:
            max_value = number
    return max_value
def rescale_to_unit(numbers: List[float]) -> List[float]:
    """Given list of numbers (of at least two elements), apply a linear transform to that list,
    such that the smallest number will become 0 and the largest will become 1.
    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if len(numbers) < 2:
        raise ValueError("The list must contain at least two elements.")
    min_value = find_min(numbers)
    max_value = find_max(numbers)
    if max_value == min_value:
        raise ValueError("All numbers in the list are the same. Cannot rescale.")
    return [(number - min_value) / (max_value - min_value) for number in numbers]