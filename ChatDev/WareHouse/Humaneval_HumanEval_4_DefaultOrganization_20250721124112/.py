from typing import List
def mean(numbers: List[float]) -> float:
    """
    Calculate the mean of a list of numbers.
    Args:
        numbers (List[float]): A list of floating-point numbers.
    Returns:
        float: The mean of the numbers.
    Raises:
        ValueError: If the list is empty.
    """
    if not numbers:
        raise ValueError("The list of numbers cannot be empty.")
    return sum(numbers) / len(numbers)
def mean_absolute_deviation(numbers: List[float]) -> float:
    """
    For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    Args:
        numbers (List[float]): A list of floating-point numbers.
    Returns:
        float: The Mean Absolute Deviation of the numbers.
    Raises:
        ValueError: If the list is empty.
    Examples:
        >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
        1.0
    """
    if not numbers:
        raise ValueError("The list of numbers cannot be empty.")
    x_mean = mean(numbers)
    absolute_deviations = [abs(x - x_mean) for x in numbers]
    return mean(absolute_deviations)
# Example usage (uncomment to test):
# print(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]))