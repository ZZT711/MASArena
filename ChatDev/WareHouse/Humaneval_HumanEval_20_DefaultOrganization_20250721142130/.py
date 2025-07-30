def find_closest_elements(numbers):
    """
    From a supplied list of numbers (of length at least two) select and return two that are the closest to each
    other and return them in order (smaller number, larger number).
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
    (2.0, 2.2)
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
    (2.0, 2.0)
    """
    if len(numbers) < 2:
        raise ValueError("The list must contain at least two elements.")
    # Sort the numbers
    sorted_numbers = sort_numbers(numbers)
    # Initialize the closest pair and the smallest difference
    closest_pair = (sorted_numbers[0], sorted_numbers[1])
    smallest_diff = abs(sorted_numbers[1] - sorted_numbers[0])
    # Iterate through the sorted list to find the closest pair
    for i in range(1, len(sorted_numbers)):
        current_diff = abs(sorted_numbers[i] - sorted_numbers[i - 1])
        if current_diff < smallest_diff:
            smallest_diff = current_diff
            closest_pair = (sorted_numbers[i - 1], sorted_numbers[i])
    return closest_pair
def sort_numbers(numbers):
    """
    Sorts the list of numbers in ascending order using a simple bubble sort algorithm.
    >>> sort_numbers([3.0, 1.0, 2.0])
    [1.0, 2.0, 3.0]
    """
    n = len(numbers)
    for i in range(n):
        for j in range(0, n-i-1):
            if numbers[j] > numbers[j+1]:
                # Swap if the element found is greater than the next element
                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
    return numbers