def count_ones(n):
    """
    Count the number of ones in the binary representation of a non-negative integer.
    Args:
        n (int): A non-negative integer.
    Returns:
        int: The count of ones in the binary representation of n.
    """
    return bin(n).count('1')
def sort_array(arr):
    """
    Sort an array of non-negative integers according to the number of ones in their
    binary representation in ascending order. For similar number of ones, sort based
    on decimal value.
    Args:
        arr (list): A list of non-negative integers.
    Returns:
        list: The sorted list of integers.
    Examples:
        >>> sort_array([1, 5, 2, 3, 4])
        [1, 2, 3, 4, 5]
        >>> sort_array([0, 1, 2, 3, 4])
        [0, 1, 2, 3, 4]
    """
    # Sort the array using a custom key
    return sorted(arr, key=lambda x: (count_ones(x), x))
# Example usage (uncomment to test)
# print(sort_array([1, 5, 2, 3, 4]))  # Output: [1, 2, 3, 4, 5]
# print(sort_array([0, 1, 2, 3, 4]))  # Output: [0, 1, 2, 3, 4]