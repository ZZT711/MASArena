'''
This module contains a function to evaluate whether a given number n can be written as the sum of exactly 4 positive even numbers.
'''
def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers.
    Args:
        n (int): The number to evaluate.
    Returns:
        bool: True if n can be expressed as the sum of 4 positive even numbers, False otherwise.
    Example:
        is_equal_to_sum_even(4) == False
        is_equal_to_sum_even(6) == False
        is_equal_to_sum_even(8) == True
    """
    # The smallest sum of 4 positive even numbers (2 + 2 + 2 + 2) is 8
    if n < 8:
        return False
    # Check if n is even
    if n % 2 != 0:
        return False
    # If n is even and >= 8, it can be expressed as the sum of 4 positive even numbers
    return True
# Example usage
if __name__ == "__main__":
    test_values = [4, 6, 8, 10, 12, 14, 16]
    for value in test_values:
        print(f'is_equal_to_sum_even({value}) = {is_equal_to_sum_even(value)}')