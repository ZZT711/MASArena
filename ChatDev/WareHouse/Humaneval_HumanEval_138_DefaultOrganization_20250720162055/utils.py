'''
Utility functions for evaluating if a number can be expressed as the sum of 4 positive even numbers.
'''
def is_equal_to_sum_even(n):
    """
    Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers.
    Args:
    n (int): The number to evaluate.
    Returns:
    bool: True if n can be expressed as the sum of 4 positive even numbers, False otherwise.
    """
    # The smallest sum of 4 positive even numbers is 8 (2 + 2 + 2 + 2)
    if n < 8:
        return False
    # Check if n is even
    if n % 2 != 0:
        return False
    # If n is even and >= 8, it can be expressed as the sum of 4 positive even numbers
    return True