'''
Module to evaluate if a number can be expressed as the sum of four positive even numbers.
'''
def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers.
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
    # If n is even and >= 8, it can be expressed as the sum of 4 even numbers
    return True