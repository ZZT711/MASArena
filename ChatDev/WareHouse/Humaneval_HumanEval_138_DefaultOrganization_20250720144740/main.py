'''
This file contains the implementation of the function is_equal_to_sum_even,
which evaluates whether the given number n can be written as the sum of exactly 
4 positive even numbers.
'''
def is_equal_to_sum_even(n):
    """
    Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers.
    Example
    is_equal_to_sum_even(4) == False
    is_equal_to_sum_even(6) == False
    is_equal_to_sum_even(8) == True
    """
    # The smallest sum of 4 positive even numbers (2 + 2 + 2 + 2) is 8.
    # Therefore, if n is less than 8, it cannot be expressed as such.
    if n < 8:
        return False
    # If n is even and greater than or equal to 8, it can be expressed as the sum of 4 positive even numbers.
    return n % 2 == 0