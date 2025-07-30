'''
Module to check if a number can be expressed as the sum of exactly 4 positive even numbers.
'''
def is_equal_to_sum_even(n):
    if n < 8:  # The smallest sum of 4 positive even numbers (2 + 2 + 2 + 2)
        return False
    # Check if n is even and at least 8
    return n % 2 == 0