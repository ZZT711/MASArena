'''
This file contains the implementation of the function is_equal_to_sum_even,
which evaluates whether the given number n can be written as the sum of
exactly 4 positive even numbers.
'''
def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers
    Example
    is_equal_to_sum_even(4) == False
    is_equal_to_sum_even(6) == False
    is_equal_to_sum_even(8) == True
    """
    # The smallest sum of 4 positive even numbers (2 + 2 + 2 + 2) is 8.
    if n < 8:
        return False
    # Check if n is even, as the sum of even numbers is always even.
    if n % 2 != 0:
        return False
    # If n is even and at least 8, it can be expressed as the sum of 4 positive even numbers.
    return True
# Example usage
if __name__ == "__main__":
    test_values = [4, 6, 8, 10, 12, 14, 16, 18]
    for value in test_values:
        print(f'is_equal_to_sum_even({value}) == {is_equal_to_sum_even(value)}')