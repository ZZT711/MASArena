'''
This script contains a function to evaluate whether a given number n can be written as the sum of exactly 4 positive even numbers.
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
    # The minimum sum of four positive even numbers (2 + 2 + 2 + 2) is 8.
    if n < 8:
        return False
    # Check if n is even, as the sum of even numbers is always even.
    if n % 2 != 0:
        return False
    # If n is even and at least 8, it can be expressed as the sum of four positive even numbers.
    return True
# Example usage
if __name__ == "__main__":
    test_cases = [4, 6, 8, 10, 12, 14, 16]
    for case in test_cases:
        print(f'is_equal_to_sum_even({case}) = {is_equal_to_sum_even(case)}')