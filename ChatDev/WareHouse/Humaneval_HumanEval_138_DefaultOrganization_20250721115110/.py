def is_equal_to_sum_even(n):
    """
    Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers.
    Parameters:
    n (int): The number to evaluate.
    Returns:
    bool: True if n can be expressed as the sum of 4 positive even numbers, False otherwise.
    Example:
    is_equal_to_sum_even(4) == False
    is_equal_to_sum_even(6) == False
    is_equal_to_sum_even(8) == True
    """
    # Check if n is less than 8
    if n < 8:
        return False
    # Check if n is odd
    if n % 2 != 0:
        return False
    # If n is even and >= 8, it can be expressed as the sum of 4 positive even numbers
    return True