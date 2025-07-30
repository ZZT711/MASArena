def starts_one_ends(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit
    positive integers that start or end with 1.
    Parameters:
    n (int): The number of digits in the positive integers to consider.
    Returns:
    int: The count of n-digit positive integers that start or end with 1.
    """
    if n < 1:
        return 0  # Edge case: no n-digit numbers if n is less than 1
    # Count of n-digit numbers starting with '1'
    count_starting_with_1 = 10 ** (n - 1)
    # Count of n-digit numbers ending with '1'
    count_ending_with_1 = 10 ** (n - 1)
    # Count of n-digit numbers starting and ending with '1'
    count_starting_and_ending_with_1 = 10 ** (n - 2) if n > 1 else 0
    # Total count
    total_count = (count_starting_with_1 + count_ending_with_1 - 
                   count_starting_and_ending_with_1)
    return total_count