'''
Module to handle interval operations including intersection and prime checking.
'''
def intersection(interval1, interval2):
    """
    Determine if the length of the intersection of two intervals is a prime number.
    Parameters:
    interval1 (tuple): The first interval as a tuple of two integers (start, end).
    interval2 (tuple): The second interval as a tuple of two integers (start, end).
    Returns:
    str: "YES" if the length of the intersection is a prime number, "NO" otherwise.
    """
    start1, end1 = interval1
    start2, end2 = interval2
    # Calculate the intersection
    start_intersection = max(start1, start2)
    end_intersection = min(end1, end2)
    if start_intersection > end_intersection:
        return "NO"  # No intersection
    length = end_intersection - start_intersection
    if is_prime(length):
        return "YES"
    else:
        return "NO"
def is_prime(n):
    """
    Check if a number is prime.
    Parameters:
    n (int): The number to check.
    Returns:
    bool: True if n is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True