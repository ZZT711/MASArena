def is_prime(n):
    """Check if a number is prime.
    A prime number is greater than 1 and has no divisors other than 1 and itself.
    Args:
        n (int): The number to check for primality.
    Returns:
        bool: True if n is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def intersection(interval1, interval2):
    """Determine if the length of the intersection of two intervals is a prime number.
    The function calculates the intersection of two closed intervals and checks
    if the length of the intersection is a prime number. If the intervals do not
    intersect, it returns "NO".
    Args:
        interval1 (tuple): The first interval as a tuple (start, end).
        interval2 (tuple): The second interval as a tuple (start, end).
    Returns:
        str: "YES" if the length of the intersection is a prime number,
             "NO" otherwise.
    """
    start1, end1 = interval1
    start2, end2 = interval2
    # Calculate the intersection
    start_intersection = max(start1, start2)
    end_intersection = min(end1, end2)
    if start_intersection > end_intersection:
        return "NO"  # No intersection
    # Length of the intersection
    length_intersection = end_intersection - start_intersection
    # Check if the length is a prime number
    if is_prime(length_intersection):
        return "YES"
    else:
        return "NO"
# Example usage (uncomment to test):
# print(intersection((1, 2), (2, 3)))  # Expected: "NO"
# print(intersection((-1, 1), (0, 4)))  # Expected: "NO"
# print(intersection((-3, -1), (-5, 5)))  # Expected: "YES"