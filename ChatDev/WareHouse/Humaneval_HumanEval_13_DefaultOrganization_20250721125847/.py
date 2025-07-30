def greatest_common_divisor(a: int, b: int) -> int:
    """
    Return the greatest common divisor of two integers a and b.
    The function uses the Euclidean algorithm to compute the GCD.
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    >>> greatest_common_divisor(-25, 15)
    5
    >>> greatest_common_divisor(0, 5)
    5
    >>> greatest_common_divisor(0, 0)
    0
    """
    if a == 0 and b == 0:
        return 0  # GCD of 0 and 0 is undefined, but we return 0 for this case.
    a, b = abs(a), abs(b)  # Ensure both numbers are positive for GCD calculation.
    while b != 0:
        a, b = b, a % b  # Apply the Euclidean algorithm.
    return a  # The GCD is the last non-zero remainder.