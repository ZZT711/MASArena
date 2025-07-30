def is_prime(n):
    """Check if a number is prime.
    Args:
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
def x_or_y(n, x, y):
    """Return x if n is a prime number, otherwise return y.
    Args:
        n (int): The number to check.
        x (any): The value to return if n is prime.
        y (any): The value to return if n is not prime.
    Returns:
        any: x if n is prime, y otherwise.
    Examples:
        x_or_y(7, 34, 12) == 34
        x_or_y(15, 8, 5) == 5
    """
    if is_prime(n):
        return x
    else:
        return y