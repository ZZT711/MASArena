def is_prime(num: int) -> bool:
    """
    Check if a number is prime.
    Args:
        num (int): The number to check.
    Returns:
        bool: True if num is prime, False otherwise.
    """
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True
def fibonacci_generator() -> int:
    """
    Generate Fibonacci numbers indefinitely.
    Yields:
        int: The next Fibonacci number.
    """
    a, b = 0, 1
    while True:
        a, b = b, a + b
        yield a
def prime_fib(n: int) -> int:
    """
    Return the n-th Fibonacci number that is also prime.
    Args:
        n (int): The position of the desired prime Fibonacci number.
    Returns:
        int: The n-th Fibonacci number that is prime.
    Examples:
        >>> prime_fib(1)
        2
        >>> prime_fib(2)
        3
        >>> prime_fib(3)
        5
        >>> prime_fib(4)
        13
        >>> prime_fib(5)
        89
    """
    count = 0
    for fib in fibonacci_generator():
        if is_prime(fib):
            count += 1
            if count == n:
                return fib