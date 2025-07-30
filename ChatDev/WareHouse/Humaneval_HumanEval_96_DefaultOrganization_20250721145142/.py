'''
This module contains a function to count prime numbers less than a given non-negative integer.
The main function `count_up_to(n)` returns a list of prime numbers that are less than `n`.
'''
def is_prime(num):
    """Check if a number is prime.
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
def count_up_to(n):
    """Return a list of prime numbers less than n.
    Args:
        n (int): A non-negative integer.
    Returns:
        list: A list of prime numbers less than n.
    """
    primes = []
    for i in range(2, n):
        if is_prime(i):
            primes.append(i)
    return primes