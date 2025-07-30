from typing import List
def factorize(n: int) -> List[int]:
    """
    Return list of prime factors of given integer in the order from smallest to largest.
    Each of the factors should be listed number of times corresponding to how many times it appears in factorization.
    Input number should be equal to the product of all factors.
    >>> factorize(8)
    [2, 2, 2]
    >>> factorize(25)
    [5, 5]
    >>> factorize(70)
    [2, 5, 7]
    """
    if n <= 1:
        return []  # Edge case: no prime factors for numbers <= 1
    return _get_prime_factors(n)
def _get_prime_factors(n: int) -> List[int]:
    """
    Helper function to compute the prime factors of a given integer n.
    >>> _get_prime_factors(8)
    [2, 2, 2]
    >>> _get_prime_factors(25)
    [5, 5]
    >>> _get_prime_factors(70)
    [2, 5, 7]
    """
    factors = []
    # Check for number of 2s that divide n
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    # n must be odd at this point, so we can skip even numbers
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    # This condition is to check if n is a prime number greater than 2
    if n > 2:
        factors.append(n)
    return factors