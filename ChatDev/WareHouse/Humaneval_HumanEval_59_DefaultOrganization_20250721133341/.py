def largest_prime_factor(n: int):
    """
    Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    def is_prime(num):
        """
        Check if a number is prime.
        """
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    largest_factor = 1
    # Check for the number of 2s that divide n
    while n % 2 == 0:
        largest_factor = 2
        n //= 2
    # n must be odd at this point, so we can skip even numbers
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            largest_factor = i
            n //= i
    # This condition is to check if n is a prime number greater than 2
    if n > 2:
        largest_factor = n
    return largest_factor