def modp(n: int, p: int):
    """
    Return 2^n modulo p (be aware of numerics).
    >>> modp(3, 5)
    3
    >>> modp(1101, 101)
    2
    >>> modp(0, 101)
    1
    >>> modp(3, 11)
    8
    >>> modp(100, 101)
    1
    """
    if p <= 1:
        raise ValueError("Modulo p must be greater than 1.")
    result = 1
    base = 2
    # Use exponentiation by squaring to compute 2^n % p
    while n > 0:
        if n % 2 == 1:  # If n is odd
            result = (result * base) % p
        base = (base * base) % p  # Square the base
        n //= 2  # Divide n by 2
    return result