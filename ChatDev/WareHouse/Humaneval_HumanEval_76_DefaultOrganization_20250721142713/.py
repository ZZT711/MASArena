def is_simple_power(x, n):
    """
    Determine if x is a simple power of n.
    A number x is a simple power of n if there exists an integer k such that n**k = x.
    For example:
    is_simple_power(1, 4) => True
    is_simple_power(2, 2) => True
    is_simple_power(8, 2) => True
    is_simple_power(3, 2) => False
    is_simple_power(3, 1) => False
    is_simple_power(5, 3) => False
    Parameters:
    x (int): The number to check.
    n (int): The base number.
    Returns:
    bool: True if x is a simple power of n, False otherwise.
    """
    # Edge case: if n is 0 or 1
    if n <= 1:
        return x == n  # 0**k is 0 for k > 0, and 1**k is always 1
    # Check powers of n until n**k exceeds x
    power = 1
    while power < x:
        power *= n
    return power == x