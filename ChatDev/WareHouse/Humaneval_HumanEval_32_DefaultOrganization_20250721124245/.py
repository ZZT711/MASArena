'''
This module provides functions to evaluate a polynomial and find its zero point.
'''
def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    Returns the value of the polynomial at x.
    :param xs: List of coefficients of the polynomial.
    :param x: The point at which to evaluate the polynomial.
    :return: The evaluated polynomial value.
    """
    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))
def find_zero(xs: list):
    """
    Finds x such that poly(x) = 0 for the polynomial defined by coefficients xs.
    Returns only one zero point, assuming the input list has an even number of coefficients
    and the largest non-zero coefficient guarantees a solution.
    :param xs: List of coefficients of the polynomial.
    :return: A zero point of the polynomial.
    >>> round(find_zero([1, 2]), 2)  # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2)  # (x - 1) * (x - 2) * (x - 3)
    1.0
    """
    if len(xs) % 2 != 0:
        raise ValueError("The number of coefficients must be even.")
    # Using the bisection method to find a root
    low, high = -1000, 1000  # Initial bounds for the search
    while high - low > 1e-7:  # Precision threshold
        mid = (low + high) / 2
        if poly(xs, mid) == 0:
            return mid
        elif poly(xs, mid) * poly(xs, low) < 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2  # Return the midpoint as the root