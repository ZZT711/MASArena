def below_threshold(l: list, t: int):
    """
    Return True if all numbers in the list l are below threshold t.
    >>> below_threshold([1, 2, 4, 10], 100)
    True
    >>> below_threshold([1, 20, 4, 10], 5)
    False
    >>> below_threshold([], 5)
    True
    >>> below_threshold([5], 5)
    False
    >>> below_threshold([-1, -2, -3], 0)
    True
    >>> below_threshold([1.5, 2.5, 3.5], 4)
    True
    >>> below_threshold([1, 2, 3, 4], 3)
    False
    """
    for number in l:
        if number >= t:
            return False
    return True