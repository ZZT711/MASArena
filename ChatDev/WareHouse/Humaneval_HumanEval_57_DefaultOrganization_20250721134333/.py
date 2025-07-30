def monotonic(l: list):
    """
    Return True if list elements are monotonically increasing or decreasing.
    >>> monotonic([1, 2, 4, 20])
    True
    >>> monotonic([1, 20, 4, 10])
    False
    >>> monotonic([4, 1, 0, -10])
    True
    >>> monotonic([])
    True
    >>> monotonic([1])
    True
    >>> monotonic([1, 1, 1])
    True
    """
    if len(l) <= 1:
        return True  # An empty list or a single element list is considered monotonic
    increasing = True
    decreasing = True
    for i in range(1, len(l)):
        if l[i] > l[i - 1]:
            decreasing = False
        elif l[i] < l[i - 1]:
            increasing = False
    return increasing or decreasing