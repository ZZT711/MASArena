def filter_integers(values):
    """
    Filter given list of any python values only for integers.
    Args:
        values (List[Any]): A list containing any Python values.
    Returns:
        List[int]: A list containing only the integers from the input list.
    Examples:
        >>> filter_integers(['a', 3.14, 5])
        [5]
        >>> filter_integers([1, 2, 3, 'abc', {}, []])
        [1, 2, 3]
    """
    result = []
    for value in values:
        if isinstance(value, int):
            result.append(value)
    return result