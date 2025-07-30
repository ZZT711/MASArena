def concatenate(strings):
    """
    Concatenate list of strings into a single string.
    Args:
        strings (List[str]): A list of strings to concatenate.
    Returns:
        str: A single concatenated string.
    Examples:
    >>> concatenate([])
    ''
    >>> concatenate(['a', 'b', 'c'])
    'abc'
    """
    result = ''
    for string in strings:
        result += string
    return result