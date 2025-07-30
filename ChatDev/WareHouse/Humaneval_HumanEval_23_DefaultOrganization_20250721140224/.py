def strlen(string: str) -> int:
    """
    Return length of given string.
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """
    # Initialize a counter for the length
    length = 0
    # Iterate through each character in the string
    for char in string:
        length += 1  # Increment the length for each character
    return length