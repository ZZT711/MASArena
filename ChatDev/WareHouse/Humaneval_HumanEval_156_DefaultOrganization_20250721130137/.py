def int_to_mini_roman(number):
    """
    Given a positive integer, obtain its roman numeral equivalent as a string,
    and return it in lowercase.
    Restrictions: 1 <= num <= 1000
    Examples:
    >>> int_to_mini_roman(19) == 'xix'
    >>> int_to_mini_roman(152) == 'clii'
    >>> int_to_mini_roman(426) == 'cdxxvi'
    """
    if not (1 <= number <= 1000):
        raise ValueError("Number must be between 1 and 1000")
    # Define the mapping of integers to Roman numerals
    roman_numerals = [
        (1000, 'm'), (900, 'cm'), (500, 'd'), (400, 'cd'),
        (100, 'c'), (90, 'xc'), (50, 'l'), (40, 'xl'),
        (10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'),
        (1, 'i')
    ]
    result = []
    for value, numeral in roman_numerals:
        while number >= value:
            result.append(numeral)
            number -= value
    return ''.join(result)
# Example usage (uncomment to test):
# print(int_to_mini_roman(19))   # Output: 'xix'
# print(int_to_mini_roman(152))  # Output: 'clii'
# print(int_to_mini_roman(426))  # Output: 'cdxxvi'