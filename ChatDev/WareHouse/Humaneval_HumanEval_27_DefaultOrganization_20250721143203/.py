'''
This module contains a function to flip the case of characters in a given string.
The function converts all lowercase letters to uppercase and all uppercase letters to lowercase.
Function:
- flip_case(string: str) -> str: Flips the case of each character in the input string.
'''
def flip_case(string: str) -> str:
    """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase.
    >>> flip_case('Hello')
    'hELLO'
    """
    flipped_string = ''
    for char in string:
        if char.islower():
            flipped_string += char.upper()
        elif char.isupper():
            flipped_string += char.lower()
        else:
            flipped_string += char  # Keep non-alphabetic characters unchanged
    return flipped_string