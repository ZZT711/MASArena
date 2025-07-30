'''
This module contains a function to encrypt a given string by rotating
the letters of the alphabet down by four places. The function handles
both lowercase and uppercase letters, while leaving non-alphabetic
characters unchanged.
Function:
- encrypt(s): Takes a string and returns the encrypted string.
'''
def encrypt(s):
    """Encrypts the input string by rotating each letter down by four places.
    Args:
        s (str): The input string to be encrypted.
    Returns:
        str: The encrypted string with letters rotated down by four places.
    """
    encrypted_string = ""
    for char in s:
        if 'a' <= char <= 'z':  # Check for lowercase letters
            # Rotate within the lowercase letters
            new_char = chr(((ord(char) - ord('a') + 4) % 26) + ord('a'))
            encrypted_string += new_char
        elif 'A' <= char <= 'Z':  # Check for uppercase letters
            # Rotate within the uppercase letters
            new_char = chr(((ord(char) - ord('A') + 4) % 26) + ord('A'))
            encrypted_string += new_char
        else:
            # Non-alphabetic characters remain unchanged
            encrypted_string += char
    return encrypted_string