'''
This module provides two functions: encode_cyclic and decode_cyclic.
The encode_cyclic function takes a string and returns an encoded string by cycling groups of three characters.
The decode_cyclic function takes an encoded string and returns the original string by reversing the encoding process.
'''
def encode_cyclic(s: str) -> str:
    """
    Returns encoded string by cycling groups of three characters.
    Args:
        s (str): The input string to be encoded.
    Returns:
        str: The encoded string.
    """
    # Split string into groups of length 3
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    # Cycle elements in each group unless group has fewer than 3 elements
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
def decode_cyclic(s: str) -> str:
    """
    Takes as input a string encoded with encode_cyclic function. Returns decoded string.
    Args:
        s (str): The encoded string to be decoded.
    Returns:
        str: The decoded original string.
    """
    # Split string into groups of length 3
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    # Reverse cycle elements in each group unless group has fewer than 3 elements
    groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
    return "".join(groups)
# Example usage (uncomment to test):
# encoded = encode_cyclic("abcdefg")
# print(encoded)  # Output: "bcadefg"
# decoded = decode_cyclic(encoded)
# print(decoded)  # Output: "abcdefg"