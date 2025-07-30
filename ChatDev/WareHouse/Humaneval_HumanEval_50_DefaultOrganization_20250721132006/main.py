def encode_shift(s: str):
    """
    Returns encoded string by shifting every character by 5 in the alphabet.
    Parameters:
    s (str): The input string to be encoded.
    Returns:
    str: The encoded string with each character shifted by 5.
    """
    return "".join([chr(((ord(ch) - ord("a") + 5) % 26) + ord("a")) for ch in s if ch.isalpha() and ch.islower()])
def decode_shift(s: str):
    """
    Takes as input string encoded with encode_shift function. Returns decoded string.
    Parameters:
    s (str): The input string to be decoded.
    Returns:
    str: The decoded string with each character shifted back by 5.
    """
    return "".join([chr(((ord(ch) - ord("a") - 5) % 26) + ord("a")) for ch in s if ch.isalpha() and ch.islower()])
# Example usage
if __name__ == "__main__":
    original = "hello"
    encoded = encode_shift(original)
    decoded = decode_shift(encoded)
    print(f"Original: {original}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")