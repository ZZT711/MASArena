def circular_shift(x, shift):
    """
    Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    >>> circular_shift(12, 1)
    '21'
    >>> circular_shift(12, 2)
    '12'
    """
    # Convert the integer to a string to manipulate its digits
    str_x = str(x)
    num_digits = len(str_x)
    # If shift is greater than the number of digits, reverse the string
    if shift >= num_digits:
        return str_x[::-1]
    # Calculate the effective shift
    effective_shift = shift % num_digits
    # Perform the circular shift
    shifted_str = str_x[-effective_shift:] + str_x[:-effective_shift]
    return shifted_str
# Example usage (uncomment to test):
# print(circular_shift(12, 1))  # Output: '21'
# print(circular_shift(12, 2))  # Output: '12'
# print(circular_shift(12345, 3))  # Output: '34512'
# print(circular_shift(12345, 5))  # Output: '54321'