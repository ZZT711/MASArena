'''
This module contains a function to multiply the unit digits of two integers.
The function takes two integers as input and returns the product of their unit digits.
'''
def multiply(a, b):
    """Complete the function that takes two integers and returns 
    the product of their unit digits.
    Args:
        a (int): The first integer.
        b (int): The second integer.
    Returns:
        int: The product of the unit digits of a and b.
    Examples:
        multiply(148, 412) should return 16.
        multiply(19, 28) should return 72.
        multiply(2020, 1851) should return 0.
        multiply(14, -15) should return 20.
    """
    # Get the unit digit of the first number
    unit_digit_a = abs(a) % 10
    # Get the unit digit of the second number
    unit_digit_b = abs(b) % 10
    # Calculate the product of the unit digits
    product = unit_digit_a * unit_digit_b
    return product