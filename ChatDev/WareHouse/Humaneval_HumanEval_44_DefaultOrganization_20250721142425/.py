'''
Change numerical base of input number x to base.
Return string representation after the conversion.
Base numbers are less than 10.
>>> change_base(8, 3)
'22'
>>> change_base(8, 2)
'1000'
>>> change_base(7, 2)
'111'
'''
def change_base(x: int, base: int) -> str:
    """Convert a decimal number x to a given base and return its string representation."""
    if base < 2 or base >= 10:
        raise ValueError("Base must be between 2 and 9 inclusive.")
    if x == 0:
        return '0'
    result = ''
    negative = x < 0
    x = abs(x)
    while x > 0:
        remainder = x % base
        result = str(remainder) + result
        x //= base
    if negative:
        result = '-' + result
    return result