'''
This module contains a function to determine if the length of a given string
is a prime number. The main function is `prime_length`, which utilizes a helper
function `is_prime` to perform the prime number check.
'''
def is_prime(n):
    """Check if a number is prime.
    Args:
        n (int): The number to check.
    Returns:
        bool: True if n is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def prime_length(string):
    """Determine if the length of the string is a prime number.
    Args:
        string (str): The input string.
    Returns:
        bool: True if the length of the string is prime, False otherwise.
    Examples:
        prime_length('Hello') == True
        prime_length('abcdcba') == True
        prime_length('kittens') == True
        prime_length('orange') == False
    """
    length = len(string)
    return is_prime(length)