'''
You are given a list of integers.
You need to find the largest prime value and return the sum of its digits.
'''
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def sum_of_digits(n):
    """Calculate the sum of the digits of a number."""
    return sum(int(digit) for digit in str(n))
def skjkasdkd(lst):
    """Find the largest prime in the list and return the sum of its digits."""
    largest_prime = -1  # Start with an invalid prime
    for number in lst:
        if is_prime(number) and number > largest_prime:
            largest_prime = number
    # If no prime was found, return 0 (or handle as needed)
    if largest_prime == -1:
        return 0
    return sum_of_digits(largest_prime)
# Example usage (not part of the function, just for testing):
# print(skjkasdkd([0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3]))  # Output: 10
# print(skjkasdkd([1,0,1,8,2,4597,2,1,3,40,1,2,1,2,4,2,5,1]))  # Output: 25
# print(skjkasdkd([1,3,1,32,5107,34,83278,109,163,23,2323,32,30,1,9,3]))  # Output: 13
# print(skjkasdkd([0,724,32,71,99,32,6,0,5,91,83,0,5,6]))  # Output: 11
# print(skjkasdkd([0,81,12,3,1,21]))  # Output: 3
# print(skjkasdkd([0,8,1,2,1,7]))  # Output: 7