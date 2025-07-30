def is_first_and_last_odd(num):
    """Check if the first and last digits of the number are odd."""
    # Convert the number to a string to easily access the first and last digits
    num_str = str(abs(num))  # Use abs to handle negative numbers
    first_digit = int(num_str[0])
    last_digit = int(num_str[-1])
    return first_digit % 2 == 1 and last_digit % 2 == 1
def specialFilter(nums):
    """Write a function that takes an array of numbers as input and returns 
    the number of elements in the array that are greater than 10 and both 
    first and last digits of a number are odd (1, 3, 5, 7, 9).
    For example:
    specialFilter([15, -73, 14, -15]) => 1 
    specialFilter([33, -2, -3, 45, 21, 109]) => 2
    """
    count = 0
    for num in nums:
        if num > 10 and is_first_and_last_odd(num):
            count += 1
    return count