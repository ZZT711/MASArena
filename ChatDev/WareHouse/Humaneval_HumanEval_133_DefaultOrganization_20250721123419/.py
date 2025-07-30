def ceiling(num):
    """Return the smallest integer greater than or equal to num."""
    if num == int(num):
        return int(num)
    else:
        return int(num) + 1 if num > 0 else int(num)
def sum_squares(lst):
    """You are given a list of numbers.
    You need to return the sum of squared numbers in the given list,
    round each element in the list to the upper int (Ceiling) first.
    Examples:
    For lst = [1,2,3] the output should be 14
    For lst = [1,4,9] the output should be 98
    For lst = [1,3,5,7] the output should be 84
    For lst = [1.4,4.2,0] the output should be 29
    For lst = [-2.4,1,1] the output should be 6
    """
    total = 0
    for number in lst:
        rounded_number = ceiling(number)
        total += rounded_number ** 2
    return total