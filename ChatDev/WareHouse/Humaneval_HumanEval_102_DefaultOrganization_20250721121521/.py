def choose_num(x, y):
    """
    This function takes two positive numbers x and y and returns the
    biggest even integer number that is in the range [x, y] inclusive. If 
    there's no such number, then the function should return -1.
    For example:
    choose_num(12, 15) = 14
    choose_num(13, 12) = -1
    """
    # Check if x is greater than y
    if x > y:
        return -1
    # Iterate from y down to x to find the largest even number
    for num in range(y, x - 1, -1):
        if num % 2 == 0:
            return num
    # If no even number is found, return -1
    return -1