def rounded_avg(n, m):
    """
    You are given two positive integers n and m, and your task is to compute the
    average of the integers from n through m (including n and m). 
    Round the answer to the nearest integer and convert that to binary.
    If n is greater than m, return -1.
    Parameters:
    n (int): The starting integer.
    m (int): The ending integer.
    Returns:
    str or int: The binary representation of the rounded average if n <= m, 
                otherwise -1.
    """
    if n > m:
        return -1
    total_sum = sum(range(n, m + 1))
    count = m - n + 1
    average = total_sum / count
    rounded_average = round(average)
    return bin(rounded_average)
# Example usage
print(rounded_avg(1, 5))  # Output: "0b11"
print(rounded_avg(7, 5))  # Output: -1
print(rounded_avg(10, 20))  # Output: "0b1111"
print(rounded_avg(20, 33))  # Output: "0b11010"