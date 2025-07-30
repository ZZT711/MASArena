def tri(n):
    """
    Generate the first n + 1 numbers of the Tribonacci sequence.
    The Tribonacci sequence is defined by the recurrence:
    tri(1) = 3
    tri(n) = 1 + n / 2, if n is even.
    tri(n) = tri(n - 1) + tri(n - 2) + tri(n + 1), if n is odd.
    Parameters:
    n (int): A non-negative integer representing the length of the sequence to generate.
    Returns:
    list: A list containing the first n + 1 numbers of the Tribonacci sequence.
    """
    if n < 0:
        return []
    sequence = []
    for i in range(n + 1):
        sequence.append(tribonacci_recursive(i))
    return sequence
def tribonacci_recursive(n):
    """
    Helper function to compute the Tribonacci value for a given n.
    Parameters:
    n (int): The index in the Tribonacci sequence.
    Returns:
    int: The Tribonacci number at index n.
    """
    if n == 1:
        return 3
    elif n % 2 == 0:  # n is even
        return 1 + n // 2
    else:  # n is odd
        return tribonacci_recursive(n - 1) + tribonacci_recursive(n - 2) + tribonacci_recursive(n + 1)
# Example usage:
# print(tri(3))  # Output: [1, 3, 2, 8]