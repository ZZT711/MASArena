def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.
    For example:
    generate_integers(2, 8) => [2, 4, 6, 8]
    generate_integers(8, 2) => [2, 4, 6, 8]
    generate_integers(10, 14) => []
    """
    # Determine the start and end of the range
    start = min(a, b)
    end = max(a, b)
    # Initialize a list to hold the even integers
    even_integers = []
    # Iterate through the range from start to end (inclusive)
    for number in range(start, end + 1):
        # Check if the number is even
        if number % 2 == 0:
            even_integers.append(number)
    return even_integers