def sum_product(numbers):
    """
    For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
    total_sum = 0
    total_product = 1
    if not numbers:  # Handle the case for an empty list
        return (total_sum, total_product)
    for number in numbers:
        total_sum += number
        total_product *= number
    return (total_sum, total_product)