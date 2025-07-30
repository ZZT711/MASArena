def solution(lst):
    """
    Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions.
    Parameters:
    lst (list): A non-empty list of integers.
    Returns:
    int: The sum of all odd elements in even positions.
    Examples:
    solution([5, 8, 7, 1]) ==> 12
    solution([3, 3, 3, 3, 3]) ==> 9
    solution([30, 13, 24, 321]) ==> 0
    """
    total_sum = 0  # Initialize the sum of odd elements in even positions
    # Iterate through the list using index
    for index in range(len(lst)):
        # Check if the index is even and the element is odd
        if index % 2 == 0 and lst[index] % 2 != 0:
            total_sum += lst[index]  # Add to the sum if conditions are met
    return total_sum  # Return the final sum