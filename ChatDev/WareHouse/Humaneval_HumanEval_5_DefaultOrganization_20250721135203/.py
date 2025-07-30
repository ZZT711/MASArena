'''
Insert a number 'delimeter' between every two consecutive elements of input list `numbers`.
Parameters:
    numbers (List[int]): A list of integers.
    delimeter (int): An integer to be inserted between the elements of the list.
Returns:
    List[int]: A new list with the delimeter inserted.
Examples:
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
'''
def intersperse(numbers, delimeter):
    # Initialize an empty list to hold the result
    result = []
    # Iterate through the input list
    for i in range(len(numbers)):
        # Append the current number to the result
        result.append(numbers[i])
        # If it's not the last element, append the delimiter
        if i < len(numbers) - 1:
            result.append(delimeter)
    return result