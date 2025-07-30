'''
Return sorted unique elements in a list.
Parameters:
l (list): A list of elements.
Returns:
list: A sorted list of unique elements.
Examples:
>>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
[0, 2, 3, 5, 9, 123]
>>> unique([])
[]
>>> unique([1, 1, 1])
[1]
>>> unique([1, 2, 3, 4, 5])
[1, 2, 3, 4, 5]
>>> unique([-1, -2, -3, -2, -1])
[-3, -2, -1]
'''
def unique(l: list):
    """Return sorted unique elements in a list."""
    # Create an empty list to store unique elements
    unique_elements = []
    # Iterate through the input list
    for item in l:
        # If the item is not already in the unique_elements list, add it
        if item not in unique_elements:
            unique_elements.append(item)
    # Sort the unique elements
    unique_elements.sort()
    return unique_elements