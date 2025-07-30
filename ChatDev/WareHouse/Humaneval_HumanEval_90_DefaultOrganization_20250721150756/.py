'''
This function returns the second smallest element in a list of integers.
If there is no second smallest element, it returns None.
'''
def next_smallest(lst):
    # Remove duplicates by converting the list to a set
    unique_elements = set(lst)
    # If there are less than 2 unique elements, return None
    if len(unique_elements) < 2:
        return None
    # Convert the set back to a sorted list
    sorted_unique_elements = sorted(unique_elements)
    # Return the second smallest element
    return sorted_unique_elements[1]
# Example test cases
print(next_smallest([1, 2, 3, 4, 5]))  # Output: 2
print(next_smallest([5, 1, 4, 3, 2]))  # Output: 2
print(next_smallest([]))                # Output: None
print(next_smallest([1, 1]))            # Output: None