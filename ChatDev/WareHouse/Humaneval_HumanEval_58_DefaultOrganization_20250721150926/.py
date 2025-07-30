def common(l1: list, l2: list):
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])
    [1, 5, 653]
    >>> common([5, 3, 2, 8], [3, 2])
    [2, 3]
    Args:
        l1 (list): The first list of elements.
        l2 (list): The second list of elements.
    Returns:
        list: A sorted list of unique common elements.
    """
    # Create a set from both lists to find unique elements
    set1 = set(l1)
    set2 = set(l2)
    # Find common elements using intersection
    common_elements = set1.intersection(set2)
    # Convert the set to a sorted list
    sorted_common = sorted(common_elements)
    return sorted_common