def unique_digits(x):
    """
    Given a list of positive integers x, return a sorted list of all 
    elements that haven't any even digit.
    Note: Returned list should be sorted in increasing order.
    For example:
    >>> unique_digits([15, 33, 1422, 1])
    [1, 15, 33]
    >>> unique_digits([152, 323, 1422, 10])
    []
    """
    def has_even_digit(n):
        """Check if the number n contains any even digit."""
        for digit in str(n):
            if int(digit) % 2 == 0:
                return True
        return False
    # Filter the list to include only numbers without even digits
    filtered_numbers = [num for num in x if not has_even_digit(num)]
    # Return the sorted list of filtered numbers
    return sorted(filtered_numbers)
# Example usage (uncomment to test):
# print(unique_digits([15, 33, 1422, 1]))  # Output: [1, 15, 33]
# print(unique_digits([152, 323, 1422, 10]))  # Output: []