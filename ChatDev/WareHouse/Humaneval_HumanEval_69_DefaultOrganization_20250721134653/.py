def search(lst):
    '''
    You are given a non-empty list of positive integers. Return the greatest integer that is greater than 
    zero, and has a frequency greater than or equal to the value of the integer itself. 
    The frequency of an integer is the number of times it appears in the list.
    If no such a value exists, return -1.
    Parameters:
    lst (list): A non-empty list of positive integers.
    Returns:
    int: The greatest integer with the required frequency condition or -1 if none exists.
    '''
    def count_frequencies(lst):
        '''
        Count the frequency of each integer in the list.
        Parameters:
        lst (list): A list of positive integers.
        Returns:
        dict: A dictionary with integers as keys and their frequencies as values.
        '''
        frequency = {}
        for number in lst:
            if number in frequency:
                frequency[number] += 1
            else:
                frequency[number] = 1
        return frequency
    frequencies = count_frequencies(lst)
    max_value = -1
    for number, count in frequencies.items():
        if number > 0 and count >= number:
            max_value = max(max_value, number)
    return max_value
# Example test cases
print(search([4, 1, 2, 2, 3, 1]))  # Output: 2
print(search([1, 2, 2, 3, 3, 3, 4, 4, 4]))  # Output: 3
print(search([5, 5, 4, 4, 4]))  # Output: -1