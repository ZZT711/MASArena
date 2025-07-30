'''
Given a string representing a space separated lowercase letters, return a dictionary
of the letter with the most repetition and containing the corresponding count.
If several letters have the same occurrence, return all of them.
'''
def histogram(test):
    """Calculate the histogram of letters in a space-separated string.
    Args:
        test (str): A string of space-separated lowercase letters.
    Returns:
        dict: A dictionary with the letter(s) that have the highest count and their corresponding count.
    """
    if not test:  # Handle the empty string case
        return {}
    # Split the input string into a list of letters
    letters = test.split()
    # Create a dictionary to count occurrences of each letter
    count_dict = {}
    for letter in letters:
        if letter in count_dict:
            count_dict[letter] += 1
        else:
            count_dict[letter] = 1
    # Find the maximum occurrence count
    max_count = max(count_dict.values())
    # Create a result dictionary for letters with the maximum count
    result = {letter: count for letter, count in count_dict.items() if count == max_count}
    return result