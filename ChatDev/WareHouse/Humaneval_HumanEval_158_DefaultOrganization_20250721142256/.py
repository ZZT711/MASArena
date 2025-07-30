def find_max(words):
    """
    Write a function that accepts a list of strings.
    The list contains different words. Return the word with maximum number
    of unique characters. If multiple strings have maximum number of unique
    characters, return the one which comes first in lexicographical order.
    :param words: List of strings
    :return: String with maximum unique characters
    """
    max_word = ""
    max_unique_count = 0
    for word in words:
        unique_chars = set(word)  # Get unique characters in the word
        unique_count = len(unique_chars)  # Count of unique characters
        # Check if we found a new maximum or if we need to update based on lexicographical order
        if (unique_count > max_unique_count) or (unique_count == max_unique_count and word < max_word):
            max_word = word
            max_unique_count = unique_count
    return max_word