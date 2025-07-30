def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    This function checks if two input strings contain the same characters,
    regardless of their order and frequency. It returns True if they do,
    and False otherwise.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """
    def char_count(s: str) -> dict:
        """
        Count the occurrences of each character in the string.
        Args:
            s (str): The input string.
        Returns:
            dict: A dictionary with characters as keys and their counts as values.
        """
        count = {}
        for char in s:
            if char in count:
                count[char] += 1
            else:
                count[char] = 1
        return count
    # Get character counts for both strings
    count_s0 = char_count(s0)
    count_s1 = char_count(s1)
    # Compare the two dictionaries
    return count_s0 == count_s1