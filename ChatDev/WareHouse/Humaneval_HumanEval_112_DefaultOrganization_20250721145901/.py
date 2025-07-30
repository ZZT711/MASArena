def reverse_delete(s, c):
    """
    Task
    We are given two strings s and c, you have to delete all the characters in s that are equal to any character in c
    then check if the result string is palindrome.
    A string is called palindrome if it reads the same backward as forward.
    You should return a tuple containing the result string and True/False for the check.
    """
    # Remove characters from s that are in c
    result = ''.join([char for char in s if char not in c])
    # Check if the result is a palindrome
    is_palindrome_result = is_palindrome(result)
    return (result, is_palindrome_result)
def is_palindrome(s):
    """
    Check if the given string s is a palindrome.
    A string is a palindrome if it reads the same backward as forward.
    """
    return s == s[::-1]
# Example usage (uncomment to test)
# print(reverse_delete("abcde", "ae"))  # Expected: ('bcd', False)
# print(reverse_delete("abcdef", "b"))   # Expected: ('acdef', False)
# print(reverse_delete("abcdedcba", "ab"))  # Expected: ('cdedc', True)