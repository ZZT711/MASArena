def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome. """
    return string == string[::-1]
def make_palindrome(string: str) -> str:
    """ 
    Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """
    if not string:  # Handle the edge case of an empty string
        return ''
    # Find the longest palindromic suffix
    for i in range(len(string)):
        if is_palindrome(string[i:]):
            # Append the reverse of the prefix to the end of the string
            return string + string[:i][::-1]
    # If no palindromic suffix is found, the whole string is returned with its reverse
    return string + string[::-1]
# Example usage (uncomment to test):
# print(make_palindrome('cat'))  # Output: 'catac'
# print(make_palindrome('cata'))  # Output: 'catac'
# print(make_palindrome(''))  # Output: ''