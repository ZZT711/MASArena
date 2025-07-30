def is_happy(s):
    """
    Check if the string s is happy.
    A string is happy if its length is at least 3 and every 3 consecutive letters are distinct.
    Args:
    s (str): The input string to check.
    Returns:
    bool: True if the string is happy, False otherwise.
    """
    # Check if the length of the string is at least 3
    if len(s) < 3:
        return False
    # Iterate through the string and check every 3 consecutive characters
    for i in range(len(s) - 2):
        # Get the current substring of 3 characters
        substring = s[i:i+3]
        # Check if all characters in the substring are distinct
        if len(set(substring)) != 3:
            return False
    return True