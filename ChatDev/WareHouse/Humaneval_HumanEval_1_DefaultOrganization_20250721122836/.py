def separate_paren_groups(paren_string: str):
    """
    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those groups into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # Remove spaces from the input string
    paren_string = paren_string.replace(" ", "")
    result = []
    current_group = ""
    balance = 0
    for char in paren_string:
        if char == '(':
            balance += 1
            current_group += char
        elif char == ')':
            balance -= 1
            current_group += char
        # When balance is zero, we have a complete group
        if balance == 0 and current_group:
            result.append(current_group)
            current_group = ""  # Reset for the next group
    return result