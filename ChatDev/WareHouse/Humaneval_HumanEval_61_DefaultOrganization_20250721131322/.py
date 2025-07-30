def correct_bracketing(brackets: str):
    """
    brackets is a string of "(" and ")".
    return True if every opening bracket has a corresponding closing bracket.
    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
    """
    balance = 0  # Initialize a counter for the balance of brackets
    for char in brackets:
        if char == '(':
            balance += 1  # Increment for an opening bracket
        elif char == ')':
            balance -= 1  # Decrement for a closing bracket
        # If balance goes negative, there are more closing brackets
        if balance < 0:
            return False
    # At the end, balance should be zero for a valid bracket sequence
    return balance == 0