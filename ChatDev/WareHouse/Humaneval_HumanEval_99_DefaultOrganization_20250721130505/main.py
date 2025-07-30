def closest_integer(value):
    '''
    Create a function that takes a value (string) representing a number
    and returns the closest integer to it. If the number is equidistant
    from two integers, round it away from zero.
    Examples
    >>> closest_integer("10")
    10
    >>> closest_integer("15.3")
    15
    Note:
    Rounding away from zero means that if the given number is equidistant
    from two integers, the one you should return is the one that is the
    farthest from zero. For example closest_integer("14.5") should
    return 15 and closest_integer("-14.5") should return -15.
    '''
    # Convert the input string to a float
    num = float(value)
    # Calculate the lower and upper integers
    lower_int = int(num // 1)
    upper_int = lower_int + 1
    # Determine the distance to the lower and upper integers
    distance_to_lower = abs(num - lower_int)
    distance_to_upper = abs(num - upper_int)
    # Apply rounding rules
    if distance_to_lower < distance_to_upper:
        return lower_int
    elif distance_to_upper < distance_to_lower:
        return upper_int
    else:
        # If equidistant, round away from zero
        if num > 0:
            return upper_int
        else:
            return lower_int
# Example usage
if __name__ == "__main__":
    print(closest_integer("10"))      # Output: 10
    print(closest_integer("15.3"))    # Output: 15
    print(closest_integer("14.5"))    # Output: 15
    print(closest_integer("-14.5"))   # Output: -15
    print(closest_integer("0.5"))     # Output: 1
    print(closest_integer("-0.5"))    # Output: -1