'''
Utility module for calculating the strength of extensions.
'''
def calculate_strength(extension):
    """
    Calculate the strength of an extension based on the number of uppercase
    and lowercase letters.
    """
    CAP = sum(1 for char in extension if char.isupper())
    SM = sum(1 for char in extension if char.islower())
    return CAP - SM