'''
This module contains utility functions for calculating the strength of extensions.
'''
def calculate_strength(extension):
    """
    Calculate the strength of the extension based on the number of uppercase (CAP)
    and lowercase (SM) letters. The strength is defined as CAP - SM.
    """
    CAP = sum(1 for char in extension if char.isupper())
    SM = sum(1 for char in extension if char.islower())
    return CAP - SM