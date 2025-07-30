'''
This module contains the Strongest_Extension function which calculates
the strongest extension based on the given criteria.
'''
def Strongest_Extension(class_name, extensions):
    strongest_extension = None
    max_strength = float('-inf')
    for extension in extensions:
        CAP = sum(1 for char in extension if char.isupper())
        SM = sum(1 for char in extension if char.islower())
        strength = CAP - SM
        if strength > max_strength:
            max_strength = strength
            strongest_extension = extension
    return f"{class_name}.{strongest_extension}"