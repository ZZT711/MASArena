'''
This module contains the Strongest_Extension function that determines the strongest
extension for a given class name based on the strength calculated from uppercase and
lowercase letters in the extension names.
'''
from utils import calculate_strength
def Strongest_Extension(class_name, extensions):
    """
    You will be given the name of a class (a string) and a list of extensions.
    The extensions are to be used to load additional classes to the class. The
    strength of the extension is as follows: Let CAP be the number of the uppercase
    letters in the extension's name, and let SM be the number of lowercase letters 
    in the extension's name, the strength is given by the fraction CAP - SM. 
    You should find the strongest extension and return a string in this 
    format: ClassName.StrongestExtensionName.
    If there are two or more extensions with the same strength, you should
    choose the one that comes first in the list.
    """
    strongest_extension = None
    max_strength = float('-inf')
    for extension in extensions:
        strength = calculate_strength(extension)
        if strength > max_strength:
            max_strength = strength
            strongest_extension = extension
    return f"{class_name}.{strongest_extension}"