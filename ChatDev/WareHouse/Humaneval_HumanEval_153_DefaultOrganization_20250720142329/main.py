'''
This file contains the implementation of the Strongest_Extension function,
which calculates the strongest extension based on uppercase and lowercase letters.
'''
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
        CAP = sum(1 for char in extension if char.isupper())
        SM = sum(1 for char in extension if char.islower())
        strength = CAP - SM
        if strength > max_strength:
            max_strength = strength
            strongest_extension = extension
    return f"{class_name}.{strongest_extension}"
# Test the function
if __name__ == "__main__":
    class_name = "Slices"
    extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
    result = Strongest_Extension(class_name, extensions)
    print(result)  # Expected output: 'Slices.SErviNGSliCes'