def Strongest_Extension(class_name, extensions):
    """
    Given a class name and a list of extensions, this function calculates the strength of each extension
    based on the number of uppercase and lowercase letters. It returns the strongest extension in the format
    ClassName.StrongestExtensionName. If there are ties, the first one in the list is chosen.
    Parameters:
    class_name (str): The name of the class.
    extensions (list): A list of extension names.
    Returns:
    str: The strongest extension in the format ClassName.StrongestExtensionName.
    """
    def calculate_strength(extension):
        """
        Calculate the strength of an extension based on the number of uppercase and lowercase letters.
        Parameters:
        extension (str): The name of the extension.
        Returns:
        int: The strength of the extension calculated as CAP - SM.
        """
        CAP = sum(1 for char in extension if char.isupper())
        SM = sum(1 for char in extension if char.islower())
        return CAP - SM
    strongest_extension = None
    strongest_strength = float('-inf')
    for extension in extensions:
        strength = calculate_strength(extension)
        if strength > strongest_strength:
            strongest_strength = strength
            strongest_extension = extension
    return f"{class_name}.{strongest_extension}"
# Example usage:
# print(Strongest_Extension('Slices', ['SErviNGSliCes', 'Cheese', 'StuFfed']))  # Output: 'Slices.SErviNGSliCes'
# print(Strongest_Extension('my_class', ['AA', 'Be', 'CC']))  # Output: 'my_class.AA'