'''
Main entry point for the Strongest Extension application.
'''
from utils import calculate_strength
def get_strongest_extension(class_name, extensions):
    """
    Given a class name and a list of extensions, find the strongest extension
    based on the defined strength criteria.
    """
    strongest_extension = None
    max_strength = float('-inf')
    for extension in extensions:
        strength = calculate_strength(extension)
        if strength > max_strength:
            max_strength = strength
            strongest_extension = extension
    return f"{class_name}.{strongest_extension}"
if __name__ == "__main__":
    # Example usage
    class_name = "Slices"
    extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
    result = get_strongest_extension(class_name, extensions)
    print(result)  # Output: Slices.SErviNGSliCes