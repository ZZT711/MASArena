'''
Main entry point of the application to find the strongest extension for a class.
'''
from extensions import Strongest_Extension
def main():
    class_name = "Slices"
    extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
    result = Strongest_Extension(class_name, extensions)
    print(result)
if __name__ == "__main__":
    main()