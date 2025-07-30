'''
Main entry point for the Strongest Extension application.
'''
from extensions import Strongest_Extension
def main():
    class_name = "Slices"
    extensions = ['SErviNGSliCes', 'Cheese', 'StuFfed']
    result = Strongest_Extension(class_name, extensions)
    print(result)  # Expected output: Slices.SErviNGSliCes
if __name__ == "__main__":
    main()