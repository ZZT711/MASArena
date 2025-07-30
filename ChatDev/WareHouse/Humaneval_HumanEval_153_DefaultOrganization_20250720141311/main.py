'''
Main entry point for the Strongest Extension application.
Handles user input and displays the result.
'''
from utils import Strongest_Extension
def main():
    class_name = input("Enter the class name: ")
    extensions = input("Enter the extensions separated by commas: ").split(',')
    extensions = [ext.strip() for ext in extensions]  # Clean up whitespace
    result = Strongest_Extension(class_name, extensions)
    print(f"The strongest extension is: {result}")
if __name__ == "__main__":
    main()