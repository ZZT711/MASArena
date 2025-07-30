'''
Main entry point for the application.
Handles user input and displays results.
'''
from even_sum_checker import is_equal_to_sum_even
def main():
    try:
        n = int(input("Enter a positive integer: "))
        if n <= 0:
            print("Please enter a positive integer.")
            return
        result = is_equal_to_sum_even(n)
        print(f"is_equal_to_sum_even({n}) == {result}")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
if __name__ == "__main__":
    main()