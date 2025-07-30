'''
Main entry point for the application to evaluate if a number can be expressed as the sum of four positive even numbers.
'''
from utils import is_equal_to_sum_even
def main():
    # Get user input
    try:
        n = int(input("Enter a positive integer: "))
        if n <= 0:
            print("Please enter a positive integer.")
            return
        # Evaluate and print the result
        result = is_equal_to_sum_even(n)
        print(f"is_equal_to_sum_even({n}) == {result}")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
if __name__ == "__main__":
    main()