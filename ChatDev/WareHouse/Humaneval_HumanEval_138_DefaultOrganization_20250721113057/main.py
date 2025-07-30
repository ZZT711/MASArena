'''
Main application file to evaluate whether a number can be expressed as the sum of four positive even numbers.
'''
from even_sum_checker import is_equal_to_sum_even
def main():
    # Get user input
    try:
        n = int(input("Enter a positive integer: "))
        if n <= 0:
            print("Please enter a positive integer.")
            return
        # Check if n can be expressed as the sum of four positive even numbers
        result = is_equal_to_sum_even(n)
        print(f"is_equal_to_sum_even({n}) == {result}")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
if __name__ == "__main__":
    main()