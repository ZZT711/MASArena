'''
Main file to execute the application and handle user input.
'''
from utils import is_equal_to_sum_even
def main():
    # Get user input
    n = int(input("Enter a positive integer: "))
    # Evaluate if n can be expressed as the sum of 4 positive even numbers
    result = is_equal_to_sum_even(n)
    # Output the result
    if result:
        print(f"{n} can be expressed as the sum of 4 positive even numbers.")
    else:
        print(f"{n} cannot be expressed as the sum of 4 positive even numbers.")
if __name__ == "__main__":
    main()