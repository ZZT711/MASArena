'''
Main entry point for the application.
'''
from utils import is_equal_to_sum_even
def main():
    # Test cases
    test_numbers = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    for number in test_numbers:
        result = is_equal_to_sum_even(number)
        print(f'is_equal_to_sum_even({number}) == {result}')
if __name__ == "__main__":
    main()