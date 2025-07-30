'''
Main entry point for the application to evaluate if a number can be expressed as the sum of 4 positive even numbers.
'''
from even_sum_checker import is_equal_to_sum_even
def main():
    test_numbers = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    for number in test_numbers:
        result = is_equal_to_sum_even(number)
        print(f'is_equal_to_sum_even({number}) == {result}')
if __name__ == "__main__":
    main()