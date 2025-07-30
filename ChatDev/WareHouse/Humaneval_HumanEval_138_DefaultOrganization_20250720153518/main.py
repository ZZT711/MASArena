'''
Main entry point of the application to test the functionality of the is_equal_to_sum_even function.
'''
from utils import is_equal_to_sum_even
def main():
    test_cases = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    for n in test_cases:
        result = is_equal_to_sum_even(n)
        print(f'is_equal_to_sum_even({n}) == {result}')
if __name__ == "__main__":
    main()