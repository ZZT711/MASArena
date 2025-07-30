'''
This file contains unit tests for the is_equal_to_sum_even function to ensure its correctness.
'''
import unittest
from main import is_equal_to_sum_even
class TestIsEqualToSumEven(unittest.TestCase):
    def test_cases(self):
        self.assertFalse(is_equal_to_sum_even(4), "Should be False for 4")
        self.assertFalse(is_equal_to_sum_even(6), "Should be False for 6")
        self.assertTrue(is_equal_to_sum_even(8), "Should be True for 8")
        self.assertTrue(is_equal_to_sum_even(10), "Should be True for 10")
        self.assertTrue(is_equal_to_sum_even(12), "Should be True for 12")
        self.assertFalse(is_equal_to_sum_even(1), "Should be False for 1")
        self.assertFalse(is_equal_to_sum_even(7), "Should be False for 7")
        self.assertTrue(is_equal_to_sum_even(20), "Should be True for 20")
        self.assertFalse(is_equal_to_sum_even(0), "Should be False for 0")
        self.assertFalse(is_equal_to_sum_even(-2), "Should be False for negative numbers")
if __name__ == '__main__':
    unittest.main()