'''
This file contains test cases for the Strongest_Extension function to ensure 
its correctness and functionality.
'''
import unittest
from main import Strongest_Extension
class TestStrongestExtension(unittest.TestCase):
    def test_example_case_1(self):
        self.assertEqual(Strongest_Extension('Slices', ['SErviNGSliCes', 'Cheese', 'StuFfed']), 'Slices.SErviNGSliCes')
    def test_example_case_2(self):
        self.assertEqual(Strongest_Extension('my_class', ['AA', 'Be', 'CC']), 'my_class.AA')
    def test_case_with_same_strength(self):
        self.assertEqual(Strongest_Extension('Test', ['AbC', 'aBc', 'ABC']), 'Test.AbC')
    def test_case_with_no_extensions(self):
        self.assertEqual(Strongest_Extension('EmptyClass', []), 'EmptyClass.')
    def test_case_with_lowercase_only(self):
        self.assertEqual(Strongest_Extension('LowerClass', ['abc', 'def']), 'LowerClass.abc')
if __name__ == '__main__':
    unittest.main()