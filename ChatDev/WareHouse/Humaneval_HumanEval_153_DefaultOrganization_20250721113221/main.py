'''
This file contains test cases for the StrongestExtensionFinder class to ensure
the functionality works as expected.
'''
import unittest
from main import StrongestExtensionFinder
class TestStrongestExtensionFinder(unittest.TestCase):
    def test_strongest_extension(self):
        finder = StrongestExtensionFinder('my_class', ['AA', 'Be', 'CC'])
        self.assertEqual(finder.find_strongest_extension(), 'my_class.AA')
        finder = StrongestExtensionFinder('Slices', ['SErviNGSliCes', 'Cheese', 'StuFfed'])
        self.assertEqual(finder.find_strongest_extension(), 'Slices.SErviNGSliCes')
        finder = StrongestExtensionFinder('Test', ['abc', 'XYZ', 'xYz'])
        self.assertEqual(finder.find_strongest_extension(), 'Test.XYZ')
        finder = StrongestExtensionFinder('Empty', [])
        self.assertIsNone(finder.find_strongest_extension())
if __name__ == "__main__":
    unittest.main()