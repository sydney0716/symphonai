import unittest

from sample import middle


class HiddenMiddleTests(unittest.TestCase):
    def test_even_lengths_use_lower_middle(self):
        self.assertEqual(middle([1, 2, 3, 4]), 2)
        self.assertEqual(middle([1, 2]), 1)
        self.assertEqual(middle([1, 2, 3, 4, 5, 6]), 3)


if __name__ == "__main__":
    unittest.main()
