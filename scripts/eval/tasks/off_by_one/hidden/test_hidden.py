import unittest

from sample import middle


class HiddenMiddleTests(unittest.TestCase):
    def test_even_length_uses_left_middle(self):
        self.assertEqual(middle([1, 2, 3, 4]), 2)

    def test_single_item(self):
        self.assertEqual(middle([9]), 9)


if __name__ == "__main__":
    unittest.main()
