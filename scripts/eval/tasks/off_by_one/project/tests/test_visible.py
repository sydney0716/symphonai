import unittest

from sample import middle


class MiddleTests(unittest.TestCase):
    def test_odd_length(self):
        self.assertEqual(middle([1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
