import unittest

from counts import parse_count


class ParseCountTests(unittest.TestCase):
    def test_valid_integer_text(self):
        self.assertEqual(parse_count("42"), 42)
        self.assertEqual(parse_count("-3"), -3)


if __name__ == "__main__":
    unittest.main()
