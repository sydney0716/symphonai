import unittest

from sample import parse_pair


class ParsePairTests(unittest.TestCase):
    def test_colon_pair(self):
        self.assertEqual(parse_pair("left:right"), ("left", "right"))

    def test_missing_separator(self):
        with self.assertRaises(ValueError):
            parse_pair("leftright")


if __name__ == "__main__":
    unittest.main()
