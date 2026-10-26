import unittest

from counts import parse_count


class HiddenParseCountTests(unittest.TestCase):
    def test_malformed_values_raise_with_original_text(self):
        for value in ("", "  \t", "many"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as raised:
                    parse_count(value)
                self.assertIn(repr(value), str(raised.exception))

    def test_valid_integer_whitespace_is_preserved(self):
        self.assertEqual(parse_count(" 42 "), 42)


if __name__ == "__main__":
    unittest.main()
