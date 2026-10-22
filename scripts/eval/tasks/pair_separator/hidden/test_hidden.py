import inspect
import unittest

from sample import parse_pair


class HiddenParsePairTests(unittest.TestCase):
    def test_colon_separator_is_preserved(self):
        self.assertEqual(parse_pair("left:right"), ("left", "right"))

    def test_missing_separator_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_pair("leftright")

    def test_equals_separator(self):
        self.assertEqual(parse_pair("left=right"), ("left", "right"))

    def test_signature_is_unchanged(self):
        self.assertEqual(str(inspect.signature(parse_pair)), "(text: str) -> tuple[str, str]")


if __name__ == "__main__":
    unittest.main()
