import unittest

from sample import normalize_name


class HiddenNormalizeNameTests(unittest.TestCase):
    def test_existing_trim_and_lowercase_behaviour(self):
        self.assertEqual(normalize_name("  Hello  "), "hello")

    def test_collapses_spaces(self):
        self.assertEqual(normalize_name("Hello   SymphonAI"), "hello-symphonai")

    def test_collapses_tabs_and_newlines(self):
        self.assertEqual(normalize_name(" A\t B\nC "), "a-b-c")


if __name__ == "__main__":
    unittest.main()
