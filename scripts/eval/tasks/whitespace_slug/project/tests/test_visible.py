import unittest

from sample import normalize_name


class NormalizeNameTests(unittest.TestCase):
    def test_trims_and_lowercases(self):
        self.assertEqual(normalize_name("  Hello  "), "hello")


if __name__ == "__main__":
    unittest.main()
