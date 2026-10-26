import inspect
import unittest

from names import display_name, format_name
from report import report_title


class DisplayNameTests(unittest.TestCase):
    def test_new_and_deprecated_names_keep_the_behavior(self):
        self.assertEqual(display_name("  quarterly results "), "Quarterly Results")
        self.assertEqual(format_name("  quarterly results "), "Quarterly Results")

    def test_project_caller_uses_new_name(self):
        source = inspect.getsource(report_title)
        self.assertIn("display_name", source)
        self.assertNotIn("format_name", source)


if __name__ == "__main__":
    unittest.main()
