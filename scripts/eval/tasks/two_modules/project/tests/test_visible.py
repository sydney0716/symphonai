import unittest

from report import report_title


class ReportTitleTests(unittest.TestCase):
    def test_title_formats_name(self):
        self.assertEqual(report_title("  quarterly results "), "Report: Quarterly Results")


if __name__ == "__main__":
    unittest.main()
