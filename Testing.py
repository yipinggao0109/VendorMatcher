import unittest
from vendor_matcher import VendorMatcher

class TestVendorMatcher(unittest.TestCase):
    """
    Unit tests for the VendorMatcher class using the real dataset.
    Tests edge cases and core recommendation functionality.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load the matcher once for all test cases.
        """
        cls.matcher = VendorMatcher("data/G2 software - CRM Category Product Overviews.csv")

    def test_no_input(self):
        """
        Test default recommendation behavior when no category or capacity is provided.
        Should return a non-empty DataFrame ranked by Bayesian score.
        """
        df = self.matcher.recommend()
        self.assertFalse(df.empty, "Expected non-empty DataFrame for no-input default case")
        self.assertIn("bayesian_score", df.columns)

    def test_only_category(self):
        """
        Test recommendations filtered by category only.
        Should return results relevant to 'Sales Force Automation'.
        """
        df = self.matcher.recommend(category="Sales Force Automation")
        self.assertFalse(df.empty, "Expected matches for 'Sales Force Automation'")
        self.assertIn("bayesian_score", df.columns)

    def test_only_capacities(self):
        """
        Test recommendations filtered by a single capacity only.
        Should return software matched to 'Customization' feature.
        """
        df = self.matcher.recommend(capacities=["Customization"])
        self.assertFalse(df.empty, "Expected matches for 'Customization'")
        self.assertIn("final_score", df.columns)

    def test_both_inputs(self):
        """
        Test combined category and capacity filtering.
        Should return high-confidence results relevant to both inputs.
        """
        df = self.matcher.recommend(capacities=["Email Marketing"], category="Marketing Automation")
        self.assertFalse(df.empty, "Expected results for both inputs")
        self.assertIn("final_score", df.columns)

    def test_no_match(self):
        """
        Test edge case where no matching feature is found.
        Should return an empty DataFrame.
        """
        df = self.matcher.recommend(capacities=["ThisFeatureDoesNotExist"])
        self.assertTrue(df.empty, "Expected no matches for a non-existent feature")

# Entry point to run the test suite
if __name__ == '__main__':
    unittest.main()
