from unittest import TestCase, mock

import app.resources.resources as resources

from tests.utils import CustomTestCase


class TestUtilityFunctions(TestCase):

    def test_classify_hour(self):
        ranges_classification = (
            (8, 1),
            (12, 2),
            (16, 3),
            (20, 4),
        )
        for hour, classification in ranges_classification:
            self.assertEqual(resources.classify_hour(hour),
                             classification)

    def test_classify_hour_raises_exception(self):
        hour_out_of_range = 7
        with self.assertRaises(ValueError):
            resources.classify_hour(hour_out_of_range)
        hour_out_of_range = 24
        with self.assertRaises(ValueError):
            resources.classify_hour(hour_out_of_range)

    def test_get_season(self):
        expectations = {
            12: 0,
            2: 0,
            3: 1,
            5: 1,
            7: 2,
            9: 3,
        }
        for month, expected_season in expectations.items():
            self.assertEqual(resources.get_season(month),
                             expected_season)

    def test_get_season_raises_exception(self):
        with self.assertRaises(ValueError):
            resources.get_season(0)
        with self.assertRaises(ValueError):
            resources.get_season(13)


class TestModellingNNClass(CustomTestCase):

    @mock.patch('app.resources.resources.get_NN_model_data', rv=None)
    def test_model_NN(self, mocked):
        """  Mocking return values from database """
        pass


# runs the unit tests in the module
if __name__ == '__main__':
    unittest.main()
