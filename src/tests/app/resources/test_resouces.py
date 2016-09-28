from unittest import TestCase, mock

from app.resources.resources import classify_hour
from tests.utils import CustomTestCase


class TestUtilsClass(TestCase):

    def test_classify_hour(self):
        ranges_classification = (
            (8, 1),
            (12, 2),
            (16, 3),
            (20, 4),
        )
        for hour, classification in ranges_classification:
            self.assertEqual(classify_hour(hour), classification)

    def test_classify_hour_raises_exception(self):
        hour_out_of_range = 0
        with self.assertRaises(Exception):
            classify_hour(hour_out_of_range)


class TestModellingNNClass(CustomTestCase):

    @mock.patch('app.resources.resources.get_NN_model_data', rv=None)
    def test_model_NN(self, mocked):
        """  Mocking return values from database """
        pass


# runs the unit tests in the module
if __name__ == '__main__':
    unittest.main()
