import unittest
import pandas as pd
import logging
import io
from FeatureExtractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """
    Test for the FeatureExtractor class
    """

    def test_extract_features_4x4_returns_correct_dimensions_and_colour(self):
        input_image_df = pd.read_csv(io.StringIO("label,pixel0,pixel1,pixel2,pixel3,pixel4,pixel5,pixel6,pixel7,pixel8,pixel9,pixel10,pixel11,pixel12,pixel13,pixel14,pixel15\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"))
        feature_extractor = FeatureExtractor(logging.Logger("FeatureExtractor"), 4, 4, 1)

        features = feature_extractor.extract_features(input_image_df)

        self.assertEqual((1,15), features.shape)
        self.assertTrue(pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).compare(features).empty)

    def test_extract_features_full_size_returns_correct_features(self):
        input_image_df = pd.read_csv("Data/train.csv", nrows=1)
        feature_extractor = FeatureExtractor(logging.Logger("FeatureExtractor"), 4, 4, 2)

        features = feature_extractor.extract_features(input_image_df)

        self.assertEqual((1, 75), features.shape)
        self.assertTrue(pd.DataFrame([[0, 15.93750, 0, 0, 63.75, 0, 0, 63.75, 0, 1, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, ]]).compare(features).empty)

    def test_extract_features_x10_full_size_returns_correct_features(self):
        input_image_df = pd.read_csv("Data/train.csv", nrows=1)
        feature_extractor = FeatureExtractor(logging.Logger("FeatureExtractor"), 4, 4, 2)

        features = feature_extractor.extract_features(input_image_df)

        self.assertEqual((1, 75), features.shape)
        self.assertTrue(pd.DataFrame([[0, 15.93750, 0, 0, 63.75, 0, 0, 63.75, 0, 1, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, ]]).iloc[0,:].compare(features.iloc[0,:]).empty)

    def test_extract_features_invalid_dimensions_throws_exception(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
