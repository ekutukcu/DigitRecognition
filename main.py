import pandas as pd
import csv
import logging

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from FeatureExtractor import FeatureExtractor

if __name__ == '__main__':
    logger = logging.Logger("MainLogger")
    digit_recogniser = FeatureExtractor(logger, 28, 28,2,True)

    pixel_training_data_df = pd.read_csv("Data/train.csv")
    training_feature_df = digit_recogniser.extract_features(pixel_training_data_df)

    print(training_feature_df.shape)
    x_data = training_feature_df.iloc[:, 1:]
    y_data = training_feature_df.iloc[:, [0]]

    scaler = preprocessing.StandardScaler().fit(x_data)
    x_data_scaled = scaler.transform(x_data)

    logistic_model = LogisticRegression(random_state=0, penalty='l1', solver='saga', tol=0.1).fit(x_data_scaled,
                                                                                                  y_data.values[:, 0])
    accuracy = logistic_model.score(x_data_scaled, y_data.values[:, 0])

    print(f"Training completed. Accuracy Rate: {accuracy * 100}%")

    # create predictions for the test data now
    pixel_test_data_df = pd.read_csv("Data/test.csv")
    digit_recogniser = FeatureExtractor(logger, 28, 28, False)
    test_feature_df = digit_recogniser.extract_features(pixel_test_data_df)
    scaler = preprocessing.StandardScaler().fit(test_feature_df)
    test_data_scaled = scaler.transform(test_feature_df)

    print(test_data_scaled.shape)
    print(test_feature_df.shape)

    predictions = logistic_model.predict(test_data_scaled)
    pd.Series({"Label": predictions}).to_csv("predictions.csv")
