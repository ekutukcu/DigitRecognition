import pandas as pd
import csv

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from FeatureExtractor import FeatureExtractor

if __name__ == '__main__':
    digit_recogniser = FeatureExtractor("train.csv", 28, 28)

    with open('Data/computed_image_metrics.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for i   in range(len(digit_recogniser.image_array)):
            csv_writer.writerow(digit_recogniser.image_array[i])

    digit_recogniser = FeatureExtractor("test.csv", 28, 28, False)
    image_arrays = digit_recogniser.extract_features("test.csv")

    with open('Data/computed_test_metrics.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(image_arrays)

    digit_df = pd.read_csv("Data/computed_image_metrics.csv")

    print(digit_df.shape)
    x_data = digit_df.iloc[:, 1:]
    y_data = digit_df.iloc[:, [0]]

    scaler = preprocessing.StandardScaler().fit(x_data)
    x_data_scaled = scaler.transform(x_data)

    logistic_model = LogisticRegression(random_state=0, penalty='l1', solver='saga', tol=0.1).fit(x_data_scaled,
                                                                                                  y_data.values[:, 0])
    accuracy = logistic_model.score(x_data_scaled, y_data.values[:, 0])

    print(f"Training completed. Accuracy Rate: {accuracy * 100}%")

    test_data_file_path = "Data/computed_test_metrics.csv"

    test_data_df = pd.read_csv(test_data_file_path)
    scaler = preprocessing.StandardScaler().fit(test_data_df)
    test_data_scaled = scaler.transform(test_data_df)

    print(test_data_scaled.shape)
    print(test_data_df.shape)

    predictions = logistic_model.predict(test_data_scaled)
    pd.Series({"Label": predictions}).to_csv("predictions.csv")
