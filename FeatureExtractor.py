import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pandas import DataFrame
from logging import Logger
from skimage.transform import rotate


class FeatureExtractor:
    """
    CLass for extracting image feature from a row containing a list of pixels that
    represents an nxn image
    """
    image_array = []
    k_start = 0
    image_width = 0
    image_height = 0
    image_quartering_count = 0
    logger = None

    def __init__(self, logger: Logger, image_width: int, image_height: int, image_quartering_count: int,
                 is_first_column_label: bool = False):
        """
        Constructor taking the image dimensions and a boolean specifying whether the data includes
        the classification label
        """
        if is_first_column_label:
            self.k_start = 1
        self.logger = logger
        self.image_width = image_width
        self.image_height = image_height
        self.image_quartering_count = image_quartering_count

    def __create_pixel_array(self, dataframe_row: DataFrame) -> DataFrame:
        """
        Creates an image_width X image_height 2D pixel array from the dataframe row
        """
        image = []
        k = self.k_start
        for i in range(self.image_height):
            image.append([])
            for j in range(self.image_width):
                image[i].append(dataframe_row[k])
                k += 1

        plt.set_cmap('gray')
        return cv2.cvtColor(plt.cm.jet(image).astype('float32'), cv2.COLOR_BGR2GRAY)

    def __get_quadrants(self,number_of_iterations: int, binary_image: DataFrame) -> []:
        """
        Splits image into quadrants and calculates metrics recursively
        :param number_of_iterations: number of times to recursively execute
        :param binary_image: image to split as numpy array
        :return: array with calculate symmetry and intensity metrics
        """
        processed_data = []

        if number_of_iterations > 0:
            # calculate metrics for whole image
            symmetry_score = self.__calculate_symmetry_score(binary_image)
            intensity = binary_image.mean()
            rotated_symmetry_score = self.__calculate_symmetry_score(rotate(binary_image, 90))
            processed_data += [symmetry_score, intensity, rotated_symmetry_score]

            # split into quadrants
            quadrants = self.__split_image_quadrants(binary_image)

            # calculate metrics recursively for quadrants
            for quadrant in quadrants:
                symmetry_score = self.__calculate_symmetry_score(quadrant)
                intensity = quadrant.mean()
                rotated_symmetry_score = self.__calculate_symmetry_score(rotate(quadrant, 90))
                processed_data += [symmetry_score, intensity, rotated_symmetry_score]
                processed_data += self.__get_quadrants(number_of_iterations - 1, quadrant)

        return processed_data

    def extract_features(self, training_data_df: DataFrame) -> DataFrame:
        """
        Extracts and returns the features (intensity, symmetry) for the image and the sub sections of the image from the
        DataFrame training_data_df
        """
        count_of_training = 0
        self.logger.debug("Training over dataset ", training_data_df.shape)

        image_array = []

        for index, row in training_data_df.iterrows():
            image = self.__create_pixel_array(row)
            data = []
            intensity = image.mean()

            binary_image = cv2.threshold(image, intensity, 255, cv2.THRESH_BINARY)[1]

            processed_data = self.__get_quadrants(self.image_quartering_count, binary_image)

            if self.k_start == 1:
                image_array.append([row[0]] + processed_data)
            else:
                image_array.append(processed_data)
            count_of_training += 1

        self.logger.debug("Trained over ", count_of_training, " iterations.")
        return pd.DataFrame(image_array)

    def __split_image_quadrants(self, binary_image: DataFrame) -> []:
        """
        Splits the given image ndarray into quadrants
        """
        nrows, ncols = binary_image.shape

        self.logger.debug(binary_image.shape)

        rsplit, csplit = nrows // 2, ncols // 2
        quadrants = [
            binary_image[:rsplit, :csplit],
            binary_image[:rsplit, csplit:],
            binary_image[rsplit:, :csplit],
            binary_image[rsplit:, csplit:],
        ]
        return quadrants

    def __calculate_symmetry_score(self, image: DataFrame) -> float:
        """Test the symmetry between two images by calculating the intersection/union of pixels"""
        flipped_image = np.flip(image, axis=1)
        intersection = cv2.bitwise_and(image, flipped_image)

        union = cv2.bitwise_or(image, flipped_image)

        intersection = cv2.countNonZero(intersection)

        if intersection == 0:
            self.logger.debug("No intersection found")
            return intersection

        res = cv2.countNonZero(intersection) / cv2.countNonZero(union)
        return res
