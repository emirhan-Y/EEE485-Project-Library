"""
KNN model definitions
=====================

classes
-------
* knn: holds the KNN data points and calculates test point results
* knn_error: manage error generated by the knn model
"""

import numpy as np

from util import log
from util import quick_sort_with_trace


class knn:
    """
    K-Nearest Neighbor model class
    """
    DEBUG_MSG = True
    WARNING_MSG = True
    ERROR_MSG = True
    _LOGGER = log(DEBUG_MSG, WARNING_MSG, ERROR_MSG)

    def __init__(self, **kwargs):
        """
        knn model constructor
        :param kwargs:
        """
        self._NO_DATA_POINTS_AT_INIT_FLAG = False  # set if no data points were given at the constructor
        self._NO_LABELS_AT_INIT_FLAG = False  # set if no labels were given at the constructor
        self._data_points = []  # initialize the data point array of the knn model as empty
        self._labels = []  # initialize the label array of the knn model as empty
        self._class_map = {}  # initialize the class map dictionary to empty
        if 'data_points' in kwargs:  # if data_points entry is given at the constructor
            self._data_point_loader(kwargs['data_points'])  # go to the data point array loader
        else:  # if data_points entry is not given at the constructor
            self._no_data_at_initialization_flag = True  # set the no data point at constructor flag to true

        if 'labels' in kwargs:  # if labels entry is given at the constructor
            self._label_loader(kwargs['labels'])  # go to the label array loader
        else:  # if labels entry is not given at the constructor
            self._NO_LABELS_AT_INIT_FLAG = True  # set the no label at constructor flag to true

        if self._NO_DATA_POINTS_AT_INIT_FLAG and not self._NO_LABELS_AT_INIT_FLAG:  # if labels were given but not
            # data points
            knn._LOGGER.error_message('knn initialization error',
                                      'Labels inputted without data points!')  # raise an error
            raise knn_error('Loading labels to knn without data points is not allowed!')

        if not self._NO_DATA_POINTS_AT_INIT_FLAG and self._NO_LABELS_AT_INIT_FLAG:  # if data points were given but
            # not labels
            knn._LOGGER.error_message('knn initialization error',
                                      'Data points inputted without labels!')  # raise an error
            raise knn_error('Loading data points to knn without labels is not allowed!')

        if len(self._data_points) != len(self._labels):  # if the number of data points and labels do not match
            knn._LOGGER.error_message('knn initialization error',
                                      'Data point and label array sizes are not equal')  # raise an error
            raise knn_error('Data point and label array size mismatch')

    def _data_point_loader(self, data_points: np.ndarray or list) -> None:
        """
        Load the given data set in the constructor to the knn instance

        Parameters
        ------------
        data_points : 2d np.ndarray or list
            Data points to apply the KNN on.

        """
        if isinstance(data_points, list):  # ndarray conversions
            self._data_points = np.array(data_points) / 255
        elif isinstance(data_points, np.ndarray):
            self._data_points = data_points / 255
        else:  # error if inconvertible
            knn._LOGGER.error_message('knn input type mismatch',
                                      'Expected type list or ndarray for data point input, got ' + type(
                                          data_points).__name__ + ' instead')
            raise knn_error('knn invalid data point input type')

    def _label_loader(self, labels: np.ndarray or list) -> None:
        """
        Load the given data set labels in the constructor to the knn instance

        Parameters
        ------------
        labels : 2d np.ndarray or list
            Data point labels to apply the KNN on.

        """
        if isinstance(labels, list):  # ndarray conversions
            self._labels = np.array(labels)
        elif isinstance(labels, np.ndarray):
            self._labels = labels
        else:  # error if inconvertible
            knn._LOGGER.error_message('knn input type mismatch',
                                      'Expected type list or ndarray for label input, got ' + type(
                                          labels).__name__ + ' instead')
            raise knn_error('knn invalid label input type')

    def test_point(self, test_x: np.ndarray, K: int) -> list:
        """
        Get KNN prediction on a new point.

        Parameters
        ----------
        test_x : np.ndarray or list
            Test point to get the KNN prediction on.

        K: int
            The value of K for this given data point.

        Returns
        -------
        y_hat: list of label_type
            An array of predicted values of the label given test point and K value
        """
        if isinstance(test_x, np.ndarray):  # ndarray conversions
            pass
        elif isinstance(test_x, list):
            test_x = np.array(test_x)
        else:
            knn._LOGGER.error_message('Invalid test point type',
                                      'KNN test_point should be of type np.ndarray or list, but got ' + type(
                                          test_x).__name__ + ' instead')
            raise TypeError('Invalid KNN test point type: ' + type(test_x).__name__)
        # create a new array with rows [this point's distance to test point, this point's label]
        distance_vector = np.sum((self._data_points - test_x) ** 2, axis=1)
        # sort the array according to the distances
        sorted_indices = np.argsort(distance_vector)
        labels_sorted = self._labels[sorted_indices]
        # find the most recurring label among the K points with the smallest distance
        res_label_list = []
        res_label_count = []
        for i in range(K):
            if labels_sorted[i] not in res_label_list:
                res_label_list.append(labels_sorted[i])
                res_label_count.append(1)
            else:
                index = res_label_list.index(labels_sorted[i])
                res_label_count[index] += 1
        res_label_count = np.array(res_label_count)
        res_label_list = np.array(res_label_list)
        indices = np.argsort(-res_label_count)
        res_label_count = res_label_count[indices]
        res_label_list = res_label_list[indices]

        if res_label_count.size == 0:  # Check if the array is empty
            raise RuntimeError('Empty KNN prediction!')

        largest = res_label_count[0]  # The first element is the largest
        largest_elements = res_label_list[res_label_count == largest]  # Extract all elements equal to the largest

        return largest_elements


class knn_error(Exception):
    def __init__(self, message):
        super().__init__(message)