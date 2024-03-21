"""
KNN model definitions
=====================

classes
-------
* knn: holds the KNN data points and calculates test point results
"""

import numpy as np

from util import quick_sort_with_trace
from .knn_error import knn_error
from util import log
from numpy.linalg import inv


class knn:
    """
    K-Nearest Neighbor model class
    """
    def __init__(self, **kwargs):
        """
        knn model constructor
        :param kwargs:
        """
        self.__NO_DATA_POINTS_AT_INIT_FLAG = False  # set if no data points were given at the constructor
        self.__NO_LABELS_AT_INIT_FLAG = False  # set if no labels were given at the constructor
        self.__data_points = []  # initialize the data point array of the knn model as empty
        self.__labels = []  # initialize the label array of the knn model as empty
        if 'data_points' in kwargs:  # if data_points entry is given at the constructor
            self.__data_point_loader(kwargs['data_points'])  # go to the data point array loader
        else:  # if data_points entry is not given at the constructor
            self.__no_data_at_initialization_flag = True  # set the no data point at constructor flag to true

        if 'labels' in kwargs:  # if labels entry is given at the constructor
            self.__label_loader(kwargs['labels'])  # go to the label array loader
        else:  # if labels entry is not given at the constructor
            self.__NO_LABELS_AT_INIT_FLAG = True  # set the no label at constructor flag to true

        if self.__NO_DATA_POINTS_AT_INIT_FLAG and not self.__NO_LABELS_AT_INIT_FLAG:  # if labels were given but not
            # data points
            log.e('knn initialization error', 'Labels inputted without data points!')  # raise an error
            raise knn_error('Loading labels to knn without data points is not allowed!')

        if not self.__NO_DATA_POINTS_AT_INIT_FLAG and self.__NO_LABELS_AT_INIT_FLAG:  # if data points were given but
            # not labels
            log.e('knn initialization error', 'Data points inputted without labels!')  # raise an error
            raise knn_error('Loading data points to knn without labels is not allowed!')

        if len(self.__data_points) != len(self.__labels):  # if the number of data points and labels do not match
            log.e('knn initialization error', 'Data point and label array sizes are not equal')  # raise an error
            raise knn_error('Data point and label array size mismatch')

    def __data_point_loader(self, data_points):
        if isinstance(data_points, list):
            self.__data_points = np.array(data_points)
        elif isinstance(data_points, np.ndarray):
            self.__data_points = data_points
        else:
            log.e('knn input type mismatch',
                  'Expected type list or ndarray for data point input, got ' + type(data_points).__name__ + ' instead')
            raise knn_error('knn invalid data point input type')

    def __label_loader(self, labels):
        if isinstance(labels, list):
            self.__labels = np.array(labels)
        elif isinstance(labels, np.ndarray):
            self.__labels = labels
        else:
            log.e('knn input type mismatch',
                  'Expected type list or ndarray for label input, got ' + type(labels).__name__ + ' instead')
            raise knn_error('knn invalid label input type')

    def test_point(self, test_x, K):
        if isinstance(test_x, np.ndarray):
            pass
        elif isinstance(test_x, list):
            test_x = np.array(test_x)
        else:
            log.e('Invalid test point type',
                  'KNN test_point should be of type np.ndarray or list, but got ' + type(test_x).__name__ + ' instead')
            raise TypeError('Invalid KNN test point type: ' + type(test_x).__name__)
        distance_vector = []
        trace = []
        for i in range(len(self.__data_points)):
            distance_vector.append(np.dot(self.__data_points[i] - test_x, self.__data_points[i] - test_x))
            trace.append(i)
        distance_vector, trace = quick_sort_with_trace(distance_vector, trace)
        res_label_list = []
        res_label_count = []
        for i in range(K):
            if self.__labels[trace[i]] not in res_label_list:
                res_label_list.append(self.__labels[trace[i]])
                res_label_count.append(1)
            else:
                index = res_label_list.index(self.__labels[trace[i]])
                res_label_count[index] += 1
        res_label_count, res_label_list = quick_sort_with_trace(res_label_count, res_label_list)
        return res_label_list[-1]

    # def enter_data(self, data_points, labels):
