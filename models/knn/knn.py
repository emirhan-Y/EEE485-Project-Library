import numpy as np

from util import quick_sort_with_trace
from .knn_error import knn_error
from util import log
from numpy.linalg import inv


class knn:
    def __init__(self, **kwargs):
        self.__NO_DATA_POINTS_AT_INIT_FLAG = False
        self.__NO_LABELS_AT_INIT_FLAG = False
        self.__data_points = []
        self.__labels = []
        if 'data_points' in kwargs:
            self.__data_point_loader(kwargs['data_points'])
        else:
            self.__no_data_at_initialization_flag = True

        if 'labels' in kwargs:
            self.__label_loader(kwargs['labels'])
        else:
            self.__NO_LABELS_AT_INIT_FLAG = True

        if self.__NO_DATA_POINTS_AT_INIT_FLAG and not self.__NO_LABELS_AT_INIT_FLAG:
            log.e('knn initialization error', 'Labels inputted without data points!')
            raise knn_error('Loading labels to knn without data points is not allowed!')

        if not self.__NO_DATA_POINTS_AT_INIT_FLAG and self.__NO_LABELS_AT_INIT_FLAG:
            log.e('knn initialization error', 'Data points inputted without labels!')
            raise knn_error('Loading data points to knn without labels is not allowed!')

        if len(self.__data_points) != len(self.__labels):
            log.e('knn initialization error', 'Data point and label array sizes are not equal')
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
