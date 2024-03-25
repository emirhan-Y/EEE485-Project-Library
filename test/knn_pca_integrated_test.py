"""
KNN & PCA integrated test script
================================
* Currently under testing
"""
import os
import cv2

from analysis import pca
from models import knn
import numpy as np

from util import log

_LOGGER = log(True, True, True)
"""Module logger"""

if __name__ == '__main__':
    data_points, labels = [], []
    main_data_path = os.path.abspath('../data/final')
    for data_path in os.listdir(main_data_path):
        if data_path != 'test':
            cur_data_path = os.path.join(main_data_path, data_path)
            for data_instance in os.listdir(cur_data_path):
                data_point = cv2.imread(os.path.join(cur_data_path, data_instance), 0).flatten()
                data_points.append(data_point)
                labels.append(data_instance.split('_')[0])
    principal_component_analysis = pca(np.array(data_points), 0.99)
    principal_component_analysis.analyze()
    new_dataset = principal_component_analysis.get_new_dataset()

    test_dataset_path = os.path.abspath('../data/final/test')
    test_dataset, test_labels = [], []
    for test_data in os.listdir(test_dataset_path):
        test_data_point = cv2.imread(os.path.join(test_dataset_path, test_data), 0).flatten()
        test_dataset.append(test_data_point)
        test_labels.append(test_data.split('_')[0])
    test_dataset = np.array(test_dataset)
    test_labels = np.array(test_labels)
    test_dataset = principal_component_analysis.reform_data(test_dataset)

    knn_0 = knn(data_points=new_dataset, labels=labels)
    for i in range(len(test_dataset)):
        y1 = knn_0.test_point(test_dataset[i], 7)
        _LOGGER.debug_message('knn test point', 'predicted: ' + str(y1) + ', actual: ' + str(test_labels[i]))
