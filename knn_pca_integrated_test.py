"""
KNN & PCA integrated test script
================================
* Currently under testing
"""
import os
import cv2

from help import draw_confusion_matrix
from models import pca
from models import knn
import numpy as np
import matplotlib.pyplot as plt

from util import log

_LOGGER = log(True, True, True)
"""Module logger"""

if __name__ == '__main__':
    data_points, training_labels = [], []
    main_data_path = os.path.abspath('_data/final')
    for data_path in os.listdir(main_data_path):
        if data_path != 'test':
            cur_data_path = os.path.join(main_data_path, data_path)
            for data_instance in os.listdir(cur_data_path):
                data_point = cv2.imread(os.path.join(cur_data_path, data_instance), 0).flatten()
                data_points.append(data_point)
                training_labels.append(data_instance.split('_')[0])
    data_points = np.array(data_points)
    data_points = 255 - data_points
    principal_component_analysis = pca(data_points, 0.999)
    principal_component_analysis.analyze()
    new_dataset = principal_component_analysis.get_new_dataset()

    test_dataset_path = os.path.abspath('_data/final/test')
    test_dataset, test_labels = [], []
    for test_data in os.listdir(test_dataset_path):
        test_data_point = cv2.imread(os.path.join(test_dataset_path, test_data), 0).flatten()
        test_dataset.append(test_data_point)
        test_labels.append(test_data.split('_')[0])
    test_dataset = np.array(test_dataset)
    test_dataset = 255 - test_dataset
    test_dataset = principal_component_analysis.reform_data(test_dataset)

    map_str_to_int = {}
    map_int_to_str = {}
    labels = []
    test_labels_int = []
    for i in range(len(test_labels)):
        if test_labels[i] not in labels:
            map_str_to_int[test_labels[i]] = len(labels)
            map_int_to_str[len(labels)] = test_labels[i]
            labels.append(test_labels[i])
        test_labels_int.append(map_str_to_int[test_labels[i]])

    # K = 11 test
    initial_K = 11
    knn_0 = knn(data_points=new_dataset, labels=training_labels)
    predictions = []
    for i in range(len(test_dataset)):
        K = initial_K
        y1 = [-1, -1]
        while len(y1) != 1:
            y1 = knn_0.test_point(test_dataset[i], K)
            K += 1
        y1 = y1[0]
        y1_int = map_str_to_int[y1]
        predictions.append(y1_int)
        _LOGGER.debug_message('knn test point', 'predicted: ' + str(y1) + ', actual: ' + str(test_labels[i]))

    draw_confusion_matrix(test_labels_int, predictions, 'KNN Test Accuracy', x_ticks=labels, y_ticks=labels)

    # K vs accuracy test
    start_K = 1
    stop_K = 3200
    num_K = 64

    points = np.logspace(np.log10(start_K), np.log10(stop_K), num=num_K)
    K_points = np.unique(np.floor(points).astype(int))
    accuracy_arr = []
    knn_0 = knn(data_points=new_dataset, labels=training_labels)

    for i in range(len(K_points)):
        initial_K = K_points[i]
        cur_acc = 0
        for i in range(len(test_dataset)):
            K = initial_K
            y1 = [-1, -1]
            while len(y1) != 1 and K < min(3200, initial_K + 10):
                y1 = knn_0.test_point(test_dataset[i], K)
                K += 1
            y1 = y1[0]
            if y1 == test_labels[i]:
                cur_acc += 1
        cur_acc /= len(test_dataset)
        accuracy_arr.append(cur_acc)
        _LOGGER.debug_message('knn k vs acc', f'KNN Accuracy for K={initial_K}: {cur_acc}')

    plt.figure(figsize=(10, 5))
    plt.plot(K_points[:-1], np.array(accuracy_arr[:-1]), marker='o',
             linestyle='-')  # Plot x and y using lines and markers
    plt.xscale('log')  # Set the x-axis to a logarithmic scale
    plt.xlabel('K (logarithmic scaled)', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=20)
    plt.title('KNN K value vs Test Accuracy Plot', fontsize=32)
    plt.show()

    print('foo')
