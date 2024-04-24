import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_signature_data(abs_path, seed, *, test_percentage=0.2, hot_encode=False):
    data_points = []
    labels = []
    ii = 0
    rng = np.random.default_rng(seed=seed)
    data_folder = abs_path
    for data_path in os.listdir(data_folder):
        data_points.append([])
        yeah = True
        abs_data_path = os.path.join(data_folder, data_path)
        for data_instance in os.listdir(abs_data_path):
            if yeah:
                labels.append(data_instance.split('_')[0])
                yeah = False
            data_point = cv2.imread(os.path.join(abs_data_path, data_instance), 0).flatten()
            data_points[ii].append(data_point)
        ii += 1

    min_length = min(len(sublist) for sublist in data_points)
    trimmed_data_points = [sublist[:min_length] for sublist in data_points]

    # Convert the trimmed lists into a NumPy array
    data_points = np.array(trimmed_data_points)
    for i in range(data_points.shape[0]):
        rng.shuffle(data_points[i, :])  # Shuffle in-place

    split_index = int(data_points.shape[1] * (1 - test_percentage))
    train_X = data_points[:, :split_index, :]
    train_Y = np.full((train_X.shape[0], train_X.shape[1]), np.arange(train_X.shape[0]).reshape(-1, 1))

    train_X = train_X.reshape(train_X.shape[0] * train_X.shape[1], train_X.shape[2])

    test_X = data_points[:, split_index:, :]
    test_Y = np.full((test_X.shape[0], test_X.shape[1]), np.arange(test_X.shape[0]).reshape(-1, 1))

    test_X = test_X.reshape(test_X.shape[0] * test_X.shape[1], test_X.shape[2])

    train_X = train_X.reshape(train_X.shape[0], 1, 50, 126) / 255
    test_X = test_X.reshape(test_X.shape[0], 1, 50, 126) / 255

    num_classes = train_Y.shape[0]

    train_Y = train_Y.reshape(train_Y.shape[0] * train_Y.shape[1], 1)
    test_Y = test_Y.reshape(test_Y.shape[0] * test_Y.shape[1], 1)

    if hot_encode:
        train_Y = np.eye(num_classes)[train_Y]
        test_Y = np.eye(num_classes)[test_Y]

    return train_X, train_Y, test_X, test_Y


def confusion_matrix(y_true, y_prediction):
    """
        Constructs the Confusion Matrix in the form of an array

        Parameters
        ----------
        y_true : Vector of real labels of test dataset.
        y_prediction :  Vector of predicted labels of test dataset.
        Returns
        -------
        cm : Confusion Matrix numpy array
    """
    # Determine the number of classes
    num_class = max(np.max(y_true), np.max(y_prediction)) + 1

    # Initialize confusion matrix array with zeros
    cm = np.zeros((num_class, num_class), dtype=int)

    # Increment the corresponding entry in the confusion matrix for each pair of true and predicted labels
    for i in range(len(y_true)):
        cm[y_true[i], y_prediction[i]] += 1
    return cm


def draw_confusion_matrix(y_true, y_prediction, label, x_ticks=None, y_ticks=None):
    """
        Draws the confusion matrix
        Parameters
        ----------
        y_prediction : Vector of predicted labels of test dataset.
        y_true :  Vector of real labels of test dataset.
        label : Are we drawing confusion matrix for test or training accuracy.
        Returns
        -------
        cm : Confusion matrix as a numpy array.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_prediction)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    if x_ticks and y_ticks:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=x_ticks, yticklabels=y_ticks)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {label}')
    plt.show()
    return cm
