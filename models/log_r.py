from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import pca
import sys


def add_column_ones_design_matrix(x):
    """
        Adds a column of 1's to column index 0 of x
        Parameters
        ----------
        x : input numpy array
        """
    # Add column 1 to the design matrix
    ones_column = np.ones((x.shape[0], 1))
    return np.hstack((ones_column, x))


def sigmoid(x):
    """
        Applies sigmoid function for a given input.
        Parameters
        ----------
        x : Input to the sigmoid function.
            For Logistic Regression purposes, apply dot product of data instance and weight vector
            before applying this function.
    """
    return 1 / (1 + np.exp(-x))


def prepare_test_train_xy(x, y, pca_var, k_str, test_size):
    """
        1) Applies PCA to the design matrix.
        2) Replace class string names with integers in the order of K_str.
        3) Adds columns of 1s to the design matrix after PCA.
        4) Splits X and y into train and test datasets based on the test size.

        Parameters
        ----------
        x : Initial unmodified design matrix.
        y : Initial unmodified response matrix. classes are writen as string class names.
        pca_var : Variance parameter for PCA
        test_size : Ratio of the dataset to allocate to test.
        k_str : list of class names (string).

        Returns
        -------
        x_train, x_test, y_train, y_test : Prepared test and train datasets.
    """
    # Enter PCA
    principal_component_analysis = pca(x, pca_var)
    principal_component_analysis.analyze()
    x = principal_component_analysis.get_new_dataset()

    # Replace class string names with integers in the order of K_str
    string_to_index = {string: i for i, string in enumerate(k_str)}
    y = np.array([string_to_index[string] for string in y])

    # Add column 1 to the design matrix
    x = add_column_ones_design_matrix(x)

    # Split Data into Training and Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test


def binary_logistic_train(x_train, y_train, iteration):
    """
        This function trains the logistic regression model for binary classification.
        Uses Newton Method to update the weights.

        Parameters
        ----------
        x_train : Design matrix of the training dataset.
        y_train : Response vector of the training dataset.
        iteration : Maximum number of Newton Method iterations if no convergence.

        Returns
        -------
        w_new : The final updated weight vector of the Logistic Model.
    """
    # Initialize model weights
    weights_init = np.zeros(x_train.shape[1])
    w_old = np.zeros(x_train.shape[1])
    w_new = np.copy(weights_init)
    iter_num = 0

    for i in range(iteration):
        w_old = np.copy(w_new)
        pi_arr = sigmoid(np.dot(x_train, w_old))
        w = np.eye(len(pi_arr))
        np.fill_diagonal(w, pi_arr * (1 - pi_arr))
        # Newton Method for updating weights
        xtwx = np.dot(np.dot(x_train.T, w), x_train)

        # Check if XTWX is singular
        if np.linalg.cond(xtwx) < 1 / sys.float_info.epsilon:
            # If XTWX is singular, add regularization to stabilize inversion
            w_new = w_old + np.dot(np.dot(np.linalg.inv(xtwx + 0.01 * np.eye(len(xtwx))), x_train.T), y_train - pi_arr)
        else:
            # Otherwise, proceed without regularization
            w_new = w_old + np.dot(np.dot(np.linalg.inv(xtwx), x_train.T), y_train - pi_arr)

        iter_num += 1

        if all(abs(w_new - w_old) < 10 ** (-8)):
            print('Weights Converged')
            print('Total Iterations Until Convergence: ', iter_num)
            break

    if not np.array_equal(w_new, w_old):
        print('Weights Did Not Converge')
        print('Total Iterations: ', iter_num)
    return w_new


def one_vs_all_classifier(x_train, y_train, k_str, iteration):
    """
    Applies all binary logistic regression classifiers and obtains each models weights
    Returns all model weights as an array.
    """
    # Initialize array
    all_classifier_weights = np.zeros((len(k_str), x_train.shape[1]))

    for i in range(len(k_str)):
        # Convert elements in y to 0 if they are not equal to i
        y_train_binary = np.where(y_train != i, 0, 1)

        # Train and obtain the model weights for ith classifier
        classifier_i_weights = binary_logistic_train(x_train, y_train_binary, iteration)

        # Save all model weights for later use.
        all_classifier_weights[i] = classifier_i_weights
    return all_classifier_weights


def test(x, y_true, all_classifier_weights, threshold):
    """
    x : Design Matrix to test
    y_true : Response matrix of the data x to be tested
    all_classifier_weights : All model weights from training
    all_classifier_weights : Weights of all trained classifiers
    """
    z = np.dot(x, all_classifier_weights.T)
    y_probability = sigmoid(z)
    # At this point the y_probability array is set
    # Check Accuracy by comparing y_prediction and y_true for each binary classifiers and the total 1 vs all classifier
    # If testing a single data point, also plot the probabilities on a graph

    # First test accuracies for binary classifiers
    for i in range(all_classifier_weights.shape[0]):
        print("CLASSIFIER", i)
        # assign ith classifier probability results to classes based on threshold.
        y_prediction_classifier_i = np.copy(y_probability[:, i])
        y_prediction_classifier_i[y_prediction_classifier_i < threshold] = 0
        y_prediction_classifier_i[y_prediction_classifier_i >= threshold] = 1
        # Convert elements in y_true to 0 if they are not equal to i
        y_true_binary = np.where(y_true != i, 0, 1)
        print("Classifier {} Accuracy: {}%\n".format(i+1, accuracy(y_prediction_classifier_i, y_true_binary)))

    # Now find accuracy of one vs all classifier.
    # First, using y_probability array find the classes of highest probability
    y_prediction = np.argmax(y_probability, axis=1)
    print("One Vs All Logistic Regression Classifier Accuracy: {}%\n".format(accuracy(y_prediction, y_true)))
    return y_probability, y_prediction


def accuracy(y_prediction, y_true):
    """
        Calculates the performance of the model by comparing real responses and the predicted responses
        and finds the Accuracy Rate.
        Parameters
        ----------
        y_prediction : Vector of predicted labels of test dataset.
        y_true :  Vector of real labels of test dataset.
        Returns
        -------
        accuracy_rate : Rate of correctly performed predictions.
    """
    # Percentage of elements that are equal
    accuracy_rate = (np.sum(y_prediction == y_true) / len(y_true)) * 100
    return accuracy_rate


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


def draw_confusion_matrix(y_true, y_prediction, label):
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    if label == 'test':
        plt.title('Confusion Matrix for Test Dataset Accuracy')
        plt.show()
        return cm
    elif label == 'train':
        plt.title('Confusion Matrix for Training Dataset Accuracy')
        plt.show()
        return cm
    else:
        print('Invalid Confusion Matrix Label')
        return cm




