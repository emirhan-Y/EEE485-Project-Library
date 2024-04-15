import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def add_column_ones_design_matrix(X):
    """
        Adds a column of 1's to the design matrix for the bias term.

        Parameters
        ----------
        X : Design Matrix

        Returns
        -------
        Returns the new design matrix X, which the first column is all 1's.
        """
    # Add column 1 to the design matrix
    ones_column = np.ones((X.shape[0], 1))
    return np.hstack((ones_column, X))


def sigmoid(x):
    """
        Applies sigmoid function for a given input.

        Parameters
        ----------
        x : Input to the sigmoid function.
            For Logistic Regression purposes, apply dot product of data instance and weight vector
            before applying this function.

        Returns
        -------
        Returns a value between [0,1].
    """
    return 1 / (1 + np.exp(-x))


def train(X_train, y_train, weights_init, iteration):
    """
        This function trains the logistic regression model for binary classification.
        Uses Newton-Raphson Method to update the weights.

        Parameters
        ----------
        X_train : Design matrix of the training dataset.
        y_train : Response vector of the training dataset.
        weights_init : Starting weight vector for Newton-Raphson Method
        iteration : Maximum number of Newton-Raphson Method iterations if no convergence.

        Returns
        -------
        w_new : The final updated wieght vector of the Logistic Model.
    """
    w_new = weights_init
    iter_num = 0
    # print(weights_init)

    for i in range(iteration):
        w_old = np.copy(w_new)
        PI = sigmoid(np.dot(X_train, w_old))
        W = np.eye(len(PI))
        np.fill_diagonal(W, PI * (1 - PI))
        # Newton-Raphson Method for updating weights
        XTWX = np.dot(np.dot(X_train.T, W), X_train)

        # Check if XTWX is singular
        if np.linalg.cond(XTWX) < 1 / sys.float_info.epsilon:
            # If XTWX is singular, add regularization to stabilize inversion
            w_new = w_old + np.dot(np.dot(np.linalg.inv(XTWX + 0.01 * np.eye(len(XTWX))), X_train.T), y_train - PI)
        else:
            # Otherwise, proceed without regularization
            w_new = w_old + np.dot(np.dot(np.linalg.inv(XTWX), X_train.T), y_train - PI)

        iter_num += 1
        # e = abs(w_new - w_old)
        # check = np.array_equal(w_new, w_old)
        # print(w_new)

        if all(abs(w_new - w_old) < 10 ** (-8)):
            print('Weights Converged')
            print('Total Iterations Until Convergence: ', iter_num)
            break

    if not np.array_equal(w_new, w_old):
        print('Weights Did Not Converge')
        print('Total Iterations: ', iter_num)
    return w_new


def test(X_test, y_test, weights):
    """
        This function tests the logistic regression model for binary classification.

        Parameters
        ----------
        X_test : Design matrix of the test dataset.
        y_test : Response vector of the test dataset.
        weights : Updated weight vector obtained from training.

        Returns
        -------
        y_probability : Response vector in terms of probabilities.
        y_prediction : Prediction vector for each test data.
                       y_probability is adjusted based on the treshold.
    """
    z = np.dot(X_test, weights)
    y_probability = sigmoid(z)
    y_prediction = y_probability.copy()
    threshold = 0.5
    y_prediction[y_prediction < threshold] = 0
    y_prediction[y_prediction >= threshold] = 1
    accuracy(y_prediction, y_test)
    return y_probability, y_prediction


def accuracy(y_prediction, y_test):
    """
        Calculates the performance of the model by comparing real responses and the predicted responses
        and finds the Accuracy Rate.

        Parameters
        ----------
        y_prediction : Vector of predicted labels of test dataset.
        y_test :  Vector of real labels of test dataset.

        Returns
        -------
        accuracy_rate : Rate of correctly performed predictions for the test dataset.
    """
    # Percentage of elements that are equal
    accuracy_rate = (np.sum(y_prediction == y_test) / len(y_test)) * 100
    print('Accuracy of the model using test data: ', accuracy_rate)
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


def draw_confusion_matrix(y_test, y_prediction):
    """
        Draws the confusion matrix

        Parameters
        ----------
        y_prediction : Vector of predicted labels of test dataset.
        y_test :  Vector of real labels of test dataset.

        Returns
        -------
        cm : Confusion matrix as a numpy array.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_prediction)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    return cm


