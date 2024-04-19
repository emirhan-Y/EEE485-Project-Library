from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import pca
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


def prepare_test_train_xy(X, y, pca_var, K_str, test_size):
    """
        1) Applies PCA to the design matrix.
        2) Replace class string names with integers in the order of K_str.
        3) Adds columns of 1s to the design matrix after PCA.
        4) Splits X and y into train and test datasets based on the test size.

        Parameters
        ----------
        X : Initial unmodified design matrix.
        y : Initial unmodified response matrix. classes are writen as string class names.
        pca_var : Variance parameter for PCA
        test_size : Ratio of the dataset to allocate to test.
        K_str : list of class names (string).

        Returns
        -------
        X_train, X_test, y_train, y_test : Prepared test and train datasets.
    """
    # Enter PCA
    principal_component_analysis = pca(X, pca_var)
    principal_component_analysis.analyze()
    X = principal_component_analysis.get_new_dataset()

    # Replace class string names with integers in the order of K_str
    string_to_index = {string: i for i, string in enumerate(K_str)}
    y = np.array([string_to_index[string] for string in y])

    # Add column 1 to the design matrix
    X = add_column_ones_design_matrix(X)

    # Split Data into Training and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test


def binary_logistic_train(X_train, y_train, iteration):
    """
        This function trains the logistic regression model for binary classification.
        Uses Newton-Raphson Method to update the weights.

        Parameters
        ----------
        X_train : Design matrix of the training dataset.
        y_train : Response vector of the training dataset.
        iteration : Maximum number of Newton-Raphson Method iterations if no convergence.

        Returns
        -------
        w_new : The final updated wieght vector of the Logistic Model.
    """
    # Initialize model weights
    weights_init = np.zeros(X_train.shape[1])
    w_new = weights_init
    iter_num = 0

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


def multiclass_logistic_regression(x_train, x_test, y_train, y_test, k_str, iteration, threshold):
    """
    Applies all binary logistic regression classifiers and obtains each models weights
    Returns all model weights as an array.
    """
    # Initialize array
    all_classifiers_prob_test = np.zeros((y_test.shape[0], len(k_str)))
    all_classifiers_prob_train = np.zeros((y_train.shape[0], len(k_str)))

    for i in range(len(k_str)):
        # Convert elements in y to 0 if they are not equal to i
        y_test_binary = np.where(y_test != i, 0, 1)
        y_train_binary = np.where(y_train != i, 0, 1)

        # Training the ith classifier
        print("CLASSIFIER", i)
        weights = binary_logistic_train(x_train, y_train_binary, iteration)

        """weights = evaluate_error_per_iteration_binary_train(x_train, y_train_binary, iteration)"""

        # Test the model for classifier i, using test and training data
        y_probability_test, y_prediction_test = test(x_test, weights, threshold)
        y_probability_train, y_prediction_train = test(x_train, weights, threshold)

        all_classifiers_prob_test[:, i] = y_probability_test
        all_classifiers_prob_train[:, i] = y_probability_train

        print('Accuracy of the model using test data: ', accuracy(y_prediction_test, y_test_binary))
        print('Accuracy of the model using train data: ', accuracy(y_prediction_train, y_train_binary))

    # A vector (y_prediction) storing the predicted classes (as integers) for each data
    y_prediction_test = np.argmax(all_classifiers_prob_test, axis=1)
    y_prediction_train = np.argmax(all_classifiers_prob_train, axis=1)

    return all_classifiers_prob_test, all_classifiers_prob_train, y_prediction_test, y_prediction_train


def test(x, weights, threshold):
    """
        This function tests the logistic regression model for binary classification.

        Parameters
        ----------
        x : Design matrix of the test dataset.
        weights : Updated weight vector obtained from training.
        threshold : Prediction boundary threshold.

        Returns
        -------
        y_probability : Response vector in terms of probabilities for binary logistic regression.
        y_prediction : Prediction vector for each test data. (y_probability is adjusted based on the threshold.)
    """
    z = np.dot(x, weights)
    y_probability = sigmoid(z)
    y_prediction = y_probability.copy()
    y_prediction[y_prediction < threshold] = 0
    y_prediction[y_prediction >= threshold] = 1
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
    elif label == 'train':
        plt.title('Confusion Matrix for Training Dataset Accuracy')
    else:
        print('Invalid label')
        return cm
    plt.show()
    return cm


def evaluate_threshold(x_train, x_test, y_train, y_test, k_str, iteration):
    """Don't need it for the project"""
    lst1 = [0.01 * i for i in range(1, 10)]
    lst2 = [0.1 * i for i in range(1, 9)]
    lst3 = [0.01 * i for i in range(90, 100)]

    threshold_arr = lst1 + lst2 + lst3

    for threshold in threshold_arr:
        print("TRAINING THRESHOLD: {}\n".format(threshold))
        all_classifiers_prob_test, all_classifiers_prob_train, y_prediction_test, y_prediction_train = (
            multiclass_logistic_regression(x_train, x_test, y_train, y_test, k_str, iteration, threshold))
        # Find Accuracies
        test_accuracy_multi_class_lr = accuracy(y_prediction_test, y_test)
        print('Test Accuracy of the model using test data: ', test_accuracy_multi_class_lr)
        train_accuracy_multi_class_lr = accuracy(y_prediction_train, y_train)
        print('Train Accuracy of the model using training data: ', train_accuracy_multi_class_lr)
        # Drawing Confusion Matrix
        test_cm = draw_confusion_matrix(y_test, y_prediction_test, 'test')
        train_cm = draw_confusion_matrix(y_train, y_prediction_train, 'train')
    return 1


def evaluate_error_per_iteration_binary_train(X_train, y_train, iteration):
    """
        Trains the binary classification logistic regression in the same way as binary_logistic_train() function.
        In addition, calculates the likelihood using the current weights at each iteration.
        At last plots the likelihood vs iteration plot.
        * When going to use this, change the training function in multiclass_logistic_regression()
            from binary_logistic_train() to evaluate_error_per_iteration_binary_train()
        Parameters
        ----------
        X_train : Design matrix of the training dataset.
        y_train : Response vector of the training dataset.
        iteration : Maximum number of Newton-Raphson Method iterations if no convergence.

        Returns
        -------
        w_new : The final updated wieght vector of the Logistic Model.
    """
    # Initializations
    weights_init = np.zeros(X_train.shape[1])
    w_new = weights_init

    iter_num = 0
    likelihood = np.zeros(iteration + 1)
    likelihood[iter_num] = log_likelihood(X_train, y_train, weights_init)

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
        likelihood[iter_num] = log_likelihood(X_train, y_train, w_new)

        if all(abs(w_new - w_old) < 10 ** (-8)):
            print('Weights Converged')
            print('Total Iterations Until Convergence: ', iter_num)
            break

    if not np.array_equal(w_new, w_old):
        print('Weights Did Not Converge')
        print('Total Iterations: ', iter_num)

    # Plot the likelihood vs iteration plot
    plt.plot(np.arange(iteration + 1), likelihood, label='Log-Likelihood', marker='*', linestyle='None')
    plt.xlabel('Iteration')
    plt.ylabel('Likelihood')
    plt.title('Log-Likelihood vs Iterations')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    return w_new


def log_likelihood(x, y, weight):
    """
        Calculates the likelihood of the data given the weights
        Parameters
        ----------
        x, y, weight : Design matrix, response matrix, weights
        Returns
        -------
        likelihood : computed likelihood value
    """
    n = x.shape[0]
    likelihood = 0
    for i in range(n):
        wx = np.dot(x[i], weight)
        likelihood_i = np.dot(y[i], wx) - np.log10(1 + np.exp(wx))
        likelihood = likelihood + likelihood_i
    return likelihood


def test_single_data():

    return 1