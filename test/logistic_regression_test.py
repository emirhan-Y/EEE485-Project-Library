from sklearn.model_selection import train_test_split
from data.csv_prepare import csv_prepare
from models.log_r import *
from analysis import pca
import numpy as np

# y = np.genfromtxt(r"....\data\excel\data_y.csv", delimiter=',', dtype=str)

n = 100            # Number of data instances
test_size = 0.2     # Ratio of splitting dataset
iteration = 100
K_str = ['gokce', 'omer','emir']

# Run this first to prepare a csv file:
if csv_prepare(n, K_str):  # csv_prepare returns True/False
    X = np.genfromtxt(r"..\data\excel\data_x.csv", delimiter=',')
    y = np.genfromtxt(r"..\data\excel\data_y.csv", delimiter=',', dtype=str)

    # Enter PCA
    principal_component_analysis = pca(X, 0.9)
    principal_component_analysis.analyze()
    X = principal_component_analysis.get_new_dataset()

    p = X.shape[1]

    # Replace class string names with integers in the order of K_str
    string_to_index = {string: i for i, string in enumerate(K_str)}
    y = np.array([string_to_index[string] for string in y])

    # Add column 1 to the design matrix
    X = add_column_ones_design_matrix(X)
    weights_init = np.zeros((p + 1))

    # Split Data into Training and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    all_classifier_prob = np.zeros((y_test.shape[0], len(K_str)))

    for i in range(len(K_str)):
        # Convert elements in y to 0 if they are not equal to i
        y_test_binary = np.where(y_test != i, 0, 1)
        y_train_binary = np.where(y_train != i, 0, 1)

        # Training
        print("CLASSIFIER", i)
        weights = train(X_train, y_train_binary, weights_init, iteration)

        y_probability, y_prediction = test(X_test, y_test_binary, weights)
        all_classifier_prob[:, i] = y_probability

    y_prediction = np.argmax(all_classifier_prob, axis=1)

    # Find Accuracy
    print('***** ONE-VS-ALL LOGISTIC REGRESSION CLASSIFIER *****', )
    accuracy(y_prediction, y_test)

    # Drawing Confusion Matrix
    cm = draw_confusion_matrix(y_test, y_prediction)

else:
    print("Unable to do Logistic Regression for n = {}".format(n))



