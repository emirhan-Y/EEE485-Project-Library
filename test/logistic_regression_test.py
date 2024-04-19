from data.csv_prepare import csv_prepare
from models.log_r import *
import numpy as np

n = 100             # Number of data instances for each class
test_size = 0.2     # Ratio of splitting dataset
iteration = 100
threshold = 0.5
pca_var = 0.9
K_str = ['gokce', 'omer', 'emir']

# Run this first to prepare a csv file:
if csv_prepare(n, K_str):  # csv_prepare returns True/False
    X = np.genfromtxt(r"..\data\excel\data_x.csv", delimiter=',')
    y = np.genfromtxt(r"..\data\excel\data_y.csv", delimiter=',', dtype=str)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_test_train_xy(X, y, pca_var, K_str, test_size)

    # Train model
    all_classifier_weights = one_vs_all_classifier(X_train, y_train, K_str, iteration)

    # Test the model
    print("*** TEST RESULTS ***")
    print("Test Using Test Dataset: ")
    y_probability_test, y_prediction_test = test(X_test, y_test, all_classifier_weights, threshold)

    print("Test Using Training Dataset: ")
    y_probability_train, y_prediction_train = test(X_train, y_train, all_classifier_weights, threshold)

    # Drawing Confusion Matrix of the multiclass classification results
    test_cm = draw_confusion_matrix(y_test, y_prediction_test, 'test')
    train_cm = draw_confusion_matrix(y_train, y_prediction_train, 'train')

else:
    print("Unable to do Logistic Regression for n = {}".format(n))


