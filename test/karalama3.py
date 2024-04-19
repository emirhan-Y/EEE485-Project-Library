from data.csv_prepare import csv_prepare
from karalama2 import *
import numpy as np

n = 100            # Number of data instances for each class
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

    all_classifiers_prob_test, all_classifiers_prob_train, y_prediction_test, y_prediction_train = (
        multiclass_logistic_regression(X_train, X_test, y_train, y_test, K_str, iteration, threshold))

    print('** ONE-VS-ALL LOGISTIC REGRESSION CLASSIFIER **', )
    # Find Accuracies
    test_accuracy_multi_class_lr = accuracy(y_prediction_test, y_test)
    print('Test Accuracy of the model using test data: ', test_accuracy_multi_class_lr)
    train_accuracy_multi_class_lr = accuracy(y_prediction_train, y_train)
    print('Train Accuracy of the model using training data: ', train_accuracy_multi_class_lr)

    # Drawing Confusion Matrix of the multiclass classification results
    test_cm = draw_confusion_matrix(y_test, y_prediction_test, 'test')
    train_cm = draw_confusion_matrix(y_train, y_prediction_train, 'train')

else:
    print("Unable to do Logistic Regression for n = {}".format(n))

np.savetxt(r"C:\Users\BERKAN\Desktop\predict\y_predict_test_y.csv", all_classifiers_prob_test, delimiter=',')

