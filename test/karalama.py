import numpy as np
from analysis import pca

X = np.genfromtxt(r"C:\Users\BERKAN\Desktop\Data_Project\EEE485-Project-Library\data\excel\data_x.csv", delimiter=',')
data1 = X[0]
print(X.shape)
print("Normal data 1: ", data1)

pca_var = 0.9
principal_component_analysis = pca(X, pca_var)
principal_component_analysis.analyze()
X_pca = principal_component_analysis.get_new_dataset()

print(X_pca.shape)
print("PCA data 1: ", X_pca[0])

data1_2d = np.expand_dims(data1, axis=0)
X_single_pca = principal_component_analysis.reform_data(data1_2d)

print(X_single_pca.shape)
print("X single pca: ", X_single_pca)

print("Equality check: ", np.array_equal(X_single_pca.flatten(), X_pca[0]))


# Bunu sil
def multiclass_logistic_regression(x_train, x_test, y_train, y_test, k_str, iteration, threshold):
    """
    Applies all binary logistic regression classifiers and obtains each models weights
    Returns all model weights as an array.
    """
    # Initialize array
    all_classifiers_prob_test = np.zeros((y_test.shape[0], len(k_str)))
    all_classifiers_prob_train = np.zeros((y_train.shape[0], len(k_str)))
    all_classifier_weights = np.zeros((len(k_str), x_train.shape[1]))

    for i in range(len(k_str)):
        # Convert elements in y to 0 if they are not equal to i
        y_test_binary = np.where(y_test != i, 0, 1)
        y_train_binary = np.where(y_train != i, 0, 1)

        # Training the ith classifier
        print("CLASSIFIER", i)
        weights = binary_logistic_train(x_train, y_train_binary, iteration)

        """weights = evaluate_error_per_iteration_binary_train(x_train, y_train_binary, iteration)"""
        # Save all model weights for later use.

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



