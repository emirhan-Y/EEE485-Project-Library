"""Linear regression test script.
Linear regression: TEST: PASSED
"""
import random

import numpy as np

from models import linear_regression
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    length = random.randint(128, 1024)  # random length dataset
    predictors = random.randint(2, 8)  # random predictor count

    real_coefficients = np.array([2 * (random.random() - 0.5) for i in range(predictors + 1)])  # generate the real
    # coefficients randomly
    print(real_coefficients)

    # generate random data
    test_X = np.array([[random.randint(-128, 128) for j in range(predictors)] for i in range(length)])
    test_X = np.c_[np.ones(len(test_X)), test_X]  # add leading one column

    test_y = np.zeros(length)  # generate training labels from the real coefficients
    error = np.random.normal(0, 16, length)
    for i in range(length):
        test_y[i] = np.dot(test_X[i], real_coefficients) + error[i]  # add measurement error too

    our_coefs = linear_regression(test_X, test_y, True)  # our linear regression model results
    sklearn_reg = LinearRegression()
    sklearn_reg.fit(test_X, test_y)
    sklearn_coefs = [sklearn_reg.intercept_] + list(sklearn_reg.coef_[1:])  # sklearn linear regression model results
    print(our_coefs)
    print(sklearn_coefs)

    success = True  # check if our coefficients and sklearn coefficients are close enough
    for i in range(len(real_coefficients)):
        if not (abs(sklearn_coefs[i] * 0.99) < abs(our_coefs[i]) < abs(sklearn_coefs[i] * 1.01)):
            success = False
    if success:
        print('Linear regression, TEST: PASSED')
    else:
        print('Linear regression, TEST: FAILED')
