"""
Linear regression model definitions

methods:
linear_regression(X: np.ndarray, y: np.ndarray, one_column: bool): trains a linear regression model from given data
"""

import numpy as np
from numpy.linalg import inv


def linear_regression(X: np.ndarray, y: np.ndarray, one_column: bool) -> np.ndarray:
    """
    apply linear regression on a given set of data

    :param X:
    is expected in the regular design matrix format X = (1 x1 x2 ...) size = n x (p+1) or X = (x1 x2 ...) size = n x p
    :param y:
    training data response vector
    :param one_column:
    set False if the input design matrix lacks the leading one column to add it in post
    :return: linear regression (the least squares) parameters (intercept, B1, B2, ..., Bp)
    :rtype np.ndarray
    """
    if not one_column:
        X = np.c_[np.ones(len(X)), X]
    A = np.matmul(np.transpose(X), X)
    A_inv = inv(A)
    return np.matmul(np.matmul(A_inv, np.transpose(X)), y)
