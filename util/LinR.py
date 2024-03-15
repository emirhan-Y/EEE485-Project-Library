import numpy as np
from numpy.linalg import inv


def lin_r(X: np.array, y: np.array):
    """
    apply linear regression on a given set of data

    :param X: is expected in the regular design matrix format X = (1_ x_1_ x_2_ ...) size = n x (p+1)
    :param y: measured response vector (is a row vector by default by python, but in mathematical terms, it is a column vector)
    :return: linear regression (the least squares) parameters
    """
    return np.matmul(np.matmul(inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)


def ridge(X: np.array, y: np.array, lam: float):
    """
    apply ridge regression (L2 regularization) on a given set of data

    :param X: is expected in the regular design matrix format X = (1_ x_1_ x_2_ ...) size = n x (p+1)
    :param y: measured response vector (is a row vector by default by python, but in mathematical terms, it is a column vector)
    :param lam: hyperparameter lambda of ridge regression
    :return: the ridge regression parameters
    """

    return np.matmul(np.matmul(inv(np.matmul(np.transpose(X), X) + lam * np.identity(X.shape[1])), np.transpose(X)), y)


def lasso(X: np.array, y: np.array, lam: float):
    """
    apply lasso regression (L1 regularization) on a given set of data

    :param X: is expected in the regular design matrix format X = (1_ x_1_ x_2_ ...) size = n x (p+1)
    :param y: measured response vector (is a row vector by default by python, but in mathematical terms, it is a column vector)
    :param lam: hyperparameter lambda of lasso regression
    :return: the lasso regression parameters
    """
    return np.matmul(np.matmul(inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)
