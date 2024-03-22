import numpy as np
from numpy.linalg import inv


def lasso(X: np.array, y: np.array, lam: float):
    """
    apply lasso regression (L1 regularization) on a given set of data

    :param X: is expected in the regular design matrix format X = (1 x1 x2 ...) size = n x (p+1)
    :param y: measured response vector (is a row vector by default by python, but in mathematical terms, it is a column vector)
    :param lam: hyperparameter lambda of lasso regression
    :return: the lasso regression parameters
    """
    return 0
