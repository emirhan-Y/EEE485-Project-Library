import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

import util.LinR as LinR
import util.LogR as LogR

if __name__ == "__main__":
    test_X = np.array([[1, 1],
                       [1, 2],
                       [1, 3],
                       [1, 4],
                       [1, 5]])
    test_y = np.array([0,
                       2,
                       4,
                       6,
                       8])
    B_lin = LinR.lin_r(test_X, test_y)
    B_ridge = LinR.ridge(test_X, test_y, 100)

    y = np.array([0,
                  2,
                  4,
                  6,
                  8])
    X = np.array([1,
                  2,
                  3,
                  4,
                  5]).reshape(-1, 1)
    clf = Ridge(alpha=100.0)
    clf.fit(X, y)

    b_re = [clf.intercept_, clf.coef_]

    print(B_lin, B_ridge, b_re, LogR.log_r())

    # test plot
    x = np.linspace(0, 10, 11)
    y1 = clf.intercept_ + clf.coef_ * x
    y2 = B_ridge[1] * x + B_ridge[0]

    plt.scatter(X, y)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
