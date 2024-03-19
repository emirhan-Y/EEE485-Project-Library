import numpy as np

from models import knn
from util import log

if __name__ == "__main__":
    # knn test
    log.DEBUG = True
    log.d('knn', 'knn data loading')
    knn_0 = knn(data_points=[[0, 3, 0],
                             [2, 0, 0],
                             [0, 1, 3],
                             [0, 1, 2],
                             [-1, 0, 1],
                             [1, 1, 1]],
                labels=np.array(['red',
                                 'red',
                                 'red',
                                 'green',
                                 'green',
                                 'red']))
    log.d('knn', 'knn data passed requirements')
    print(knn_0.test_point([0, 0, 0], 1))
    print(knn_0.test_point([0, 0, 0], 3))
    print(knn_0.test_point([0, 1, 1], 1))
    print(knn_0.test_point([0, 1, 1], 2))
    print(knn_0.test_point([0, 1, 1], 3))
    print(knn_0.test_point([0, 1, 1], 4))
    print(knn_0.test_point([0, 1, 1], 5))
    print(knn_0.test_point([0, 1, 1], 6))

    """
    #hullo
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
    B_ridge = LinR.ridge(test_X, test_y, 1)

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
    clf = Ridge(alpha=1.0)
    clf.fit(X, y)

    b_re = [clf.intercept_, clf.coef_]

    print(B_lin, [np.mean(y), B_ridge], b_re, LogR.log_r())

    # test plot
    x = np.linspace(0, 10, 11)
    y1 = clf.intercept_ + clf.coef_ * x
    y2 = B_ridge * x - np.mean(y)

    plt.scatter(X, y)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
    """
