"""KNN test script.
KNN: TEST: PASSED
"""

import numpy as np

from models import knn
from util import log

if __name__ == "__main__":
    # knn test
    log.DEBUG = True  # enable log debug messages
    log.d('knn', 'knn data loading')
    knn_0 = knn(data_points=[[0, 3, 0],  # load the data points
                             [2, 0, 0],
                             [0, 1, 3],
                             [0, 1, 2],
                             [-1, 0, 1],
                             [1, 1, 1]],
                labels=np.array(['red',  # load the labels corresponding to the points
                                 'red',
                                 'red',
                                 'green',
                                 'green',
                                 'red']))
    log.d('knn', 'knn data passed requirements')
    print(knn_0.test_point([0, 0, 0], 1))  # test several points, checked manually afterward
    print(knn_0.test_point([0, 0, 0], 3))
    print(knn_0.test_point([0, 1, 1], 1))
    print(knn_0.test_point([0, 1, 1], 2))
    print(knn_0.test_point([0, 1, 1], 3))
    print(knn_0.test_point([0, 1, 1], 4))
    print(knn_0.test_point([0, 1, 1], 5))
    print(knn_0.test_point([0, 1, 1], 6))
