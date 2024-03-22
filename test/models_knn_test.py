"""
KNN test script.
================
* KNN: TEST: PASSED
"""
import random
import time

import numpy as np

from models import knn
from util import log
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    logger = log(True, True)
    logger.debug_message('knn test', 'knn data generation')
    predictor_count = 64
    data_points_count = 262140
    data_points = np.array([[random.randint(0, 255) for j in range(predictor_count)] for i in range(data_points_count)])
    labels = np.array([random.randint(0, 31) for i in range(data_points_count)])
    logger.debug_message('knn', 'knn data generation complete')
    logger.debug_message('knn', 'knn data loading')
    knn_0 = knn(data_points=data_points, labels=labels)
    neigh = KNeighborsClassifier(n_neighbors=127)
    neigh.fit(data_points, labels)
    logger.debug_message('knn', 'knn data passed requirements')

    logger.debug_message('knn', 'knn test begins')
    success1 = True
    TEST_POINT_COUNT = 256
    random_test_points = np.array(
        [[random.randint(0, 255) for j in range(predictor_count)] for i in range(TEST_POINT_COUNT)])
    for i in range(TEST_POINT_COUNT):
        logger.debug_message('point X under test', str(random_test_points[i]))
        t1 = time.time()
        y1 = knn_0.test_point(random_test_points[i], 127)
        t2 = time.time()
        y2 = neigh.predict([random_test_points[i]])[0]
        t3 = time.time()
        logger.debug_message('result', 'our knn: ' + str(y1) + '; sklearn knn: ' + str(y2) + '. our knn time: ' +
                             str(t2 - t1) + '; sklearn knn time: ' + str(t3 - t2))

        if y2 not in y1:
            success1 = False
            break

    if success1:
        logger.debug_message('knn', 'TEST: PASSED')
    else:
        logger.debug_message('knn', 'TEST: FAILED')

    logger.close()
