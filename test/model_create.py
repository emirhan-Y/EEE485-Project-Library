import numpy as np

from legacy.activation_layer import activation_layer
from cnn.dense_layer import dense_layer
from cnn.cnn import cnn

if __name__ == '__main__':
    cnn = cnn([dense_layer(80, 40), activation_layer('sigmoid'),
               dense_layer(40, 8), activation_layer('sigmoid'),
               dense_layer(8, 2)])
    for i in range(1000000):
        X = np.random.randn(80, 1)
        Y_hat = cnn.fwd_prop(X)
        Y = np.array(
            [7 * (X[0][0] + X[2][0] + X[4][0] + X[6][0]), 15 * (X[1][0] + X[3][0] + X[5][0] + X[7][0])]).reshape(2, 1)
        dE_dY = Y_hat - Y
        cnn.bck_prop(dE_dY, 0.001)

    X = np.zeros((80, 1))
    print(cnn.fwd_prop(X))
    print([7 * (X[0][0] + X[2][0] + X[4][0] + X[6][0]), 15 * (X[1][0] + X[3][0] + X[5][0] + X[7][0])])
    X = np.ones((80, 1))
    print(cnn.fwd_prop(X))
    print([7 * (X[0][0] + X[2][0] + X[4][0] + X[6][0]), 15 * (X[1][0] + X[3][0] + X[5][0] + X[7][0])])
    X = np.random.randn(80, 1)
    print(cnn.fwd_prop(X))
    print([7 * (X[0][0] + X[2][0] + X[4][0] + X[6][0]), 15 * (X[1][0] + X[3][0] + X[5][0] + X[7][0])])

    print('foo')
