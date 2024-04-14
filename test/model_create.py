import random

import numpy as np
import random
import matplotlib.pyplot as plt

from cnn.activation_layer import activation_layer
from cnn.dense_layer import dense_layer
from cnn.cnn import cnn

if __name__ == '__main__':
    cnn = cnn([dense_layer(8, 4), dense_layer(4, 2), dense_layer(2, 1)])
    coef = random.random()
    bias = random.random()
    print(f'coef: {coef}. bias: {bias}')
    for i in range(1000000):
        X = np.random.randn(8, 1)
        Y_hat = cnn.fwd_prop(X)
        Y = coef * np.sum(X).reshape(1, 1) + bias + coef * np.random.randn(1, 1) / 10
        dE_dY = Y_hat - Y
        cnn.bck_prop(dE_dY, 0.0001)

    print(cnn.fwd_prop(np.array([0, 0, 0, 0, 0, 0, 0, 0]).reshape(8, 1)))
    print(coef * 0 + bias)
    print(cnn.fwd_prop(np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape(8, 1)))
    print(coef * 8 + bias)
    print(cnn.fwd_prop(np.array([1, -1, 2, -2, 3, -3, 1, 0]).reshape(8, 1)))
    print(coef * 1 + bias)

    print('foo')
