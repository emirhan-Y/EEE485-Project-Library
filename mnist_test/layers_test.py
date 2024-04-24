import numpy as np
from dense_layer import dense_layer

rng = np.random.default_rng(1337)
sigmoid_dense = dense_layer(2, 1, rng, 'SIGMOID')
sigmoid_dense._W = np.array([[1, 1]])
sigmoid_dense._B = np.array([[0]])
test_X = np.array([[-2, -1], [0, 1], [2, 3]])

for X in test_X:
    X = X.reshape(2, -1)
    Y = sigmoid_dense.fwd_prop(X)
    print(X, Y)
