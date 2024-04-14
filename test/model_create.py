import numpy as np

from dense_layer import activation_interpreter

if __name__ == '__main__:':
    # D = dense_layer(10, 1, 'sigmoid')
    X = np.linspace(-5, 5, 11)
    phi = lambda v: 1 / (1 + np.exp(-v))

    w1 = phi(X)
    print('joe')
    print(w1)
    print('mama')
