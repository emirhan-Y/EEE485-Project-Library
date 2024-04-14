import numpy as np


class dense_layer:
    def __init__(self, X_size, V_size):
        self._W = np.random.randn(V_size, X_size)
        self._B = np.random.randn(V_size, 1)
        self._X = None

    def fwd_prop(self, X):
        self._X = X
        return np.dot(self._W, self._X) + self._B

    def bck_prop(self, dE_dV, eta):
        dE_dW = np.multiply(dE_dV, self._X.T)
        dE_dX = np.dot(self._W.T, dE_dV)
        self._W -= eta * dE_dW
        self._B -= eta * dE_dV
        return dE_dX

    def get_W(self):
        return self._W

    def get_B(self):
        return self._B
