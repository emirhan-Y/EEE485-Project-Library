import numpy as np

def activation_interpreter(activations_type: str):
    match
    return
class dense_layer:
    def __init__(self, X_size, Y_size, activation_type):
        self._W = np.random.randn(Y_size, X_size)
        self._B = np.random.randn(Y_size, 1)
        self._X = None
        self._phi, self._d_phi = activation_interpreter(activation_type)

    def fwd_prop(self, X):
        self._X = X
        return np.dot(self._W, self._X) + self._B

    def bck_prop(self, dE_dY, eta):
        dE_dW = np.dot(dE_dY, self._X.T)
        self._W -= eta * dE_dW
        self._B -= eta * dE_dY
        dE_dX = np.dot(self._W.T, dE_dY)
        return dE_dX


