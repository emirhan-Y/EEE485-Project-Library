import numpy as np

class sigmoid_activation:
    def __init__(self):
        self._X = None
        self._V = None

    def fwd_prop(self, X):
        X = np.clip(X, -50, 50)  # Clipping input to avoid extreme values
        self._X = X
        self._V = 1 / (1 + np.exp(-X))
        return self._V

    def bck_prop(self, dE_dV, eta):
        dV_dX = self._V * (1 - self._V)
        dE_dX = np.multiply(dE_dV, dV_dX)
        return dE_dX
