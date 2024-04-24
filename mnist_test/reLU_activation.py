import numpy as np

class reLU_activation:
    def __init__(self):
        self._X = None
        self._V = None

    def fwd_prop(self, X):
        self._X = X
        self._V = np.clip(X, 0)
        return self._V

    def bck_prop(self, dL_dY, eta):
        self._dY_dV = np.max(0, eta)
        self._dL_dV = np.multiply(dL_dY, self._dY_dV)
        return self._dL_dV
