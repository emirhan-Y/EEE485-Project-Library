import numpy as np
from scipy import signal


class convolutional_layer:
    def __init__(self, dim_X: tuple, depth_X: int, dim_K: tuple, num_K: int, rng):
        self._rng = rng

        self._dim_X = np.array([dim_X[0], dim_X[1]])
        self._dim_K = np.array([dim_K[0], dim_K[1]])
        self._dim_V = self._dim_X - self._dim_K + 1
        self._num_K = num_K
        self._depth_X = depth_X
        self._depth_K = depth_X
        self._depth_V = num_K

        self._K = self._rng.standard_normal(size=(self._num_K, self._depth_K, self._dim_K[0], self._dim_K[1]),
                                            dtype=np.float64) * 1e-3
        self._B = self._rng.standard_normal(size=(self._num_K, self._dim_V[0], self._dim_V[1]),
                                            dtype=np.float64) * 1e-3

        self._X = None
        self._V = None
        self._K_grad = None
        self._X_grad = None

    def fwd_prop(self, X):
        self._X = X
        self._V = np.copy(self._B)
        for i in range(self._num_K):
            for j in range(self._depth_K):
                self._V += signal.correlate2d(self._X[j], self._K[i, j], 'valid')
        return self._V

    def bck_prop(self, dE_dV, eta):
        self._K_grad = np.zeros_like(self._K, dtype=np.float64)
        self._X_grad = np.zeros_like(self._X, dtype=np.float64)

        for i in range(self._num_K):
            for j in range(self._depth_K):
                self._K_grad[i, j] = signal.correlate2d(self._X[j], dE_dV[i], 'valid')
                self._X_grad[j] += signal.correlate2d(dE_dV[i], self._K[i, j], 'full')

        self._K -= eta * self._K_grad
        self._B -= eta * dE_dV
        return self._X_grad
