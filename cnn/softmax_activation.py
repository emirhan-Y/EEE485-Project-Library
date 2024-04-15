import numpy as np

class softmax_activation:
    def __init__(self):
        self._X = None

    def fwd_prop(self, X):
        # Subtract max for numerical stability (shift each row by its max)
        self._X = X
        shift_x = X - np.max(X)
        exp = np.exp(shift_x)
        self._V = exp / np.sum(exp)
        return self._V

    def bck_prop(self, dE_dV, eta):
        s = self._V.reshape(-1, 1)
        jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)

        # For each class, compute the derivative of the loss with respect to each class score
        dE_dX = np.dot(jacobian_matrix, dE_dV)
        return dE_dX
