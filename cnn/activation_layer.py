import numpy as np

class activation_layer:
    def __init__(self, activation_type):
        self._activator, self._activator_derivative = self._activation_interpreter(activation_type)
        self._X = None

    def _activation_interpreter(self, activations_type: str):
        match activations_type.lower():
            case 'sigmoid':
                return lambda v: 1 / (1 + np.exp(-v)), lambda v: np.exp(-v) / ((1 + np.exp(-v)) ** 2)
            case 'relu':
                return lambda v: np.maximum(v, 0), lambda v: (v >= 0).astype(int)
            case 'softmax':
                return lambda v: np.exp(v) / np.sum(np.exp(v)), lambda v: np.exp(v) / np.sum(np.exp(v))
            case _:
                raise RuntimeError(f'Invalid activation function type {activations_type}!')

    def fwd_prop(self, X):
        self._X = X
        return self._activator(X)

    def bck_prop(self, dE_dV, eta):
        dV_dX = self._activator_derivative(self._X)
        return np.multiply(dE_dV, dV_dX)
