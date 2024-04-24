import numpy as np

class sigmoid_activation:
    def __init__(self):
        self._V = None
        self._Y = None

    def fwd_prop(self, V):
        V = np.clip(V, -50, 50)  # Clipping input to avoid extreme values
        self._V = V
        self._Y = 1 / (1 + np.exp(-V))
        return self._Y

    def bck_prop(self, dL_dY, eta):
        dY_dV = self._Y * (1 - self._Y)
        dL_dV = np.multiply(dL_dY, dY_dV)
        return dL_dV
