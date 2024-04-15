class flatten_layer:
    def __init__(self, dim_X, dim_V):
        self._dim_X = dim_X
        self._dim_V = dim_V

    def fwd_prop(self, X):
        return X.reshape(self._dim_V)

    def bck_prop(self, dE_dV, eta):
        return dE_dV.reshape(self._dim_X)
