import numpy as np


class max_pooling_layer:
    def __init__(self, dim_X: tuple, pooling_dim: tuple):
        self._dim_X = dim_X
        self._pooling_dim = pooling_dim
        self._dim_Y = ((dim_X[0] - 1) // pooling_dim[0] + 1, (dim_X[1] - 1) // pooling_dim[1] + 1)

        self._activation_type = 'none'

    def fwd_prop(self, X):
        self.X = X
        self._correct_X_shape = (
            self.X.shape[0], self._dim_Y[0] * self._pooling_dim[0], self._dim_Y[1] * self._pooling_dim[1])

        # Create a larger matrix filled with -inf
        self._expanded_X = np.full(self._correct_X_shape, -np.inf)

        # Copy the content of matrix X into the top-left corner of the expanded matrix
        self._expanded_X[:, :self._dim_X[0], :self._dim_X[1]] = X

        self._expanded_X = self._expanded_X.reshape(self.X.shape[0], self._expanded_X.shape[1] // self._pooling_dim[0],
                                                    self._pooling_dim[0],
                                                    self._expanded_X.shape[2] // self._pooling_dim[1],
                                                    self._pooling_dim[1])
        self._expanded_X = self._expanded_X.transpose(0, 1, 3, 2, 4)
        self._expanded_X = self._expanded_X.reshape(self.X.shape[0], -1, self._pooling_dim[0], self._pooling_dim[1])

        # Apply max pooling along specified dimensions
        pooled_output = np.max(self._expanded_X, axis=(2, 3))
        self._max_indexes = np.argmax(self._expanded_X.reshape(self._expanded_X.shape[0], self._expanded_X.shape[1], -1), axis=2)

        return pooled_output.reshape(self.X.shape[0], self._dim_Y[0], self._dim_Y[1])

    def bck_prop(self, dL_dY, eta):
        dY_dX = np.zeros((self._expanded_X.shape[0], self._expanded_X.shape[1], self._expanded_X.shape[2] * self._expanded_X.shape[3]))

        dY_dX[np.arange(dY_dX.shape[0])[:, None], np.arange(self._max_indexes.shape[1]), self._max_indexes] = 1

        dL_dY = dL_dY.reshape(2, dL_dY.shape[1] * dL_dY.shape[2], 1)

        dL_dX = np.multiply(dL_dY, dY_dX)

        dL_dX = dL_dX.reshape(dY_dX.shape[0], self._dim_Y[0], self._dim_Y[1], self._pooling_dim[0], self._pooling_dim[1])

        dL_dX = dL_dX.transpose(0, 1, 3, 2, 4)

        dL_dX = dL_dX.reshape(dY_dX.shape[0], self._correct_X_shape[1], self._correct_X_shape[2])

        return dL_dX[:, :self._dim_X[0], :self._dim_X[1]]
