import numpy as np
from scipy import signal
from .activation import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax, softmax_derivative


class convolutional_layer:
    def __init__(self, dim_X: tuple, depth_X: int, dim_K: tuple, num_K: int, rng: np.random.default_rng,
                 activation_type: str):
        """
        Convolutional layer class for the convolutional neural networks

        :param tuple dim_X: Dimension of a layer of input. In the case of images, it is the image resolution
        :param depth_X: Input depth. In the case of images it can be 3 to represent the color channels
        :param dim_K: Kernel dimensions: (height, width) of each kernel
        :param num_K: Number of kernel vectors
        :param rng:
        :param activation_type:
        """
        self._activation_type = activation_type.lower()
        if self._activation_type == 'relu':
            std_dev = np.sqrt(2. / (dim_X[0] * dim_X[1] * depth_X))  # He initialization standard deviation
        else:
            std_dev = np.sqrt(1. / (dim_X[0] * dim_X[
                1] * depth_X))  # Xavier initialization standard deviation for Sigmoid or other activations

        self._dim_X = np.array([dim_X[0], dim_X[1]])
        self._dim_K = np.array([dim_K[0], dim_K[1]])
        self._dim_V = self._dim_X - self._dim_K + 1
        self._num_K = num_K
        self._depth_X = depth_X
        self._depth_K = depth_X
        self._depth_V = num_K

        self._K = rng.normal(0, std_dev, size=(self._num_K, self._depth_K, self._dim_K[0], self._dim_K[1]))
        self._B = np.zeros((self._num_K, self._dim_V[0], self._dim_V[1])) + 0.001
        self._phi, self._d_phi = self._activation_interpreter()  # layer activation function and its derivative,
        # as lambda functions

        self._X = None
        self._V = None
        self._Y = None  # layer output vector
        self._K_grad = None
        self._X_grad = None

    def _activation_interpreter(self):
        """
        Returns two anonymous functions:

        1. The activation function of the layer, which takes the induced local field vector, and calculates the layer
            output, according to user's activation function type choice.

        2. The derivative of the activation function of the layer, which takes the induced local field vector, and
        calculates the derivative of the layer output, with respect to the current layer induced field, according to
        user's activation function type choice.

        Returns
        -------
        Layer activation function, and its derivative as anonymous functions.

        Raises
        ------
        RuntimeError if the activations_type is an invalid activation function type.
        """
        match self._activation_type:
            case 'sigmoid':  # generate sigmoid activation function and its derivative
                return sigmoid, sigmoid_derivative
            case 'relu':  # generate relu activation  function and its derivative
                return relu, relu_derivative
            case 'softmax':  # generate softmax activation function and its derivative
                return softmax, softmax_derivative
            case _:  # invalid activator
                raise RuntimeError(f'Invalid activation function type {self._activation_type}!')

    def fwd_prop(self, X):
        self._X = X
        self._V = np.copy(self._B)
        for i in range(self._num_K):
            for j in range(self._depth_K):
                self._V[i] += signal.correlate2d(self._X[j], self._K[i, j], 'valid')
        self._Y = self._phi(self._V)
        return self._Y

    def bck_prop(self, dL_dY, eta, is_last_softmax_with_cross_entropy=False):
        if is_last_softmax_with_cross_entropy:
            # Handle softmax layer differently if flagged
            self._dL_dV = dL_dY  # Directly use dL_dV since it's prepared for softmax
        else:
            if self._activation_type == 'softmax':
                self._dY_dV = self._d_phi(self._Y)
                self._dL_dV = np.dot(self._dY_dV.T, dL_dY)
            else:
                self._dY_dV = self._d_phi(self._V)
                self._dL_dV = np.multiply(dL_dY, self._dY_dV)

        self._K_grad = np.zeros_like(self._K, dtype=np.float64)
        self._X_grad = np.zeros_like(self._X, dtype=np.float64)

        for i in range(self._num_K):
            for j in range(self._depth_K):
                self._K_grad[i, j] = signal.correlate2d(self._X[j], self._dL_dV[i], 'valid')
                self._X_grad[j] += signal.correlate2d(self._dL_dV[i], self._K[i, j], 'full')

        self._K -= eta * self._K_grad
        self._B -= eta * self._dL_dV
        return self._X_grad
