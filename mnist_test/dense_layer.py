import numpy as np


def sigmoid(v):
    v = np.clip(v, -50, 50)  # Clipping input to avoid extreme values
    y = 1 / (1 + np.exp(-v))
    return y


def sigmoid_derivative(v):
    y = sigmoid(v)
    dY_dV = y * (1 - y)
    return dY_dV


def relu(v):
    y = np.maximum(v, 0)
    return y


def relu_derivative(v):
    dY_dV = (v > 0).astype(int)
    return dY_dV


def softmax(v):
    shift_v = v - np.max(v)
    exp = np.exp(shift_v)
    y = exp / np.sum(exp)
    return y


def softmax_derivative(v):
    s = v.reshape(-1, 1)
    dY_dV = np.diagflat(s) - np.dot(s, s.T)
    return dY_dV


class dense_layer:
    def __init__(self, X_size: int, Y_size: int, rng: np.random.default_rng, activation_type: str):
        self._activation_type = activation_type.lower()
        self._X = None  # layer input vector
        if self._activation_type == 'relu':
            std_dev = np.sqrt(2. / X_size)  # He initialization standard deviation
        else:
            std_dev = np.sqrt(1. / X_size)  # Xavier initialization standard deviation for Sigmoid or other activations

        self._W = rng.normal(0, std_dev, size=(Y_size, X_size))  # layer weight matrix, He
        self._B = np.zeros((Y_size, 1)) + 0.001  # layer bias vector, small
        self._phi, self._d_phi = self._activation_interpreter()  # layer activation function and its derivative,
        # as lambda functions
        self._V = None  # layer induced field vector
        self._Y = None  # layer output vector

        self._dY_dV = None  # layer output gradient wrt induced field vector
        self._dL_dV = None  # loss gradient wrt induced field vector
        self._dL_dW = None  # loss derivative wrt layer weights matrix
        self._dL_dB = None  # loss derivative wrt layer weights vector
        self._dL_dX = None  # loss derivative wrt layer input (used in backpropagation)

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
        """
        Forward propagation of the layer, returns the layer output vector, given an input vector

        Parameters
        ----------
        X : np.array
            The layer input vector

        Returns
        -------
        self._Y: np.array
            The layer output vector
        """
        self._X = X
        self._V = np.dot(self._W, self._X) + self._B
        self._Y = self._phi(self._V)
        return self._Y

    def bck_prop(self, dL_dY, eta, is_last_softmax_with_cross_entropy=False):
        """
        Backpropagation of the layer, returns the derivative of the loss function with respect to layer input vector,
        given the derivative of the loss function with respect to layer output vector. Updates the layer parameters
        using the learning rate, eta.

        Parameters
        ----------
        dL_dY : np.array
            The derivative of the loss function with respect to layer output vector
        eta: float
            The learning rate
        is_last_softmax_with_cross_entropy: bool
            Used for shortcutting the last layer, if it is a softmax layer with cross entropy loss

        Returns
        -------
        self._dL_dX: np.array
            The derivative of the loss function with respect to layer input vector
        """
        # derivative of layer output w.r.t. induced field
        # derivative of the loss w.r.t. the induced field vector
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

        # gradient w.r.t. weights and biases
        self._dL_dW = np.multiply(self._dL_dV, self._X.T)
        self._dL_dB = self._dL_dV
        # the gradient w.r.t. the input (for backpropagation to previous layers)
        self._dL_dX = np.dot(self._W.T, self._dL_dV)
        # update weights and biases
        self._W -= eta * self._dL_dW
        self._B -= eta * self._dL_dB
        return self._dL_dX

    def get_W(self):
        return self._W

    def get_B(self):
        return self._B
