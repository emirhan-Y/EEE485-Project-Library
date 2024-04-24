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
