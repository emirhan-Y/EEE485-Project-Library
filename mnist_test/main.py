import numpy as np

from dense_layer import dense_layer


def test_relu():
    rng = np.random.default_rng(42)
    activation = dense_layer(X_size=1, Y_size=1, rng=rng, activation_type='relu')
    test_inputs = np.array([-1, 0, 2]).reshape(-1, 1)  # Testing with negative, zero, and positive inputs

    for input_val in test_inputs:
        output = activation.fwd_prop(input_val.reshape(-1, 1))
        expected_output = np.maximum(np.dot(activation._W, input_val) + activation._B, 0)
        assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"


def test_sigmoid():
    rng = np.random.default_rng(42)
    activation = dense_layer(X_size=1, Y_size=1, rng=rng, activation_type='sigmoid')
    test_inputs = np.array([-1, 0, 1]).reshape(-1, 1)  # Cover a range of typical inputs

    for input_val in test_inputs:
        output = activation.fwd_prop(input_val.reshape(-1, 1))
        z = np.dot(activation._W, input_val) + activation._B
        expected_output = 1 / (1 + np.exp(-z))
        assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"


def test_softmax():
    rng = np.random.default_rng(42)
    activation = dense_layer(X_size=3, Y_size=3, rng=rng, activation_type='softmax')
    test_input = np.array([[1], [2], [3]])  # A column vector for single-instance input

    output = activation.fwd_prop(test_input)
    z = np.dot(activation._W, test_input) + activation._B
    shift_z = z - np.max(z)
    exp_z = np.exp(shift_z)
    expected_output = exp_z / np.sum(exp_z)
    assert np.allclose(output, expected_output,
                       atol=1e-7), f"Expected {expected_output.flatten()}, but got {output.flatten()}"


test_softmax()
test_sigmoid()
test_relu()
