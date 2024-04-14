class conversion_layer:
    def __init__(self, vector_length):
        self._converter =

    def _activation_interpreter(self, activations_type: str):
        match activations_type.lower():
            case 'sigmoid':
                return lambda v: 1 / (1 + np.exp(-v)), lambda v: np.exp(-v) / ((1 + np.exp(-v)) ** 2)
            case 'relu':
                return lambda v: np.maximum(v, 0), lambda v: np.exp(-v) / (v >= 0).astype(int)
            case 'softmax':
                return lambda v: np.exp(v) / np.sum(np.exp(v)), lambda v: np.exp(v) / np.sum(np.exp(v))
            case _:
                raise RuntimeError(f'Invalid activation function type {activations_type}!')