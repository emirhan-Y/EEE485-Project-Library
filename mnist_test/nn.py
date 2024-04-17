class nn:
    def __init__(self, layers: list, loss_type: str):
        self._layers = layers
        self._loss_type = loss_type.lower()

    def fwd_prop(self, X):
        current_layer_output = X
        for layer in self._layers:
            current_layer_output = layer.fwd_prop(current_layer_output)
        return current_layer_output

    def bck_prop(self, dL_dY, eta):
        current_derivative = dL_dY
        for layer in reversed(self._layers):
            if layer == self._layers[-1] and layer._activation_type == 'softmax' and self._loss_type == 'cross_entropy':
                # Assume dL_dY for softmax is actually dL_dV
                current_derivative = layer.bck_prop(current_derivative, eta, is_last_softmax_with_cross_entropy=True)
            else:
                current_derivative = layer.bck_prop(current_derivative, eta)
