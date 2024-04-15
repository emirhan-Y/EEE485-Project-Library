class cnn:
    def __init__(self, layers: list):
        self._layers = layers

    def fwd_prop(self, X):
        current_layer_output = X
        for layer in self._layers:
            current_layer_output = layer.fwd_prop(current_layer_output)
        return current_layer_output

    def bck_prop(self, derivative_of_loss_wrt_final_output, eta):
        current_reverse_derivative = derivative_of_loss_wrt_final_output
        for layer in reversed(self._layers):
            current_reverse_derivative = layer.bck_prop(current_reverse_derivative, eta)
