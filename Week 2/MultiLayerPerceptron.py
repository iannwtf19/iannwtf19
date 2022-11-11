from Layer import Layer


class MultiLayerPerceptron:

    def __init__(self, layers):
        self.layers = layers

    def forward_step(self, x, target):
        inputs = x
        for layer in self.layers[:-1]:
            layer(inputs)
            inputs = layer.activation
        self.layers[-1](inputs, target)

    def backpropagation(self, learning_rate=0.01):
        self.layers[-1].backward_step(learning_rate)
        next_layer = self.layers[-1]
        for layer in reversed(self.layers[:-1]):
            layer.backward_step(learning_rate, next_layer.error_signal, next_layer.weight_matrix)
