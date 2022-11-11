from Layer import Layer


class MultiLayerPerceptron:

    def __init__(self):
        self.hidden_layer = Layer(10, 1)
        self.output_layer = Layer(1, 10)

    def forward_step(self, x, target):
        self.hidden_layer(x)
        self.output_layer(self.hidden_layer.activation, target)

    def backpropagation(self, learning_rate=0.01):
        self.output_layer.backward_step(learning_rate)
        self.hidden_layer.backward_step(learning_rate, self.output_layer.error_signal, self.output_layer.weight_matrix)
