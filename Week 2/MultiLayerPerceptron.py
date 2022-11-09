import numpy as np
from Layer import Layer


class MultiLayerPerceptron:

    def __init__(self):
        self.hidden_layer = Layer(10, 1)
        self.output_layer = Layer(1, 10)

    def forward_step(self, x):
        self.hidden_layer.input_vector = np.ndarray(x)
        self.hidden_layer.forward_step()

        self.output_layer.input_vector = self.hidden_layer.activation
        self.output_layer.forward_step()

    def backpropagation(self):
        return
# bla
