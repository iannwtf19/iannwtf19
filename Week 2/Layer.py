import numpy as np

relu = np.vectorize(lambda x: x if x > 0 else 0)


class Layer:

    def __init__(self, n_units, input_units):
        self.n_units = n_units  # number of perceptrons in this layer
        self.input_units = input_units  # number of perceptrons in previous layer
        # create an initial bias vector of zeros
        self.bias_vector = np.ones((1, n_units))
        print("bias vector: ")
        print(self.bias_vector)
        # create initial weight matrix with random values
        self.weight_matrix = np.random.random((input_units, n_units))
        print("weight matrix: ")
        print(self.weight_matrix)
        # put biases into the weight matrix as the last column
        self.weights_and_biases = np.append(self.weight_matrix, self.bias_vector, axis=0)
        print("weights and biases")
        print(self.weights_and_biases)
        # create uninitialized pre-activation, activation and input vectors
        self.pre_activation = np.empty((1, self.n_units))
        self.activation = np.empty((1, self.n_units))
        self.input_vector = np.empty((1, self.input_units))

    def forward_step(self):
        self.pre_activation = self.input_vector @ self.weights_and_biases
        print("pre_activation:")
        print(self.pre_activation)
        self.activation = relu(self.pre_activation)
        print("activation:")
        print(self.activation)


layer = Layer(3, 2)
layer.input_vector = np.array([[-3, 2, 1]])
layer.forward_step()
