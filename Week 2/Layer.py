import numpy as np

relu = np.vectorize(lambda x: 0 if x < 0 else x)
d_relu = np.vectorize(lambda x: 0 if (x < 0) else 1)  # what if x = 0?


class Layer:

    def __init__(self, n_units, input_units):
        self.loss = None
        self.error_signal = None
        self.gradient = None
        self.n_units = n_units  # number of perceptrons in this layer
        self.input_units = input_units  # number of perceptrons in previous layer

        # create an initial bias vector of zeros
        self.bias_vector = np.zeros((n_units, 1))
        print("bias vector: ")
        print(self.bias_vector)

        # create initial weight matrix with random values
        self.weight_matrix = np.random.random((n_units, input_units))
        print("weight matrix: ")
        print(self.weight_matrix)

        # put biases into the weight matrix as the last column
        self.weights_and_biases = np.column_stack((self.weight_matrix, self.bias_vector))
        print("weights and biases")
        print(self.weights_and_biases)

        # create uninitialized pre-activation, activation and input vectors
        self.pre_activation = np.empty((1, self.n_units))
        self.activation = np.empty((1, self.n_units))
        self.input_vector = np.empty((self.input_units + 1, 1))

    def __call__(self, x: np.ndarray, targets: np.ndarray):
        self.input_vector = np.append(x, [[1]], axis=0)
        self.forward_step()
        self.calculate_loss(targets)
        self.backward_step()

    def forward_step(self):
        self.pre_activation = self.weights_and_biases @ self.input_vector
        print("pre_activation:")
        print(self.pre_activation)
        self.activation = relu(self.pre_activation)
        print("activation:")
        print(self.activation)

    def calculate_loss(self, targets):
        self.loss = targets - self.activation
        print("loss:")
        print(self.loss)

    def compute_gradient(self):
        d_l_d_a = -1 * self.loss
        self.error_signal = d_relu(self.pre_activation) * d_l_d_a
        return np.transpose(self.input_vector) @ self.error_signal
    # matrix_mult(transpose(input_vector), error_signal?
    # error_signal = d_relu(preactivation) * dL / d_activation
    # dL / d_activation = (y - t)? so loss * (-1)?

    def backward_step(self):
        gradient = self.compute_gradient()
        print("gradient:")
        print(gradient)
        self.weights_and_biases = self.weights_and_biases - gradient
        print("new weights and biases:")
        print(self.weights_and_biases)


layer = Layer(1, 3)
layer(np.array([[-1], [0], [1]]), np.array([[3]]))
