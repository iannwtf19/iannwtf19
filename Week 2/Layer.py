import numpy as np

relu = np.vectorize(lambda x: 0 if x < 0 else x)
d_relu = np.vectorize(lambda x: 0 if (x < 0) else 1)  # what if x = 0?


class Layer:

    def __init__(self, n_units, input_units):
        self.targets = None
        self.loss = None
        self.error_signal = None
        self.n_units = n_units  # number of perceptrons in this layer
        self.input_units = input_units  # number of perceptrons in previous layer

        # create an initial bias vector of zeros
        self.bias_vector = np.zeros((n_units, 1))
        # print("bias vector: ")
        # print(self.bias_vector)

        # create initial weight matrix with random values
        self.weight_matrix = np.random.random((n_units, input_units))
        # print("weight matrix: ")
        # print(self.weight_matrix)

        # put biases into the weight matrix as the last column
        self.weights_and_biases = np.column_stack((self.weight_matrix, self.bias_vector))
        # print("weights and biases")
        # print(self.weights_and_biases)

        # create uninitialized pre-activation, activation and input vectors
        self.pre_activation = np.empty((1, self.n_units))
        self.activation = np.empty((1, self.n_units))
        self.input_vector = np.empty((self.input_units + 1, 1))

    def __call__(self, x: np.ndarray, targets: np.ndarray = None):
        # append an input of 1 for the biases
        self.input_vector = np.append(x, [[1]], axis=0)
        # print("input vector:")
        # print(self.input_vector)
        self.forward_step()
        if targets is not None:
            # we know the targets, so we are in the last layer. calculate the loss
            self.targets = targets
            self.calculate_loss(targets)

    def forward_step(self):
        self.pre_activation = self.weights_and_biases @ self.input_vector
        # print("pre_activation:")
        # print(self.pre_activation)
        self.activation = relu(self.pre_activation)
        # print("activation:")
        # print(self.activation)

    def calculate_loss(self, targets):
        self.loss = 0.5 * ((targets - self.activation) ** 2)
        # print("loss:")
        # print(self.loss)

    def backward_step(self, learning_rate=0.01, error_signal_next=None, weights_next=None):
        self.error_signal = self.calculate_error_signal(error_signal_next, weights_next)
        # print("error_signal:")
        # print(self.error_signal)
        gradient = self.compute_gradient()
        # print("gradient:")
        # print(gradient)
        self.weights_and_biases = self.weights_and_biases - (learning_rate * gradient)
        # print("new weights and biases:")
        # print(self.weights_and_biases)
        self.weight_matrix = self.weights_and_biases[:, :-1]

    def calculate_error_signal(self, error_signal_next=None, weights_next=None):
        if error_signal_next is None:
            # output layer, calculate error signal directly
            d_l_d_a = self.activation - self.targets
            return d_relu(self.pre_activation) * d_l_d_a
        else:
            # hidden layer, calculate error signal from next layer
            return np.transpose(weights_next) @ error_signal_next

    def compute_gradient(self):
        return self.error_signal @ np.transpose(self.input_vector)

