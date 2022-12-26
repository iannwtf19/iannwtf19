import tensorflow as tf


class LSTMCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)

        self.forget_gate = tf.keras.layers.Dense(units,
                                                 kernel_initializer=tf.keras.initializers.Orthogonal(),
                                                 activation=tf.nn.sigmoid)
        self.input_gate = tf.keras.layers.Dense(units,
                                                kernel_initializer=tf.keras.initializers.Orthogonal(),
                                                activation=tf.nn.sigmoid)
        self.cell_state_candidate = tf.keras.layers.Dense(units,
                                                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                                                          activation=tf.nn.tanh)
        self.output_gate = tf.keras.layers.Dense(units,
                                                 kernel_initializer=tf.keras.initializers.Orthogonal(),
                                                 activation=tf.nn.sigmoid)

    def call(self, inputs, states):
        hidden_state = states[0]
        cell_state = states[1]

        # concatenate input and previous hidden state
        x = tf.concat([hidden_state, inputs])

        # forget some portion of the previous cell state
        forget_rate = self.forget_gate(x)
        retained_cell_state = forget_rate * cell_state

        # calculate new cell state
        input_rate = self.input_gate(x)
        cell_state_candidate = self.cell_state_candidate(x)
        new_cell_state = input_rate * cell_state_candidate + retained_cell_state

        # calculate output (new hidden state)
        output_rate = self.output_gate(x)
        hidden_state_candidate = tf.nn.tanh(new_cell_state)  # no weights and biases, directly apply tanh
        new_hidden_state = output_rate * hidden_state_candidate

        return new_hidden_state, [new_hidden_state, new_cell_state]

    @property
    def state_size(self):
        return [tf.TensorShape([self.units]), tf.TensorShape([self.units])]

    @property
    def output_size(self):
        return [tf.TensorShape([self.units])]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, self.units]), tf.zeros([batch_size, self.units])]
