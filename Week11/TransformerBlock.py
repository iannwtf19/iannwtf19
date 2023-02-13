import tensorflow as tf


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_size):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=2)  # TODO parameters?
        self.dense1 = tf.keras.layers.Dense(units=embedding_size, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense2 = tf.keras.layers.Dense(units=embedding_size)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None, mask=None):
        y = self.mha(query=inputs, value=inputs, use_causal_mask=True)
        y = self.dropout1(y, training=training)
        y = inputs + y
        ln_out = self.layer_norm1(y, training=training)
        y = self.dense1(y)
        y = self.dense2(y)
        y = self.dropout2(y, training=training)
        y = ln_out + y
        y = self.layer_norm2(y, training=training)
        return y


