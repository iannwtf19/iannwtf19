import tensorflow as tf


class ConvolutionalEncoder(tf.keras.Model):
    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                                            strides=2, input_shape=(28, 28, 1))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                                            strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(embedding_size)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x
