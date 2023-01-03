import tensorflow as tf


class ConvolutionalDecoder(tf.keras.Model):
    def __init__(self, embedding_size):
        super().__init__()
        self.dense = tf.keras.layers.Dense(7 * 7, activation='relu', input_shape=(embedding_size,))
        self.reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 1))

        self.transpose1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, activation='relu', strides=2,
                                                          padding='same')
        self.transpose2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=2,
                                                          padding='same')
        self.output_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')

    def call(self, x, **kwargs):
        x = self.dense(x)
        x = self.reshape(x)
        x = self.transpose1(x)
        x = self.transpose2(x)
        x = self.output_conv(x)

        return x
