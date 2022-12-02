import tensorflow as tf


class ConvolutionModel(tf.keras.Model):

    def __init__(self, loss_function, optimizer, metrics):
        super(ConvolutionModel, self).__init__()

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loss = metrics["training"]["loss"]
        self.train_accuracy = metrics["training"]["accuracy"]
        self.test_loss = metrics["test"]["loss"]
        self.test_accuracy = metrics["test"]["accuracy"]

        self.convlayer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.convlayer2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.convlayer3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.convlayer4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.pooling(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.global_pool(x)
        x = self.out(x)
        return x

    @tf.function
    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            output = self(images, training=True)
            loss = self.loss_function(labels, output)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, output)

    @tf.function
    def test_step(self, data):
        images, labels = data
        output = self(images, training=False)
        loss = self.loss_function(labels, output)

        self.test_loss.update_state(loss)
        self.test_accuracy.update_state(labels, output)
