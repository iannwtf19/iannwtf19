import tensorflow as tf
from ConvolutionalEncoder import ConvolutionalEncoder
from ConvolutionalDecoder import ConvolutionalDecoder


class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, embedding_size):
        super().__init__()

        self.encoder = ConvolutionalEncoder(embedding_size)
        self.decoder = ConvolutionalDecoder(embedding_size)

        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                             tf.keras.metrics.MeanAbsoluteError(name="MAE")]

    def call(self, x, **kwargs):
        z = self.encoder(x)
        x_prime = self.decoder(z)

        return x_prime

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    def train_step(self, data):
        sequence, label = data
        with tf.GradientTape() as tape:
            output = self(sequence, training=True)
            loss = self.compiled_loss(label, output, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(label, output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        sequence, label = data
        output = self(sequence, training=False)
        loss = self.compiled_loss(label, output, regularization_losses=self.losses)

        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(label, output)

        return {m.name: m.result() for m in self.metrics}
