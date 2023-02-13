import tensorflow as tf
from EmbeddingLayer import EmbeddingLayer
from TransformerBlock import TransformerBlock


class TransformerModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, sequence_length, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length

        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_size, sequence_length)
        self.transformer_block = TransformerBlock(embedding_size)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None, mask=None):
        y = self.embedding_size(inputs)
        y = self.transformer_block(y, training=training)
        y = self.output_layer(y)

        return y

    @tf.function
    def train_step(self, data):

        x, targets = data  # TODO how is the data? what are our inputs and targets?

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)

            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)

        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def generate_text(self, prompt, output_length, top_k):
        pass  # TODO implement this!

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()
