import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, sequence_length):
        super().__init__()
        self.index_embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=1)
        self.positional_embedding_layer = tf.keras.layers.Embedding(sequence_length, embedding_size, input_length=1)

    def call(self, inputs, training=None, mask=None):
        word_index_embeddings = self.index_embedding_layer(inputs)
        sequence_length = tf.shape(inputs).numpy()[1]  # TODO does this work?
        positional_indices = tf.range(sequence_length)
        positional_embeddings = self.positional_embedding_layer(positional_indices)
        return word_index_embeddings + positional_embeddings
