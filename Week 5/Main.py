import tensorflow as tf

import PrepareDatasets
from ConvolutionalModel import ConvolutionModel
from Trainer import Trainer

train_ds, test_ds = PrepareDatasets.get_cifar()

# Set loss function & optimizer
loss_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Set metrics
metrics = {"training": {"loss": tf.keras.metrics.Mean(name='train_loss'),
                        "accuracy": tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')},
           "test": {"loss": tf.keras.metrics.Mean(name='test_loss'),
                    "accuracy": tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')}}
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

# Set model
basic_model = ConvolutionModel(loss_function, optimizer, metrics)

# Train model
num_epochs = 15
trainer = Trainer(basic_model, metrics)
trainer.training_loop(train_ds, test_ds, num_epochs)
