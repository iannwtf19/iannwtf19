import tensorflow as tf
import tensorflow_datasets as tfds
import MnistNoiseGenerator as mng
import matplotlib.pyplot as plt
from ConvolutionalAutoencoder import ConvolutionalAutoencoder

# Define hyperparameters
batch_size = 32
learning_rate = 0.001
noise_factor = 0.2
num_epochs = 10

embedding_size = 10

# Generate train & test datasets
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
noisy_train_ds = mng.prepare_data(train_ds, batch_size, noise_factor)
noisy_test_ds = mng.prepare_data(test_ds, batch_size, noise_factor)

# Show noisy and original examples
for noisy, orig in noisy_train_ds.take(1):
    mng.plot_examples({"noisy": noisy, "original": orig}, n=5)

# Instantiate autoencoder, compile and train
autoencoder = ConvolutionalAutoencoder(embedding_size)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.MeanSquaredError())
history = autoencoder.fit(noisy_train_ds, validation_data=noisy_test_ds, epochs=num_epochs)

# Show layer properties
autoencoder.encoder.summary()
autoencoder.decoder.summary()

# Plot training and validation losses
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(labels=["training", "validation"])
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.show()

# Denoise some new examples from test dataset, display alongside noised & original versions for comparison
for noisy, orig in noisy_test_ds.take(1):
    decoded = autoencoder(noisy, training=False)
    mng.plot_examples({"noised": noisy, "original": orig, "denoised": decoded}, n=5)
