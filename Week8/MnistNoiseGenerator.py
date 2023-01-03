import tensorflow as tf
import matplotlib.pyplot as plt


def add_noise(img, noise_factor):
    # add random noise to the image data
    img = img + tf.random.normal(shape=img.shape) * noise_factor
    # clip values, so we don't go above 1.0
    img = tf.clip_by_value(img, clip_value_min=0., clip_value_max=1.)

    return img


def prepare_data(ds, batch_size, noise_factor):
    # drop labels, cast & normalize image data to between 0 and 1 (since last layer is sigmoid)
    ds = ds.map(lambda img, target: tf.cast(img, tf.float32) / 255.)
    # add noise to images and map to a tuple of (noised, original)
    ds = ds.map(lambda img: (add_noise(img, noise_factor), img))

    return ds.cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def plot_examples(data_dict, n):
    """
    Plots a given number of images for each dataset in the dictionary.
    It takes `n` examples from each dataset and displays them on separate rows.
    Each row will have n elements from the corresponding dataset.
    :param data_dict: titles and datasets to be plotted.
        should have the type {"title1": dataset1, "title2": dataset2...}
    :param n: number of examples, which corresponds to number of examples from each dataset
    """
    rows = len(data_dict)
    plt.figure()
    for i in range(n):
        for j, (title, data) in enumerate(data_dict.items()):
            ax = plt.subplot(rows, n, (j * n) + i + 1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(title)
            plt.imshow(data[i])
    plt.show()
