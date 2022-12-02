import tensorflow as tf
import tensorflow_datasets as tfds


def cifar_prepare(cifar):
    # Normalize image values to floats between 0 and 1
    cifar = cifar.map(lambda img, target: (tf.cast(img, tf.float32) / 255., target))

    # Convert targets to one-hot vector
    cifar = cifar.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # Shuffle, batch and prefetch
    cifar = cifar.shuffle(10000)
    cifar = cifar.batch(32)
    cifar = cifar.prefetch(32)

    cifar.cache()

    return cifar


def get_cifar():
    # Download CIFAR 10 and have a look at the data
    (train_ds, test_ds), ds_info = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, with_info=True)
    print(ds_info)
    tfds.show_examples(train_ds, ds_info)

    train_ds = cifar_prepare(train_ds)
    test_ds = cifar_prepare(test_ds)

    return train_ds, test_ds
