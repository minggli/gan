import tensorflow as tf

import warnings
from config import NNConfig
from tensorflow.examples.tutorials.mnist import input_data

warnings.simplefilter('ignore')

__all__ = ['feed', 'mnist_batch_iter']

BATCH_SIZE = NNConfig.BATCH_SIZE

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data = mnist.train.images.reshape(-1, 28, 28, 1)

mnist_images = tf.data.Dataset.from_tensor_slices(data)
mnist_images = mnist_images.map(lambda x: tf.image.resize_images(x, [64, 64]))
# mean centering so (-1, 1)
mnist_images = mnist_images.map(lambda x: (x - 0.5) / 0.5)

mnist_batch = mnist_images.shuffle(1000).apply(
                        tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
mnist_batch_iter = mnist_batch.make_initializable_iterator()
feed = mnist_batch_iter.get_next()
