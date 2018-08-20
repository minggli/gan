import warnings
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from config import NNConfig

warnings.simplefilter('ignore')

__all__ = ['feed', 'mnist_batch_iter']

BATCH_SIZE = NNConfig.BATCH_SIZE

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data = mnist.train.images.reshape(-1, 28, 28, 1)
label = mnist.train.labels

mnist = tf.data.Dataset.from_tensor_slices((data, label))
mnist = mnist.map(lambda x, y: (tf.image.resize_images(x, [64, 64]), y))
# mean centering so (-1, 1)
mnist = mnist.map(lambda x, y: ((x - 0.5) / 0.5, tf.cast(y, tf.float32)))

mnist_batch = mnist.shuffle(1000).batch(BATCH_SIZE)
mnist_batch_iter = mnist_batch.make_initializable_iterator()
feed = mnist_batch_iter.get_next()
