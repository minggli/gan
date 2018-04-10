import tensorflow as tf

from models.official.mnist.dataset import train, test
from config import NNConfig

__all__ = ['iter', 'mnist_batch_iter']

BATCH_SIZE = NNConfig.BATCH_SIZE
EPOCH = NNConfig.EPOCH

mnist = train("./mnist_data/").concatenate(test("./mnist_data/"))
mnist_images = mnist.map(lambda img, label: tf.reshape(img, [28, 28, 1]))
mnist_images = mnist_images.map(lambda x: tf.image.resize_images(x, [64, 64]))
# mnist_images already normalized to 0 - 1, tanh produces -1, 1, hence below
mnist_images = mnist_images.map(lambda x: (x - 0.5) / 0.5)

mnist_batch = mnist_images.shuffle(1000).batch(BATCH_SIZE)
mnist_batch_iter = mnist_batch.make_initializable_iterator()
iter = mnist_batch_iter.get_next()
