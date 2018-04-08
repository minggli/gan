import tensorflow as tf

from models.official.mnist.dataset import train, test
from config import NNConfig

__all__ = ['mnist_tensor']

BATCH_SIZE = NNConfig.BATCH_SIZE

mnist = train("./mnist_data/").concatenate(test("./mnist_data/"))
mnist_images = mnist.map(lambda img, label: tf.reshape(img, [28, 28, 1]))
mnist_images_resized = mnist_images.map(
                            lambda x: tf.image.resize_images(x, [64, 64]))
mnist_images = mnist_images_resized.map(tf.image.per_image_standardization)
mnist_batch = mnist_images_resized.shuffle(1000).repeat().batch(BATCH_SIZE)
mnist_batch_iter = mnist_batch.make_one_shot_iterator()
mnist_tensor = mnist_batch_iter.get_next()
