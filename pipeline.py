import tensorflow as tf

from models.official.mnist.dataset import train
from config import BATCH_SIZE

mnist_train = train("./mnist_data/").shuffle(1000).repeat().batch(BATCH_SIZE)
iterator = mnist_train.make_one_shot_iterator()

sample_batch = iterator.get_next()

sess = tf.Session()

images, labels = sess.run(sample_batch)
print(images.shape)
