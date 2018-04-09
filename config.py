"""
config

module keeping some of model parameters
"""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 20,
                            """Number of epochs to run.""")


class BaseConfig(object):
    pass


class NNConfig(BaseConfig):
    BATCH_SIZE = 100
    EPOCH = FLAGS.num_epochs
    ALPHA = 1e-4
