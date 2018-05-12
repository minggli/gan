import numpy as np
import tensorflow as tf


def weight_variable(shape):
    init = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('weight', shape=shape, initializer=init)


def bias_variable(shape):
    init = tf.constant_initializer(0.1)
    return tf.get_variable('bias', shape=shape, initializer=init)


def gaussian_noise(ph):
    return np.random.normal(0, 1, size=ph.shape)


def batch_norm(tensor, **kwargs):
    ph_is_train = tf.get_default_graph().get_tensor_by_name('is_train:0')
    return tf.layers.batch_normalization(inputs=tensor,
                                         training=ph_is_train,
                                         **kwargs)


def lrelu(tensor, alpha=.2, bn=True):
    """Leaky Rectified Linear Unit, alleviating gradient vanishing."""
    if bn:
        tensor = batch_norm(tensor)
    return tf.maximum(alpha * tensor, tensor)
