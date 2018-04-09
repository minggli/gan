import tensorflow as tf

from functools import wraps


def weight_variable(shape):
    init = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('weight', shape=shape, initializer=init)


def bias_variable(shape):
    init = tf.constant_initializer(0.1)
    return tf.get_variable('bias', shape=shape, initializer=init)


def batch_norm(params):
    def decorator(func):
        """batch normalization"""
        @wraps(func)
        def wrapper(arg):
            return tf.layers.batch_normalization(inputs=func(arg), **params)
        return wrapper
    return decorator
