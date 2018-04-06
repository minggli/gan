#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Deep Convolutional Generative Adversial Networks (DCGANs) with MNIST
data.
"""

import tensorflow as tf

from functools import wraps
from config import BATCH_SIZE

tf.set_random_seed(0)

# global is_train flag for both generative and discriminative models.
is_train = tf.placeholder_with_default(input=False, shape=[], name='is_train')


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


@batch_norm({'training': is_train})
def lrelu(tensor, alpha=.2):
    """Leaky Rectified Linear Unit, alleviating gradient vanishing."""
    return tf.maximum(alpha * tensor, tensor)


# construct generative network using transposed convolution layers
# (also known as deconvolution) generate images from white noise signal.
with tf.variable_scope('generator', reuse=False):
    # generative network
    g_x = tf.random_normal([-1, 1, 1, 100], name='gaussian')
    with tf.variable_scope('deconv_1'):
        W = weight_variable([4, 4, 1024, 100])
        b = bias_variable([1024])
        g_conv_1 = lrelu(tf.nn.conv2d_transpose(g_x,
                                                filter=W,
                                                output_shape=[-1, 4, 4, 1024],
                                                strides=[1, 1, 1, 1],
                                                padding='VALID') + b)
    with tf.variable_scope('deconv_2'):
        W = weight_variable([4, 4, 512, 1024])
        b = bias_variable([512])
        g_conv_2 = lrelu(tf.nn.conv2d_transpose(g_conv_1,
                                                filter=W,
                                                output_shape=[-1, 8, 8, 512],
                                                strides=[1, 2, 2, 1],
                                                padding='SAME') + b)
    with tf.variable_scope('deconv_3'):
        W = weight_variable([4, 4, 256, 512])
        b = bias_variable([256])
        g_conv_3 = lrelu(tf.nn.conv2d_transpose(g_conv_2,
                                                filter=W,
                                                output_shape=[-1, 16, 16, 256],
                                                strides=[1, 2, 2, 1],
                                                padding='SAME') + b)
    with tf.variable_scope('deconv_4'):
        W = weight_variable([4, 4, 128, 256])
        b = bias_variable([128])
        g_conv_4 = lrelu(tf.nn.conv2d_transpose(g_conv_3,
                                                filter=W,
                                                output_shape=[-1, 32, 32, 128],
                                                strides=[1, 2, 2, 1],
                                                padding='SAME') + b)
    with tf.variable_scope('logits'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([1])
        g_logits = tf.nn.conv2d_transpose(g_conv_4,
                                          filter=W,
                                          output_shape=[-1, 64, 64, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME') + b
    g_o = tf.nn.tanh(g_logits)

# there is only one single discriminative network with variables reused
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for fake (generated) images
    with tf.variable_scope('conv_1'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([128])
        d_fake_conv_1 = lrelu(tf.nn.conv2d(input=g_o,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_2'):
        W = weight_variable([4, 4, 128, 256])
        b = bias_variable([256])
        d_fake_conv_2 = lrelu(tf.nn.conv2d(input=d_fake_conv_1,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_3'):
        W = weight_variable([4, 4, 256, 512])
        b = bias_variable([512])
        d_fake_conv_3 = lrelu(tf.nn.conv2d(input=d_fake_conv_2,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_4'):
        W = weight_variable([4, 4, 512, 1024])
        b = bias_variable([1024])
        d_fake_conv_4 = lrelu(tf.nn.conv2d(input=d_fake_conv_3,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('logits'):
        W = weight_variable([4, 4, 1024, 1])
        b = bias_variable([1])
        d_fake_logits = tf.nn.conv2d(input=d_fake_conv_4,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID') + b
    d_fake_o = tf.nn.sigmoid(d_fake_logits)

# there is only one single discriminative network with variables reused
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for real images
    d_real_x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1],
                              name='feature')
    with tf.variable_scope('conv_1'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([128])
        d_real_conv_1 = lrelu(tf.nn.conv2d(input=d_real_x,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_2'):
        W = weight_variable([4, 4, 128, 256])
        b = bias_variable([256])
        d_real_conv_2 = lrelu(tf.nn.conv2d(input=d_real_conv_1,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_3'):
        W = weight_variable([4, 4, 256, 512])
        b = bias_variable([512])
        d_real_conv_3 = lrelu(tf.nn.conv2d(input=d_real_conv_2,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_4'):
        W = weight_variable([4, 4, 512, 1024])
        b = bias_variable([1024])
        d_real_conv_4 = lrelu(tf.nn.conv2d(input=d_real_conv_3,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('logits'):
        W = weight_variable([4, 4, 1024, 1])
        b = bias_variable([1])
        d_real_logits = tf.nn.conv2d(input=d_real_conv_4,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID') + b
    d_real_o = tf.nn.sigmoid(d_real_logits)

# cost functions for D(x) and G(z) respectively
d_real_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_real_logits,
                                    labels=tf.ones([BATCH_SIZE, 1, 1, 1]))
d_fake_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_fake_logits,
                                    labels=tf.zeros([BATCH_SIZE, 1, 1, 1]))
d_loss = tf.reduce_mean(d_real_xentropy) + tf.reduce_mean(d_fake_xentropy)
g_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_fake_logits,
                                    labels=tf.ones([BATCH_SIZE, 1, 1, 1]))
g_loss = tf.reduce_mean(g_xentropy)

# Mini-batch SGD optimisers for J for both Networks
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_train_step = tf.train.RMSPropOptimizer(learning_rate=1e-2).minimize(
                            d_loss,
                            var_list=tf.trainable_variables('discriminator'))
    g_train_step = tf.train.RMSPropOptimizer(learning_rate=1e-2).minimize(
                            g_loss,
                            var_list=tf.trainable_variables('generator'))

dis_vars = [var for var in tf.trainable_variables('dis')
            if 'weight' in var.name]
gen_vars = [var for var in tf.trainable_variables('gen')
            if 'weight' in var.name]

for d_var, g_var in zip(dis_vars, gen_vars):
    print('{0:<90}{1}'.format(str(d_var), str(g_var)))
