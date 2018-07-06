#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
import tensorflow as tf

from functools import partial


class _BaseNN(object):

    @staticmethod
    def weight_variable(shape):
        init = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable('weight', shape=shape, initializer=init)

    @staticmethod
    def bias_variable(shape):
        init = tf.constant_initializer(0.1)
        return tf.get_variable('bias', shape=shape, initializer=init)

    @staticmethod
    def gaussian_noise(ph):
        return np.random.normal(0, 1, size=ph.shape)

    @staticmethod
    def σ(self, func, input_tensor, bn=True, **kwargs):
        """non-linearity activation function."""
        if bn:
            input_tensor = self._batch_normalize(input_tensor)

        if func == 'lrelu':
            alpha = kwargs.get('alpha', .2)
            return tf.maximum(alpha * input_tensor, input_tensor)
        elif func == 'relu':
            return tf.maximum(0, input_tensor)
        elif func == 'sigmoid':
            return tf.nn.sigmoid(input_tensor)
        elif func == 'tanh':
            return tf.tanh(input_tensor)
        elif callable(func):
            return func(input_tensor)
        else:
            raise TypeError('unknown activation found.')

    @staticmethod
    def _batch_normalize(self, input_tensor, **kwargs):
        """batch normalization stablizes input, tracking first and second
        central moments of distribution p(x = i).

        For its full effect, read Ioffe and Szegedy 2015.
        """
        try:
            is_train = tf.get_default_graph().get_tensor_by_name('is_train:0')
        except KeyError:
            with tf.get_default_graph():
                is_train = tf.placeholder_with_default(input=False,
                                                       shape=[],
                                                       name='is_train')
        return tf.layers.batch_normalization(inputs=input_tensor,
                                             training=is_train,
                                             **kwargs)


class Discriminator(_BaseNN):
    """Deep Convolutional structure according to Radford et al. 2015
    removed pooling and densely connected layers.
    """
    def __init__(self, x, layers, hyperparams, name=None):
        self._x = x
        self.layers = layers
        self.hyperparams = hyperparams
        self.name = name or self.__class__.__name__

    def lrelu(self, input_tensor, bn=True):
        return partial(self.σ, 'lrelu', input_tensor, bn=bn, alpha=.2)

    def conv_layer(self, input_tensor, hyperparams, name, **kwargs):
        shape_w, shape_b = hyperparams
        strides = kwargs.get('strides', [1, 2, 2, 1])
        padding = kwargs.get('padding', 'SAME')
        with tf.variable_scope(name):
            w = self.weight_variable(shape_w)
            b = self.bias_variable(shape_b)
            matmul = tf.nn.conv2d(input=input_tensor,
                                  filter=w,
                                  strides=strides,
                                  padding=padding)
            conv_layer = self.lrelu(matmul + b)
        return conv_layer

    def build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            i = self._x
            for name, params in zip(self.layers, self.hyperparams):
                i = self.conv_layer(i, params, name)

            # logits
            logits = self.conv_layer(i, [[4, 4, 1024, 1], [1]], 'logits',
                                     strides=[1, 1, 1, 1], padding='VALID')
            o = self.σ('sigmoid', logits, bn=False)

        return logits, o



# there is only one single discriminative network with variables reused
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for fake (generated) images
    with tf.variable_scope('conv_1'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([128])
        d_fake_conv_1 = lrelu(tf.nn.conv2d(input=g_o,
                                           # takes in output of generator
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


# construct generative network using transposed convolution layers
# (also known as deconvolution) generate images from white noise signal.
with tf.variable_scope('generator', reuse=False):
    # generative network
    with tf.variable_scope('deconv_1'):
        W = weight_variable([4, 4, 1024, 100])
        b = bias_variable([1024])
        g_conv_1 = lrelu(tf.nn.conv2d_transpose(
                            g_x,
                            filter=W,
                            output_shape=[BATCH_SIZE, 4, 4, 1024],
                            strides=[1, 1, 1, 1],
                            padding='VALID') + b)
    with tf.variable_scope('deconv_2'):
        W = weight_variable([4, 4, 512, 1024])
        b = bias_variable([512])
        g_conv_2 = lrelu(tf.nn.conv2d_transpose(
                            g_conv_1,
                            filter=W,
                            output_shape=[BATCH_SIZE, 8, 8, 512],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b)
    with tf.variable_scope('deconv_3'):
        W = weight_variable([4, 4, 256, 512])
        b = bias_variable([256])
        g_conv_3 = lrelu(tf.nn.conv2d_transpose(
                            g_conv_2,
                            filter=W,
                            output_shape=[BATCH_SIZE, 16, 16, 256],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b)
    with tf.variable_scope('deconv_4'):
        W = weight_variable([4, 4, 128, 256])
        b = bias_variable([128])
        g_conv_4 = lrelu(tf.nn.conv2d_transpose(
                            g_conv_3,
                            filter=W,
                            output_shape=[BATCH_SIZE, 32, 32, 128],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b)
    with tf.variable_scope('logits'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([1])
        g_logits = tf.nn.conv2d_transpose(
                            g_conv_4,
                            filter=W,
                            output_shape=[BATCH_SIZE, 64, 64, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b
    g_o = tf.nn.tanh(g_logits)
