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
            for name, params in zip(self.layers[:-1], self.hyperparams[:-1]):
                i = self.conv_layer(i, params, name)

            logits = self.conv_layer(i, [[4, 4, 1024, 1], [1]], 'logits',
                                     strides=[1, 1, 1, 1], padding='VALID')
            o = self.σ('sigmoid', logits, bn=False)

        return logits, o


class Generator(_BaseNN):
    """Deep Convolutional structure according to Radford et al. 2015
    removed pooling and densely connected layers.
    """
    def __init__(self, z, layers, hyperparams, output_shape, name=None):
        self._z = z
        self.layers = layers
        self.hyperparams = hyperparams
        self.o_shape = output_shape
        self.name = name or self.__class__.__name__

    def relu(self, input_tensor, bn=True):
        return partial(self.σ, 'relu', input_tensor, bn=bn)

    def deconv_layer(self, input_tensor, hyperparams, o_shape, name, **kwargs):
        shape_w, shape_b = hyperparams
        strides = kwargs.get('strides', [1, 2, 2, 1])
        padding = kwargs.get('padding', 'SAME')
        with tf.variable_scope(name):
            w = self.weight_variable(shape_w)
            b = self.bias_variable(shape_b)
            matmul = tf.nn.conv2d_transpose(input=input_tensor,
                                            filter=w,
                                            output_shape=o_shape,
                                            strides=strides,
                                            padding=padding)
            deconv_layer = self.relu(matmul + b)
        return deconv_layer

    def build(self):
        with tf.variable_scope(self.name, reuse=False):
            i = self.deconv_layer(self._z,
                                  self.hyperparams[0],
                                  self.layers[0],
                                  self.output_shape[0],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID')
            for name, params, o_shape in zip(self.layers[1:],
                                             self.hyperparams[1:],
                                             self.output_shape[1:]):
                i = self.conv_layer(i, params, name, o_shape)
            o = self.σ('tanh', i, bn=False)
        return i, o
