# -*- encoding: utf-8 -*-

import numpy as np
import tensorflow as tf


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

    def σ(self, func, input_tensor, bn=True, **kwargs):
        """non-linearity activation function."""
        if bn:
            input_tensor = self._batch_normalize(input_tensor)

        if func == 'lrelu':
            alpha = kwargs.get('alpha', .2)
            return tf.nn.leaky_relu(input_tensor, alpha=alpha)
        elif func == 'relu':
            return tf.nn.relu(input_tensor)
        elif func == 'sigmoid':
            return tf.nn.sigmoid(input_tensor)
        elif func == 'tanh':
            return tf.nn.tanh(input_tensor)
        elif callable(func):
            return func(input_tensor)
        else:
            raise TypeError('unknown activation found.')

    @staticmethod
    def _batch_normalize(input_tensor, **kwargs):
        """batch normalization stablizes input, tracking first and second
        central moments of distribution p(x = i).

        For its full effect, read Ioffe and Szegedy 2015.
        """
        try:
            is_train = tf.get_default_graph().get_tensor_by_name('is_train:0')
        except KeyError:
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
    def __init__(self, x, hyperparams, name=None):
        self._x = x
        self.hyperparams = hyperparams
        self.name = name or self.__class__.__name__

    def lrelu(self, input_tensor, bn=True):
        return self.σ('lrelu', input_tensor, bn=bn, alpha=.2)

    def conv_layer(self, input_tensor, hyperparams):
        name, (shape_w, shape_b), strides, padding = hyperparams
        with tf.variable_scope(name):
            w = self.weight_variable(shape_w)
            b = self.bias_variable(shape_b)
            matmul = tf.nn.conv2d(input=input_tensor,
                                  filter=w,
                                  strides=strides,
                                  padding=padding)
        return matmul + b

    def build(self, **kwargs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            i = self._x
            for params in self.hyperparams[:-1]:
                i = self.lrelu(self.conv_layer(i, params), **kwargs)
            i = self.conv_layer(i, self.hyperparams[-1])
            o = self.σ('sigmoid', i, bn=False)
        return i, o


class Generator(_BaseNN):
    """Deep Convolutional structure according to Radford et al. 2015
    removed pooling and densely connected layers.
    """
    def __init__(self, z, hyperparams, name=None):
        self._z = z
        self.hyperparams = hyperparams
        self.name = name or self.__class__.__name__

    def relu(self, input_tensor, bn=True):
        return self.σ('relu', input_tensor, bn=bn)

    def deconv_layer(self, input_tensor, hyperparams):
        name, (shape_w, shape_b, shape_o), strides, padding = hyperparams
        with tf.variable_scope(name):
            w = self.weight_variable(shape_w)
            b = self.bias_variable(shape_b)
            matmul = tf.nn.conv2d_transpose(value=input_tensor,
                                            filter=w,
                                            output_shape=shape_o,
                                            strides=strides,
                                            padding=padding)
        return matmul + b

    def build(self):
        with tf.variable_scope(self.name, reuse=False):
            i = self._z
            for params in self.hyperparams[:-1]:
                i = self.relu(self.deconv_layer(i, params))
            i = self.deconv_layer(i, self.hyperparams[-1])
            o = self.σ('tanh', i, bn=False)
        return i, o


class Loss(object):
    def __init__(self, d_real_logits, d_fake_logits):
        self.d_real = d_real_logits
        self.d_fake = d_fake_logits

    def goodfellow(self, *args, **kwargs):
        """
        Loss as in Goodfellow et al 2014:
                J(D, G) = 1/2m * sum{log(D(x)) + log(1 - D(G(z)))}
        There is no prebuild loss for GANs, customized loss as below.
        Given tensorflow.nn.sigmoid_cross_entropy_with_logits is:
                J(θ) = - y * log g(z) - (1 - y) * log (1 - g(z))
                                    where z = θ.T * x, g = sigmoid function
        when y = 1, we obtain left side of d_loss: - log(D(x));
        when y = 0, we obtain right side of d_loss: - log(1 - D(G(z)))
        D and G are interpreted as probabilities so sigmoid function squashes
        logits to interval (0, 1).
        hence:
            d_loss = 1/2m * sum{log(D(x)) + log(1 - D[G(z)])}
        """
        # - log D(x)
        d_left_term = tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=self.d_real,
                                            labels=tf.ones_like(self.d_real))
        # - log {1 - D(G(z))}
        d_right_term = tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=self.d_fake,
                                            labels=tf.zeros_like(self.d_fake))
        # - 1 / 2m sum{log D(x) + log {1 - D(G(z))}}
        d_loss = tf.reduce_mean(d_left_term + d_right_term) / 2.
        # - log D(G(z))
        g_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=self.d_fake,
                                            labels=tf.ones_like(self.d_fake))

        g_loss = tf.reduce_mean(g_xentropy)
        return d_loss, g_loss

    def wasserstein(self, derivative=None, lda=10, **kwargs):
        """
        Wasserstein distance as in Arjosky et al 2017:
                J(D, G) = 1/m * sum {f(x)} - 1/m * sum {f(G(z))}
        Gradient Penalty as in Gulrajani et al 2017:
                J(D, G) + ƛ * 1/m * {l2_norm[d f(r) / d r] - 1}**2
            where r = G(z), f does not produce interval (0, 1)
        """
        d_loss = tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake)
        # minimize {- 1/m * sum f(G(z))}
        g_loss = tf.reduce_mean(self.d_fake)
        if derivative is not None:
            norm = tf.sqrt(
                tf.reduce_sum(tf.square(derivative), axis=[1, 2, 3]))
            gradient_penalty = lda * tf.reduce_mean(tf.square(norm - 1.))
            d_loss += gradient_penalty
        return d_loss, g_loss
