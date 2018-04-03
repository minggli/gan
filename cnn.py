# -*- coding: utf-8 -*-
"""
cnn

abstraction built on top of tensorflow for constructing Convolutional Neural
Network
"""

import warnings

import tensorflow as tf


class _BaseCNN(object):

    def __init__(self, shape, num_classes, keep_prob):
        """shape: [n_samples, channels, n_features]"""
        self._shape = shape
        self._n_class = num_classes
        self._keep_rate = keep_prob
        self.is_train = self._is_train

        self.keep_prob = self._keep_prob

    @staticmethod
    def _weight_variable(shape):
        init = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable('weight',
                               shape=shape,
                               initializer=init)

    @staticmethod
    def _bias_variable(shape):
        init = tf.constant_initializer(0.1)
        return tf.get_variable('bias',
                               shape=shape,
                               initializer=init)

    @staticmethod
    def _conv2d(x, W):
        """
        core operation that convolves through image data and extract features
        input: takes a 4-D shaped tensor e.g. (-1, 90, 160, 3)
        receptive field (filter): filter size and number of output channels are
            inferred from weight hyperparams.
        receptive field moves by 1 pixel at a time during convolution.
        Zero Padding algorthm appies to keep output size the same
            e.g. 3x3xn filter with 1 zero-padding, 5x5 2, 7x7 3 etc.
        """
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    @staticmethod
    def _max_pool(x, kernel_size):
        """max pooling with kernal size 2x2 and slide by 2 pixels each time"""
        return tf.nn.max_pool(value=x,
                              ksize=kernel_size,
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    @staticmethod
    def _average_pool(x, kernel_size):
        """avg pooling with kernal size 2x2 and slide by 2 pixels each time"""
        return tf.nn.avg_pool(value=x,
                              ksize=kernel_size,
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    @staticmethod
    def _nonlinearity(activation='relu'):
        assert activation in ['relu', 'sigmoid', 'tanh', 'lrelu', None], \
                              "unspecified activation function."
        if activation == 'sigmoid':
            return tf.nn.sigmoid
        elif activation == 'relu':
            return tf.nn.relu
        elif activation == 'tanh':
            return tf.tanh
        elif activation == 'lrelu':
            return lambda x: tf.maximum(.2 * x, x)
        elif activation is None:
            return lambda x: x

    @property
    def x(self):
        """feature set"""
        warnings.warn("using placeholder to feed data will cause low "
                      "efficiency between Python and C++ interface",
                      RuntimeWarning)
        return tf.placeholder(dtype=tf.float32,
                              # transform 3D shape to 4D to include batch size
                              shape=(None, ) + self._shape,
                              name='feature')

    @property
    def y_(self):
        """ground truth, in one-hot format"""
        warnings.warn("using placeholder to feed data will cause low "
                      "efficiency between Python and C++ interface",
                      RuntimeWarning)
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._n_class),
                              name='label')

    @property
    def _keep_prob(self):
        """the probability constant to keep output from previous layer."""
        return tf.cond(self.is_train, lambda: tf.constant(self._keep_rate),
                       lambda: tf.constant(1.))

    @property
    def _is_train(self):
        """indicates if network is under training mode, default False."""
        return tf.placeholder_with_default(input=False,
                                           shape=[],
                                           name='is_train')

    def _batch_normalize(self, input_layer):
        """batch normalization layer"""
        reuse_flag = True if self.is_train is False else None
        return tf.contrib.layers.batch_norm(inputs=input_layer,
                                            decay=0.99,
                                            center=True,
                                            scale=True,
                                            is_training=self.is_train,
                                            reuse=reuse_flag)


class BasicCNN(_BaseCNN):
    def __init__(self, shape, num_classes, keep_prob=.5):
        super(BasicCNN, self).__init__(shape, num_classes, keep_prob)

    def add_conv_layer(self, input_layer, name, hyperparams, func='lrelu',
                       bn=True):
        """Convolution Layer with hyperparamters and activation and batch
        normalization after nonlinearity as opposed to before nonlinearity as
        cited in Ioffe and Szegedy 2015."""
        with tf.variable_scope(name):
            W = self._weight_variable(shape=hyperparams[0])
            b = self._bias_variable(shape=hyperparams[1])
            if bn:
                return self._batch_normalize(self._nonlinearity(func)(
                                             self._conv2d(input_layer, W) + b))
            elif not bn:
                return self._nonlinearity(func)(
                                             self._conv2d(input_layer, W) + b)

    def add_pooling_layer(self,
                          input_layer,
                          name,
                          kernel_size=[1, 2, 2, 1],
                          mode='max'):
        """max pooling layer to reduce overfitting"""
        with tf.variable_scope(name):
            if mode == 'max':
                return self._max_pool(input_layer, kernel_size)
            elif mode == 'average':
                return self._average_pool(input_layer, kernel_size)

    def add_dense_layer(self, input_layer, name, hyperparams, func='lrelu',
                        bn=True):
        """Densely Connected Layer with hyperparamters and activation. Batch
        normalization inserted after nonlinearity as opposed to before as
        cited in Ioffe and Szegedy 2015."""
        with tf.variable_scope(name):
            W = self._weight_variable(shape=hyperparams[0])
            b = self._bias_variable(shape=hyperparams[1])
            x_ravel = tf.reshape(input_layer, shape=[-1, hyperparams[0][0]])
            if bn:
                return self._batch_normalize(
                       self._nonlinearity(func)(tf.matmul(x_ravel, W) + b))
            elif not bn:
                return self._nonlinearity(func)(tf.matmul(x_ravel, W) + b)

    def add_drop_out_layer(self, input_layer, name):
        """drop out layer to reduce overfitting"""
        with tf.variable_scope(name):
            return tf.nn.dropout(input_layer, self.keep_prob)

    def add_read_out_layer(self, input_layer, name):
        """read out layer with output shape of [batch_size, num_classes]
        in order to feed into softmax"""
        with tf.variable_scope(name):
            input_layer_m = int(input_layer.get_shape()[1])
            W = self._weight_variable(shape=[input_layer_m, self._n_class])
            b = self._bias_variable(shape=[self._n_class])
            return tf.matmul(input_layer, W) + b
