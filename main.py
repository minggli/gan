#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
# use framework pointer to download data.
from tensorflow.examples.tutorials.mnist import input_data

from cnn import BasicCNN

BATCH_SIZE = 50

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

print(mnist.train.next_batch(BATCH_SIZE)[0].shape)

with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
    # generative network
    g = BasicCNN(shape=(1, 1, 100), num_classes=10)
    g_x, g_y_, g_is_train = g.x, g.y_, g.is_train
    with tf.variable_scope('conv_1'):
        W = g._weight_variable(shape=[1, 1, 100, 1024])
        b = g._bias_variable(shape=[1024])
        g_conv_1 = g._batch_normalize(g._nonlinearity('lrelu')(
                                tf.nn.conv2d(input=g_x,
                                             filter=W,
                                             strides=[1, 1, 1, 1],
                                             padding='VALID') + b))
    g_conv_2 = g.add_conv_layer(g_conv_1, 'conv_2',
                                [[4, 4, 1024, 512], [512]])
    g_conv_3 = g.add_conv_layer(g_conv_2, 'conv_3',
                                [[4, 4, 512, 256], [256]])
    g_conv_4 = g.add_conv_layer(g_conv_3, 'conv_4',
                                [[4, 4, 256, 128], [128]])
    g_logits = g.add_conv_layer(g_conv_4, 'logits',
                                [[4, 4, 128, 1], [1]], func=None, bn=False)
    g_o = tf.nn.tanh(g_logits)

with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for real images
    d_real = BasicCNN(shape=(64, 64, 1), num_classes=10)
    d_real_x, d_real_y_, d_real_is_train = d_real.x, d_real.y_, d_real.is_train
    d_real_conv_1 = d_real.add_conv_layer(d_real_x, 'conv_1',
                                          [[4, 4, 1, 128], [128]])
    d_real_conv_2 = d_real.add_conv_layer(d_real_conv_1, 'conv_2',
                                          [[4, 4, 128, 256], [256]])
    d_real_conv_3 = d_real.add_conv_layer(d_real_conv_2, 'conv_3',
                                          [[4, 4, 256, 512], [512]])
    d_real_conv_4 = d_real.add_conv_layer(d_real_conv_3, 'conv_4',
                                          [[4, 4, 512, 1024], [1024]])
    with tf.variable_scope('logits'):
        W = g._weight_variable(shape=[4, 4, 1024, 1])
        b = g._bias_variable(shape=[1])
        d_real_logits = tf.nn.conv2d(input=d_real_conv_4,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID') + b
    d_real_o = tf.nn.sigmoid(d_real_logits)

# there is only one single discriminative network with variables reused
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for fake (generated) images
    d_fake = BasicCNN(shape=(64, 64, 1), num_classes=10)
    d_fake_x, d_fake_y_, d_fake_is_train = d_fake.x, d_fake.y_, d_fake.is_train
    d_fake_conv_1 = d_fake.add_conv_layer(d_fake_x, 'conv_1',
                                          [[4, 4, 1, 128], [128]])
    d_fake_conv_2 = d_fake.add_conv_layer(d_fake_conv_1, 'conv_2',
                                          [[4, 4, 128, 256], [256]])
    d_fake_conv_3 = d_fake.add_conv_layer(d_fake_conv_2, 'conv_3',
                                          [[4, 4, 256, 512], [512]])
    d_fake_conv_4 = d_fake.add_conv_layer(d_fake_conv_3, 'conv_4',
                                          [[4, 4, 512, 1024], [1024]])
    with tf.variable_scope('logits'):
        W = g._weight_variable(shape=[4, 4, 1024, 1])
        b = g._bias_variable(shape=[1])
        d_fake_logits = tf.nn.conv2d(input=d_fake_conv_4,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID') + b
    d_fake_o = tf.nn.sigmoid(d_fake_logits)

# cost functions for D(x) and G(z) respectively
d_real_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_real_logits,
                                    labels=tf.ones([BATCH_SIZE, 1, 1, 1]))
d_fake_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_fake_logits,
                                    labels=tf.zeros([BATCH_SIZE, 1, 1, 1]))
d_loss = tf.reduce_mean(d_real_xentropy) + tf.reduce_mean(d_fake_xentropy)
g_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=g_logits,
                                    labels=tf.ones([BATCH_SIZE, 1, 1, 1]))
g_loss = tf.reduce_mean(g_xentropy)

for var in tf.global_variables():
    print(var.name)

for i in [d_real_conv_1, d_real_conv_2, d_real_conv_3, d_real_conv_4,
          d_real_logits]:
    print(i.shape)
