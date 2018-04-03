#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
# use framework pointer to download data.
from tensorflow.examples.tutorials.mnist import input_data

from cnn import BasicCNN

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

print(mnist.train.next_batch(50)[0].shape)

with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
    # generative network
    g = BasicCNN(shape=(1, 1, 100), num_classes=10)
    g_x, g_y_, g_is_train = g.x, g.y_, g.is_train
    g_conv_1 = g.add_conv_layer(g_x, 'conv_1',
                                [[1, 1, 100, 1024], [1024]])
    g_conv_2 = g.add_conv_layer(g_conv_1, 'conv_2',
                                [[3, 3, 1024, 512], [512]])
    g_conv_3 = g.add_conv_layer(g_conv_2, 'conv_3',
                                [[3, 3, 512, 256], [256]])
    g_conv_4 = g.add_conv_layer(g_conv_3, 'conv_4',
                                [[3, 3, 256, 128], [128]])
    g_o = g.add_conv_layer(g_conv_4, 'output',
                           [[3, 3, 128, 1], [128]],
                           func='tanh',
                           bn=False)

# there is only one single discriminative network but with variables reused
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for real images
    d_real = BasicCNN(shape=(28, 28, 1), num_classes=10)
    d_real_x, d_real_y_, d_real_is_train = d_real.x, d_real.y_, d_real.is_train
    d_real_conv_1 = d_real.add_conv_layer(d_real_x, 'conv_1',
                                          [[3, 3, 1, 128], [128]])
    d_real_conv_2 = d_real.add_conv_layer(d_real_conv_1, 'conv_2',
                                          [[3, 3, 128, 256], [256]])
    d_real_conv_3 = d_real.add_conv_layer(d_real_conv_2, 'conv_3',
                                          [[3, 3, 256, 512], [512]])
    d_real_conv_4 = d_real.add_conv_layer(d_real_conv_3, 'conv_4',
                                          [[3, 3, 512, 1024], [1024]])
    d_real_o = d_real.add_conv_layer(d_real_conv_4, 'output',
                                     [[3, 3, 1024, 1], [1]],
                                     func='sigmoid', bn=False)

with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for fake (generated) images
    d_fake = BasicCNN(shape=(28, 28, 1), num_classes=10)
    d_fake_x, d_fake_y_, d_fake_is_train = d_fake.x, d_fake.y_, d_fake.is_train
    d_fake_conv_1 = d_fake.add_conv_layer(d_fake_x, 'conv_1',
                                          [[3, 3, 1, 128], [128]])
    d_fake_conv_2 = d_fake.add_conv_layer(d_fake_conv_1, 'conv_2',
                                          [[3, 3, 128, 256], [256]])
    d_fake_conv_3 = d_fake.add_conv_layer(d_fake_conv_2, 'conv_3',
                                          [[3, 3, 256, 512], [512]])
    d_fake_conv_4 = d_fake.add_conv_layer(d_fake_conv_3, 'conv_4',
                                          [[3, 3, 512, 1024], [1024]])
    d_fake_o = d_fake.add_conv_layer(d_fake_conv_4, 'output',
                                     [[3, 3, 1024, 1], [1]],
                                     func='sigmoid', bn=False)


for var in tf.global_variables():
    print(var.name)
