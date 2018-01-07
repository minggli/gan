#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# use framework pointer to download data.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from cnn import BasicCNN

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

print(mnist.train.next_batch(50)[0].shape)

# generative
g = BasicCNN([None, 100], 10)
# g_dense_1 = g.add_dense_layer(x, [[None, 100], [100], [1 * 1 * 4 * 4]])
# g_conv_1 = g.add_conv_layer(g_dense_1, [[100, 128], [128]], func='relu', bn=True)


if __name__ == '__main__':
    print('Hello world.')
