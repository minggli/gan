#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# use framework pointer to download data.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from cnn import BasicCNN

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

print(mnist.train.next_batch(50)[0].shape)

# generative network
g = BasicCNN(shape=(1, 1, 100), num_classes=10)
g_x, g_y_, g_is_train = g.x, g.y_, g.is_train
g_conv_1 = g.add_conv_layer(g_x, [[1, 1, 100, 1024], [1024]], func='lrelu')
g_conv_2 = g.add_conv_layer(g_conv_1, [[3, 3, 1024, 512], [512]], func='lrelu')
g_conv_3 = g.add_conv_layer(g_conv_2, [[3, 3, 512, 256], [256]], func='lrelu')
g_conv_4 = g.add_conv_layer(g_conv_3, [[3, 3, 256, 128], [128]], func='lrelu')
g_o = g.add_conv_layer(g_conv_4, [[3, 3, 128, 1]], [1], func='tanh', bn=False)
print(g_o.shape)

# discriminative network
d = BasicCNN(shape=(784, ), num_classes=10)
d_x, d_y_, d_is_train = d.x, d.y_, d.is_train
d_conv_1 = d.add_conv_layer(d_x, [[3, 3, 784, 128], [128]], func='lrelu')
d_conv_2 = d.add_conv_layer(d_conv_1, [[3, 3, 128, 256], [256]], func='lrelu')
d_conv_3 = d.add_conv_layer(d_conv_2, [[3, 3, 256, 512], [512]], func='lrelu')
d_conv_4 = d.add_conv_layer(d_conv_3, [[3, 3, 512, 1024], [1024]], func='lrelu')
d_o = d.add_conv_layer(g_conv_4, [[3, 3, 1024, 1]], [1], func='sigmoid', bn=False)
print(d_o)


if __name__ == '__main__':
    print('Hello world.')
