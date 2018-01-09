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
gx, gy_ = g.x, g.y_
g_conv_1 = g.add_conv_layer(gx, [[1, 1, 100, 1024], [1024]], func='lrelu')
g_conv_2 = g.add_conv_layer(g_conv_1, [[3, 3, 1024, 512], [512]], func='lrelu')
g_conv_3 = g.add_conv_layer(g_conv_2, [[3, 3, 512, 256], [256]], func='lrelu')
g_conv_4 = g.add_conv_layer(g_conv_3, [[3, 3, 256, 128], [128]], func='lrelu')
go = g.add_conv_layer(g_conv_4, [[3, 3, 128, 1]], [1], func='tanh', bn=False)
print(go.shape)

# discriminative network
d = BasicCNN(shape=(784, ), num_classes=10)

if __name__ == '__main__':
    print('Hello world.')
