#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# use framework pointer to download data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)


if __name__ == '__main__':
    print('Hello world.')
