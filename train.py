#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import subprocess

import numpy as np
import tensorflow as tf

from pipeline import mnist_batch_iter
from output import produce_grid, produce_gif
from core import Graph
from helper import train
from config import NNConfig, d_params, g_params


d_train_step, d_loss, g_train_step, g_loss, g_z, g_o, gz, dx, d_real_x = \
    Graph(NNConfig, d_params, g_params).build()

is_train = tf.get_default_graph().get_tensor_by_name('is_train:0')
y_dx = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(dx.name))
y_gz = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(gz.name))

global_saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init_op = tf.global_variables_initializer()

sess.run(init_op)

# # examples over epochs
# grids_through_epochs = list()
# # 10 * 10 grid
# const_z = np.random.normal(0, 1, size=[100, 1, 1, 100])
# assortment = np.array([[i] * 10 for i in range(10)])
# identity = np.eye(10)
# onehot_matrix = identity[assortment]
# const_gz_fill, const_dx_fill = condition_matrice(onehot_matrix)


train(sess, mnist_batch_iter, d_real_x, g_z, y_dx, y_gz, d_train_step, d_loss,
      g_train_step, g_loss)

#
# test_images = sess.run(g_o, feed_dict={g_z: const_z,
#                                     is_train: False,
#                                            y_dx: const_dx_fill,
#                                            y_gz: const_gz_fill})
# grids_through_epochs.append(
#         produce_grid(test_images, epoch, './results', save=True, grid_size=10))
#
# produce_gif(grids_through_epochs, path='./results')
# subprocess.call("./send_terminate.sh", shell=True)
