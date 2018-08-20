#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import subprocess

import numpy as np
import tensorflow as tf

from pipeline import mnist_batch_iter, feed
from output import produce_grid, produce_gif
from core import Graph
from helper import condition_matrice

from config import NNConfig, d_params, g_params

N_CRITIC, EPOCH = NNConfig.N_CRITIC, NNConfig.EPOCH

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

# examples over epochs
grids_through_epochs = list()
# 10 * 10 grid
const_z = np.random.normal(0, 1, size=[100, 1, 1, 100])
assortment = np.array([[i] * 10 for i in range(10)])
identity = np.eye(10)
onehot_matrix = identity[assortment]
const_gz_fill, const_dx_fill = condition_matrice(onehot_matrix)


for epoch in range(1, EPOCH + 1):
    try:
        step = 0
        sess.run(mnist_batch_iter.initializer)
        while True:
            try:
                # update D(x) for i in N_CRITIC
                for i in range(N_CRITIC):
                    step += 1
                    X, y = sess.run(feed)
                    y_gz_fill, y_dx_fill = condition_matrice(y)
                    dictionary = {d_real_x: X,
                                  g_z: gz.gaussian_noise(X.shape[0]),
                                  y_dx: y_dx_fill,
                                  y_gz: y_gz_fill}
                    _, d_loss_score = sess.run(fetches=[d_train_step, d_loss],
                                               feed_dict=dictionary)
                    print("Epoch {0} of {1}, step {2} "
                          "Discriminator log loss {3:.4f}".format(
                            epoch, EPOCH, step, d_loss_score))

                # update G(z)
                step += 1
                dictionary = {g_z: gz.gaussian_noise(X.shape[0]),
                              y_dx: y_dx_fill,
                              y_gz: y_gz_fill}

                _, g_loss_score = sess.run(fetches=[g_train_step, g_loss],
                                           feed_dict=dictionary)
                print("Epoch {0} of {1}, step {2}"
                      " Generator log loss {3:.4f}".format(
                        epoch, EPOCH, step, g_loss_score))

            except tf.errors.OutOfRangeError:
                print("Epoch {0} has finished.".format(epoch))
                break
    except KeyboardInterrupt:
        print("Ending Training during {0} epoch.".format(epoch))
        break

    dictionary = {g_z: const_z,
                  is_train: False,
                  y_dx: const_dx_fill,
                  y_gz: const_gz_fill}
    test_images = sess.run(g_o, feed_dict=dictionary)
    grids_through_epochs.append(
        produce_grid(test_images, epoch, './results', save=True, grid_size=10))

produce_gif(grids_through_epochs, path='./results')
subprocess.call("./send_terminate.sh", shell=True)
