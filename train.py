#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import numpy as np
import tensorflow as tf

from pipeline import mnist_batch_iter
from output import produce_grid, produce_gif
from graph import d_train_step, d_loss, g_train_step, g_loss, g_z, g_o, gz, dx
from graph import d_real_y
from config import NNConfig

N_CRITIC, EPOCH = NNConfig.N_CRITIC, NNConfig.EPOCH

is_train = tf.get_default_graph().get_tensor_by_name('is_train:0')
try:
    # y_dx [BATCH_SIZE, 64, 64, 10]
    y_dx = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(dx.name))
    # y_gz [BATCH_SIZE, 1, 1, 10]
    y_gz = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(gz.name))
    y_dx_fill = d_real_y.reshape(y_gz) * tf.ones_like(y_dx)
    y_gz_fill = d_real_y.reshape(y_gz)
except KeyError:
    pass

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
init_op = tf.global_variables_initializer()

sess.run(init_op)

grids_through_epochs = list()
# 10 * 10 grid
const_z = gz.gaussian_noise([100, 1, 1, 100])
identity = np.eye(10)
assortment = np.array([[i] * 10 for i in range(10)])
const_gz_fill = identity[assortment].reshape(-1, 1, 1, 10)
const_dx_fill = const_gz_fill * np.ones_like(y_dx)

for epoch in range(1, EPOCH + 1):
    try:
        step = 0
        sess.run(mnist_batch_iter.initializer)
        while True:
            try:
                for i in range(N_CRITIC):
                    step += 1
                    x_given_y, z_given_y = sess.run([y_dx_fill, y_gz_fill])
                    _, d_loss_score = sess.run(
                                    fetches=[d_train_step, d_loss],
                                    feed_dict={g_z: gz.gaussian_noise(g_z),
                                               y_dx: x_given_y,
                                               y_gz: z_given_y})
                    print("Epoch {0} of {1}, step {2} "
                          "Discriminator log loss {3:.4f}".format(
                            epoch, EPOCH, step, d_loss_score))

                step += 1
                x_given_y, z_given_y = sess.run([y_dx_fill, y_gz_fill])
                _, g_loss_score = sess.run(
                                    fetches=[g_train_step, g_loss],
                                    feed_dict={g_z: gz.gaussian_noise(g_z),
                                               y_dx: x_given_y,
                                               y_gz: z_given_y})
                print("Epoch {0} of {1}, step {2}"
                      " Generator log loss {3:.4f}".format(
                        epoch, EPOCH, step, g_loss_score))

            except tf.errors.OutOfRangeError:
                print("Epoch {0} has finished.".format(epoch))
                break
    except KeyboardInterrupt:
        print("Ending Training during {0} epoch.".format(epoch))
        break
    test_images = sess.run(g_o, feed_dict={g_z: const_z,
                                           y_dx: const_dx_fill,
                                           y_gz: const_gz_fill,
                                           is_train: False})
    grids_through_epochs.append(
        produce_grid(test_images, epoch, './results', save=True, grid_size=10))

produce_gif(grids_through_epochs, path='./results')
