#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import tensorflow as tf

from graph import Generator, Discriminator, Loss
from config import NNConfig, g_params, d_params
from pipeline import feed, mnist_batch_iter
from output import produce_grid, produce_gif

BATCH_SIZE, EPOCH, LR, N_CRITIC = \
    NNConfig.BATCH_SIZE, NNConfig.EPOCH, NNConfig.ALPHA, NNConfig.N_CRITIC

g_x = tf.placeholder(shape=[BATCH_SIZE, 1, 1, 100], dtype=tf.float32)
is_train = tf.placeholder_with_default(input=True, shape=[], name='is_train')

d_real_x = feed
# gaussian noise to improve chance of intersection of D and G.
input_dims = d_real_x.shape
ε = tf.random_normal(input_dims)

d_real_logits, _ = Discriminator(d_real_x + ε, d_params).build(bn=False)

g = Generator(g_x, g_params)
g_logits, g_o = g.build()
d_fake_logits, _ = Discriminator(g_o + ε, d_params).build(bn=False)

# uniform noise for penalty terms
ε_penalty = tf.random_uniform(input_dims, name='epsilon')
x_hat = (1 - ε_penalty) * d_real_x + ε_penalty * g_o
d_penalty_logits, _ = Discriminator(x_hat, d_params).build(bn=False)
derivative, = tf.gradients(d_penalty_logits, [x_hat])

# Wasserstein distance with gradient penalty
d_loss, g_loss = Loss(d_real_logits, d_fake_logits).wasserstein(derivative)

d_train_step = tf.train.AdamOptimizer(LR, beta1=.5, beta2=.9).minimize(
                        d_loss,
                        var_list=tf.trainable_variables('Discriminator'))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    g_train_step = tf.train.AdamOptimizer(LR, beta1=.5, beta2=.9).minimize(
                            g_loss,
                            var_list=tf.trainable_variables('Generator'))

sess = tf.Session()
init_op = tf.global_variables_initializer()

sess.run(init_op)

for var in tf.trainable_variables():
    print(var)

grids_through_epochs = list()
constant = g.gaussian_noise(g_x)

for epoch in range(1, EPOCH + 1):
    try:
        step = 0
        sess.run(mnist_batch_iter.initializer)
        while True:
            try:
                for i in range(N_CRITIC):
                    step += 1
                    _, d_loss_score = sess.run(
                                        fetches=[d_train_step, d_loss],
                                        feed_dict={g_x: g.gaussian_noise(g_x)})
                    print("Epoch {0} of {1}, step {2} "
                          "Discriminator log loss {3:.4f}".format(
                            epoch, EPOCH, step, d_loss_score))

                step += 1
                _, g_loss_score = sess.run(
                                        fetches=[g_train_step, g_loss],
                                        feed_dict={g_x: g.gaussian_noise(g_x)})
                print("Epoch {0} of {1}, step {2}"
                      " Generator log loss {3:.4f}".format(
                        epoch, EPOCH, step, g_loss_score))

            except tf.errors.OutOfRangeError:
                print("Epoch {0} has finished.".format(epoch))
                break
    except KeyboardInterrupt:
        print("Ending Training during {0} epoch.".format(epoch))
        break
    test_images = sess.run(g_o, feed_dict={g_x: constant, is_train: False})
    grids_through_epochs.append(
        produce_grid(test_images, epoch, './results', save=True))

produce_gif(grids_through_epochs, path='./results')
