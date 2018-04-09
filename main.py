#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Deep Convolutional Generative Adversial Networks (DCGANs) with MNIST
data.
"""
import tensorflow as tf

from config import NNConfig
from pipeline import iter, mnist_batch_iter
from helpers import weight_variable, bias_variable, batch_norm


BATCH_SIZE, EPOCH, LR = NNConfig.BATCH_SIZE, NNConfig.EPOCH, NNConfig.ALPHA

# global is_train flag for both generative and discriminative models.
is_train = tf.placeholder_with_default(input=False, shape=[], name='is_train')


@batch_norm({'training': is_train})
def lrelu(tensor, alpha=.2):
    """Leaky Rectified Linear Unit, alleviating gradient vanishing."""
    return tf.maximum(alpha * tensor, tensor)


# construct generative network using transposed convolution layers
# (also known as deconvolution) generate images from white noise signal.
with tf.variable_scope('generator', reuse=False):
    # generative network
    g_x = tf.random_normal([BATCH_SIZE, 1, 1, 100], name='gaussian_generator')
    with tf.variable_scope('deconv_1'):
        W = weight_variable([4, 4, 1024, 100])
        b = bias_variable([1024])
        g_conv_1 = lrelu(tf.nn.conv2d_transpose(
                            g_x,
                            filter=W,
                            output_shape=[BATCH_SIZE, 4, 4, 1024],
                            strides=[1, 1, 1, 1],
                            padding='VALID') + b)
    with tf.variable_scope('deconv_2'):
        W = weight_variable([4, 4, 512, 1024])
        b = bias_variable([512])
        g_conv_2 = lrelu(tf.nn.conv2d_transpose(
                            g_conv_1,
                            filter=W,
                            output_shape=[BATCH_SIZE, 8, 8, 512],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b)
    with tf.variable_scope('deconv_3'):
        W = weight_variable([4, 4, 256, 512])
        b = bias_variable([256])
        g_conv_3 = lrelu(tf.nn.conv2d_transpose(
                            g_conv_2,
                            filter=W,
                            output_shape=[BATCH_SIZE, 16, 16, 256],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b)
    with tf.variable_scope('deconv_4'):
        W = weight_variable([4, 4, 128, 256])
        b = bias_variable([128])
        g_conv_4 = lrelu(tf.nn.conv2d_transpose(
                            g_conv_3,
                            filter=W,
                            output_shape=[BATCH_SIZE, 32, 32, 128],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b)
    with tf.variable_scope('logits'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([1])
        g_logits = tf.nn.conv2d_transpose(
                            g_conv_4,
                            filter=W,
                            output_shape=[BATCH_SIZE, 64, 64, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME') + b
    g_o = tf.nn.tanh(g_logits)

# there is only one single discriminative network with variables reused
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for fake (generated) images
    with tf.variable_scope('conv_1'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([128])
        d_fake_conv_1 = lrelu(tf.nn.conv2d(input=g_o,
                                           # takes in output of generator
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_2'):
        W = weight_variable([4, 4, 128, 256])
        b = bias_variable([256])
        d_fake_conv_2 = lrelu(tf.nn.conv2d(input=d_fake_conv_1,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_3'):
        W = weight_variable([4, 4, 256, 512])
        b = bias_variable([512])
        d_fake_conv_3 = lrelu(tf.nn.conv2d(input=d_fake_conv_2,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_4'):
        W = weight_variable([4, 4, 512, 1024])
        b = bias_variable([1024])
        d_fake_conv_4 = lrelu(tf.nn.conv2d(input=d_fake_conv_3,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('logits'):
        W = weight_variable([4, 4, 1024, 1])
        b = bias_variable([1])
        d_fake_logits = tf.nn.conv2d(input=d_fake_conv_4,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID') + b
    d_fake_o = tf.nn.sigmoid(d_fake_logits)

# there is only one single discriminative network with variables reused
with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
    # discriminative network for real images
    with tf.variable_scope('conv_1'):
        W = weight_variable([4, 4, 1, 128])
        b = bias_variable([128])
        d_real_conv_1 = lrelu(tf.nn.conv2d(input=iter,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_2'):
        W = weight_variable([4, 4, 128, 256])
        b = bias_variable([256])
        d_real_conv_2 = lrelu(tf.nn.conv2d(input=d_real_conv_1,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_3'):
        W = weight_variable([4, 4, 256, 512])
        b = bias_variable([512])
        d_real_conv_3 = lrelu(tf.nn.conv2d(input=d_real_conv_2,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('conv_4'):
        W = weight_variable([4, 4, 512, 1024])
        b = bias_variable([1024])
        d_real_conv_4 = lrelu(tf.nn.conv2d(input=d_real_conv_3,
                                           filter=W,
                                           strides=[1, 2, 2, 1],
                                           padding='SAME') + b)
    with tf.variable_scope('logits'):
        W = weight_variable([4, 4, 1024, 1])
        b = bias_variable([1])
        d_real_logits = tf.nn.conv2d(input=d_real_conv_4,
                                     filter=W,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID') + b
    d_real_o = tf.nn.sigmoid(d_real_logits)

# cost functions for D(x) and G(z) respectively
d_real_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_real_logits,
                                    labels=tf.ones([BATCH_SIZE, 1, 1, 1]))
d_fake_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_fake_logits,
                                    labels=tf.zeros([BATCH_SIZE, 1, 1, 1]))
d_loss = tf.reduce_mean(d_real_xentropy) + tf.reduce_mean(d_fake_xentropy)
g_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=d_fake_logits,
                                    labels=tf.ones([BATCH_SIZE, 1, 1, 1]))
g_loss = tf.reduce_mean(g_xentropy)

# Mini-batch SGD optimisers for J for both Networks
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_train_step = tf.train.AdamOptimizer(learning_rate=LR, beta1=.5).minimize(
                            d_loss,
                            var_list=tf.trainable_variables('discriminator'))
    g_train_step = tf.train.AdamOptimizer(learning_rate=LR, beta1=.5).minimize(
                            g_loss,
                            var_list=tf.trainable_variables('generator'))

sess = tf.Session()
init_op = tf.global_variables_initializer()

sess.run(init_op)

for epoch in range(1, EPOCH + 1):
    step = 0
    sess.run(mnist_batch_iter.initializer)
    while True:
        step += 1
        try:
            _, d_loss_score = sess.run(fetches=[d_train_step, d_loss],
                                       feed_dict={is_train: True})
            _, g_loss_score = sess.run(fetches=[g_train_step, g_loss],
                                       feed_dict={is_train: True})
            print("Epoch {0} of {1}, step {2}, "
                  "Discriminator log loss {3:.4f}, "
                  "Generator log loss {4:.4f}".format(
                    epoch, EPOCH, step, d_loss_score, g_loss_score))
        except tf.errors.OutOfRangeError:
            print("Epoch {0} has finished.".format(epoch))
            break
