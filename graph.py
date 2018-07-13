# -*- encoding: utf-8 -*-
import tensorflow as tf

from pipeline import feed
from model import Generator, Discriminator, Loss
from config import NNConfig, g_params, d_params

BATCH_SIZE, LR = NNConfig.BATCH_SIZE, NNConfig.ALPHA

is_train = tf.placeholder_with_default(input=True, shape=[], name='is_train')
g_z = tf.placeholder(shape=[BATCH_SIZE, 1, 1, 100], dtype=tf.float32)
d_real_x, d_real_y = feed

# gaussian noise to improve chance of intersection of D and G.
input_dims = d_real_x.shape
ε = tf.random_normal(input_dims)
ε = 0

# set up D(x + ε)
dx = Discriminator(d_real_x + ε, d_params)
# conditional on class y
dx.given_y(10)
dx.hyperparams[0][1][0][2] = dx._input_tensor.shape[-1]
d_real_logits = dx.build(bn=False)

# set up D(G(z) + ε)
gz = Generator(g_z, g_params)
# conditional on class y
gz.given_y(10)
gz.hyperparams[0][1][0][-1] = gz._input_tensor.shape.as_list()[-1]
gz.hyperparams[-1][1][0][2] += 10
gz.hyperparams[-1][1][2][-1] += 10
g_o = gz.build()
d_fake_logits = Discriminator(g_o + ε, d_params).build(bn=False)

# uniform noise for Gradient Penalty terms
ε_penalty = tf.random_uniform(input_dims, name='epsilon')
x_hat = ε_penalty * d_real_x + (1 - ε_penalty) * g_o
d_penalty_logits = Discriminator(x_hat, d_params).build(bn=False)
derivative, = tf.gradients(d_penalty_logits, [x_hat])

# Wasserstein distance with gradient penalty
d_loss, g_loss = Loss(d_real_logits, d_fake_logits).wasserstein(derivative)

# gradient computation with respect to D and G variables.
d_train_step = tf.train.AdamOptimizer(LR, beta1=0., beta2=.9).minimize(
                        d_loss,
                        var_list=tf.trainable_variables('Discriminator'))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    g_train_step = tf.train.AdamOptimizer(LR, beta1=0., beta2=.9).minimize(
                            g_loss,
                            var_list=tf.trainable_variables('Generator'))
