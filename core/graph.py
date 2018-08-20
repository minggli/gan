# -*- encoding: utf-8 -*-
import tensorflow as tf

from .model import Generator, Discriminator, Loss


class Graph(object):
    """build graph of tensors and ops."""
    def __init__(self,
                 config,
                 D_parameters,
                 G_parameters,
                 D_fn=Discriminator,
                 G_fn=Generator,
                 Loss_fn=Loss):
        self.n_class, self.lr = config.N_CLASS, config.ALPHA
        self.D_param = D_parameters
        self.G_param = G_parameters
        self.Loss_fn = Loss_fn
        self.D_fn = Discriminator
        self.G_fn = Generator

    def _initialize_ph(self):
        d_real_x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='x')
        g_z = tf.placeholder(tf.float32, [None, 1, 1, 100], name='z')
        is_train = tf.placeholder_with_default(True, [], name='is_train')

        # replace static batch size in g_params to dynamic
        rt_batchsize = tf.shape(g_z)[0]
        for param in self.G_param:
            param[1][2][0] = rt_batchsize

        return [d_real_x, g_z, is_train]

    def build(self):
        self.placeholders = self._initialize_ph()
        d_real_x, g_z, *discard = self.placeholders

        # Discriminator
        dx = self.D_fn(d_real_x, self.D_param).conditional_y(self.n_class)
        d_real_logits = dx.build(bn=False)

        # Generator
        gz = self.G_fn(g_z, self.G_param).conditional_y(self.n_class)
        g_o = gz.build()
        d_fake = self.D_fn(g_o, self.D_param).conditional_y(self.n_class)
        d_fake_logits = d_fake.build(bn=False)

        # Gradient Penalty
        ε_penalty = tf.random_uniform([], name='epsilon')
        x_hat = ε_penalty * d_real_x + (1 - ε_penalty) * g_o
        _ = self.D_fn(x_hat, self.D_param).conditional_y(self.n_class)
        d_penalty_logits = _.build(bn=False)
        derivative, = tf.gradients(d_penalty_logits, [x_hat])

        # Wasserstein distance with gradient penalty
        loss = self.Loss_fn(d_real_logits, d_fake_logits)
        d_loss, g_loss = loss.wasserstein(derivative)

        # gradient computation with respect to variables in D and G.
        d_train_step = tf.train.AdamOptimizer(self.lr,
                                              beta1=0.,
                                              beta2=.9).\
            minimize(d_loss,
                     var_list=tf.trainable_variables('Discriminator'))
        dependent_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(dependent_ops):
            g_train_step = tf.train.AdamOptimizer(self.lr,
                                                  beta1=0.,
                                                  beta2=.9).\
                minimize(g_loss,
                         var_list=tf.trainable_variables('Generator'))

        return (d_train_step, d_loss, g_train_step, g_loss, g_z, g_o, gz, dx,
                d_real_x)
