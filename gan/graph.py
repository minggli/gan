# -*- encoding: utf-8 -*-
"""
graph

abstraction to instantiate computation graph and dependencies.
"""
import numpy as np
import tensorflow as tf

from .model import Generator, Discriminator, Loss


def gaussian_noise(batch_size):
    return np.random.normal(0, 1, size=[batch_size, 1, 1, 100])


def condition_matrice(label, img_size=64):
    """convert one_hot class matrix into 4-D condition matrice where class
    information resides in the last dimension"""
    y_gz = label.reshape(-1, 1, 1, label.shape[-1])
    y_dx = y_gz * np.ones([img_size, img_size, label.shape[-1]])
    return y_gz.astype(np.float32), y_dx.astype(np.float32)


def train(sess, *args, config=None):

    data_iterator, d_real_x, g_z, y_dx, y_gz, d_train_step, d_loss,\
        g_train_step, g_loss = args

    feed = data_iterator.get_next()
    num_epochs = config.EPOCH
    # number of Discriminator updates for each Generator update
    num_epochs_per_g = config.N_CRITIC
    is_train = tf.get_default_graph().get_tensor_by_name('is_train:0')

    for epoch in range(1, config.EPOCH + 1):
        try:
            step = 0
            sess.run(data_iterator.initializer)
            while True:
                try:
                    # update D(x) n times
                    for i in range(num_epochs_per_g):
                        step += 1
                        X, y = sess.run(feed)
                        y_gz_fill, y_dx_fill = condition_matrice(y)
                        dictionary = {d_real_x: X,
                                      g_z: gaussian_noise(X.shape[0]),
                                      y_dx: y_dx_fill,
                                      y_gz: y_gz_fill,
                                      is_train: True}
                        _, d_loss_score = sess.run([d_train_step, d_loss],
                                                   feed_dict=dictionary)
                        print("Epoch {0} of {1}, step {2} "
                              "Discriminator log loss {3:.4f}".format(
                                epoch, num_epochs, step, d_loss_score))

                    # update G(z)
                    step += 1
                    dictionary = {g_z: gaussian_noise(X.shape[0]),
                                  y_dx: y_dx_fill,
                                  y_gz: y_gz_fill,
                                  is_train: True}

                    _, g_loss_score = sess.run([g_train_step, g_loss],
                                               feed_dict=dictionary)
                    print("Epoch {0} of {1}, step {2}"
                          " Generator log loss {3:.4f}".format(
                            epoch, num_epochs, step, g_loss_score))

                except tf.errors.OutOfRangeError:
                    print("Epoch {0} has finished.".format(epoch))
                    break

        except KeyboardInterrupt:
            print("Ending Training after {0} epochs.".format(epoch))
            break


class Graph(object):
    """build graph of tensors and ops."""
    def __init__(self,
                 config,
                 D_param,
                 G_param,
                 D_fn=Discriminator,
                 G_fn=Generator,
                 Loss_fn=Loss):
        self.n_class, self.lr = config.N_CLASS, config.ALPHA
        self.D_param = D_param
        self.G_param = G_param
        self.Loss_fn = Loss_fn
        self.D_fn = D_fn
        self.G_fn = G_fn

    def _initialize_ph(self):
        d_real_x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='x')
        g_z = tf.placeholder(tf.float32, [None, 1, 1, 100], name='z')
        is_train = tf.placeholder_with_default(False, [], name='is_train')

        # replace static batch size in g_params to dynamic
        rt_batchsize = tf.shape(g_z)[0]
        for param in self.G_param:
            param[1][2][0] = rt_batchsize

        return [d_real_x, g_z, is_train]

    def build(self):
        """build computtion graph for Conditional WGAN-GP."""
        self.placeholders = self._initialize_ph()
        d_real_x, g_z, *discard = self.placeholders

        # Discriminator
        dx = self.D_fn(d_real_x, self.D_param).conditional_y(self.n_class)
        d_real_logits = dx.build(bn=False)

        # conditional probability, unused during training
        p_given_y = tf.nn.sigmoid(d_real_logits)

        # Generator
        gz = self.G_fn(g_z, self.G_param).conditional_y(self.n_class)
        g_o = gz.build()
        d_fake = self.D_fn(g_o, self.D_param).conditional_y(self.n_class)
        d_fake_logits = d_fake.build(bn=False)

        # image output tensor, unused during training
        image = 255 * (g_o * .5 + .5)

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
                d_real_x, image, p_given_y)
