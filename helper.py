"""
helpers

utility functions and controllers
"""
import numpy as np
import tensorflow as tf

from config import NNConfig


def condition_matrice(label, img_size=64):
    """convert one_hot class matrix into 4-D condition matrice where class
    information resides in the last dimension"""
    y_gz = label.reshape(-1, 1, 1, label.shape[-1])
    y_dx = y_gz * np.ones([img_size, img_size, label.shape[-1]])
    return y_gz, y_dx


def gaussian_noise(batch_size):
    return np.random.normal(0, 1, size=[batch_size, 1, 1, 100])


def train(sess, *args, config=NNConfig):

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
