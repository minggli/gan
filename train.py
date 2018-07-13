import tensorflow as tf

from pipeline import mnist_batch_iter
from output import produce_grid, produce_gif
from graph import d_train_step, d_loss, g_train_step, g_loss, g_x, g_o, g
from graph import is_train
from config import NNConfig

N_CRITIC, EPOCH = NNConfig.N_CRITIC, NNConfig.EPOCH

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
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
