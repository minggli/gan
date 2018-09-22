"""
config

module keeping some of model parameters, further refactoring pending.
"""

import tensorflow as tf

tf.app.flags.DEFINE_integer('epochs', 20,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_integer('model_version', 0, 'version number of the model.')

FLAGS = tf.app.flags.FLAGS

APP_CONFIG = {
    'SECRET_KEY': 'e07a29ed46559202147d177680570f'
                  '331fa4e1e3e0570d95f591d2cab9f6c49e',
    'SESSION_COOKIE_NAME': 'session',
    'DEBUG': False
}

d_params = [
    ('conv_1', [[4, 4, 1, 128], [128]], [1, 2, 2, 1], 'SAME'),
    ('conv_2', [[4, 4, 128, 256], [256]], [1, 2, 2, 1], 'SAME'),
    ('conv_3', [[4, 4, 256, 512], [512]], [1, 2, 2, 1], 'SAME'),
    ('conv_4', [[4, 4, 512, 1024], [1024]], [1, 2, 2, 1], 'SAME'),
    ('logits', [[4, 4, 1024, 1], [1]], [1, 1, 1, 1], 'VALID')
]

g_params = [
    ('deconv_1', [[4, 4, 1024, 100], [1024], [64, 4, 4, 1024]],
     [1, 1, 1, 1], 'VALID'),
    ('deconv_2', [[4, 4, 512, 1024], [512], [64, 8, 8, 512]],
     [1, 2, 2, 1], 'SAME'),
    ('deconv_3', [[4, 4, 256, 512], [256], [64, 16, 16, 256]],
     [1, 2, 2, 1], 'SAME'),
    ('deconv_4', [[4, 4, 128, 256], [128], [64, 32, 32, 128]],
     [1, 2, 2, 1], 'SAME'),
    ('logits', [[4, 4, 1, 128], [1], [64, 64, 64, 1]],
     [1, 2, 2, 1], 'SAME')
]


class BaseConfig(object):
    pass


class DataConfig(BaseConfig):
    # Fashion MNIST
    FASHION = ("https://github.com/zalandoresearch/"
               "fashion-mnist/raw/master/data/fashion/")
    # handwriting digits
    DIGIT = "http://yann.lecun.com/exdb/mnist/"
    LOCAL_PATH = './data'


class NNConfig(BaseConfig):
    BATCH_SIZE = 64
    EPOCH = FLAGS.epochs
    ALPHA = 1e-4
    N_CRITIC = 3
    N_CLASS = 10


class ServingConfig(BaseConfig):
    MODEL_BASE_PATH = './bin'
    MODEL_NAME = 'model'
    MODEL_VER = str(FLAGS.model_version)
    SERVING_DOMAIN = 'serving:8500'
