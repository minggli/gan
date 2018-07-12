"""
config

module keeping some of model parameters
"""


class BaseConfig(object):
    DATA_URL = ('https://storage.googleapis.com/cvdf-datasets/mnist/'
                'train-images-idx3-ubyte.gz')
    DATA = './data/train-images-idx3-ubyte.gz'


class NNConfig(BaseConfig):
    BATCH_SIZE = 64
    EPOCH = 20
    ALPHA = 1e-4
    N_CRITIC = 3


d_params = [
    ('conv_1', [[4, 4, 1, 128], [128]], [1, 2, 2, 1], 'SAME'),
    ('conv_2', [[4, 4, 128, 256], [256]], [1, 2, 2, 1], 'SAME'),
    ('conv_3', [[4, 4, 256, 512], [512]], [1, 2, 2, 1], 'SAME'),
    ('conv_4', [[4, 4, 512, 1024], [1024]], [1, 2, 2, 1], 'SAME'),
    ('logits', [[4, 4, 1024, 1], [1]], [1, 1, 1, 1], 'VALID')
]

BATCH_SIZE = NNConfig.BATCH_SIZE

g_params = [
    ('deconv_1', [[4, 4, 1024, 100], [1024], [BATCH_SIZE, 4, 4, 1024]],
     [1, 1, 1, 1], 'VALID'),
    ('deconv_2', [[4, 4, 512, 1024], [512], [BATCH_SIZE, 8, 8, 512]],
     [1, 2, 2, 1], 'SAME'),
    ('deconv_3', [[4, 4, 256, 512], [256], [BATCH_SIZE, 16, 16, 256]],
     [1, 2, 2, 1], 'SAME'),
    ('deconv_4', [[4, 4, 128, 256], [128], [BATCH_SIZE, 32, 32, 128]],
     [1, 2, 2, 1], 'SAME'),
    ('logits', [[4, 4, 1, 128], [1], [BATCH_SIZE, 64, 64, 1]],
     [1, 2, 2, 1], 'SAME')
]
