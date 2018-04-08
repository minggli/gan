"""
config

module keeping some of model parameters
"""


class BaseConfig(object):
    pass


class NNConfig(BaseConfig):
    BATCH_SIZE = 100
    EPOCH = 100
    ALPHA = 1e-4
