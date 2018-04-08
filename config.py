"""
config

module keeping some of model parameters
"""


class BaseConfig(object):
    pass


class NNConfig(BaseConfig):
    BATCH_SIZE = 50
    EPOCH = 100
    ALPHA = 1e-4
