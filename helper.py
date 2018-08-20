"""
helper

utility functions
"""
import numpy as np


def condition_matrice(label, img_size=64):
    """convert one_hot class matrix into 4-D condition matrice where class
    information resides in the last dimension"""
    y_gz = label.reshape(-1, 1, 1, label.shape[-1])
    y_dx = y_gz * np.ones([img_size, img_size, label.shape[-1]])
    return y_gz, y_dx
