"""
logging

produce grid of generated digits through epochs.

Reference : https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/
"""
import os
from itertools import product

import imageio
import numpy as np
import matplotlib.pyplot as plt


def produce_grid(test_images, num_epoch, path, show=False, save=False,
                 grid_size=5):
    """produce squared grid"""

    if test_images.shape[0] < grid_size**2:
        raise ValueError(
                "not enough samples generated, minimum {0}.".format(grid_size))

    fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    for i, j in product(range(grid_size), range(grid_size)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(grid_size**2):
        i = k // grid_size
        j = k % grid_size
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)
    elif show:
        plt.show()
    else:
        plt.close()

    return plt


def produce_gif(images, path):
    path = os.path.join(path, './generation_animation.gif')
    imageio.mimsave(path, images, fps=5)
