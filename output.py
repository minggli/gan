"""
output

produce grid of generated digits through epochs.

Reference : https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/
"""

import io
import os
from itertools import product
import imageio
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def produce_grid(test_images, num_epoch, path='./results', show=False,
                 save=False, grid_size=5):
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
        test_image = np.reshape(test_images[k], (64, 64)) * .5 + .5
        test_image *= 255
        ax[i, j].imshow(test_image, cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(os.path.join(path, 'epoch_{0}.png'.format(num_epoch)))
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()

    with io.BytesIO() as buf:
        fig.canvas.print_png(buf)
        size = fig.canvas.get_width_height()
        img = Image.open(buf)
        grayscale_img = img.convert('L')

    return np.fromstring(grayscale_img.tobytes(), np.uint8).reshape(size)


def produce_gif(images, path):
    path = os.path.join(path, './generation_animation.gif')
    imageio.mimsave(path, images, fps=5)
