import warnings
import os
import sys
import gzip
import requests

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import NNConfig, DataConfig

warnings.simplefilter('ignore')

__all__ = ['mnist_batch_iter']


BASEURL = DataConfig.DIGIT
LOCAL_PATH = DataConfig.LOCAL_PATH

TRAIN_IMAGE = 'train-images-idx3-ubyte.gz'
TRAIN_LABEL = 'train-labels-idx1-ubyte.gz'
TEST_IMAGE = 't10k-images-idx3-ubyte.gz'
TEST_LABEL = 't10k-labels-idx1-ubyte.gz'

BATCH_SIZE = NNConfig.BATCH_SIZE


def traverse(root, ext=None):
    """recusively traverse files with specified extension from root."""
    files = []
    for entry in os.scandir(root):
        if entry.is_dir():
            files.extend(traverse(entry.path, ext))
        elif ext is None or entry.name.endswith(ext):
            files.append(entry.path)
    return files


def maybe_download(local, url=BASEURL, chunk_size=4096):
    if not local.exists():
        filename = os.path.basename(local)
        sys.stdout.write("Downloading {}".format(filename) + '\n')
        sys.stdout.flush()
        remote_file = os.path.join(url, filename)
        r = requests.get(remote_file, stream=True)
        kilobytes = int(int(r.headers['Content-Length']) / 1024)
        with open(local, 'wb') as f:
            with tqdm(total=kilobytes, unit='KB') as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(int(chunk_size / 1024))

    return open(local, 'rb')


def parse_image(filelike, magic=2051):
    """parse gzip compressed IDX file format for images."""
    with gzip.GzipFile(fileobj=filelike, mode='rb') as buf:
        assert magic == int(buf.read(4).hex(), 16)
        number_samples = int(buf.read(4).hex(), 16)
        length = int(buf.read(4).hex(), 16)
        width = int(buf.read(4).hex(), 16)
        array = np.frombuffer(buf.read(), dtype=np.uint8)
    return array.reshape(number_samples, length, width, 1)


def parse_label(filelike, magic=2049):
    """parse gzip compressed IDX file format for one-hot encoded labels."""
    with gzip.GzipFile(fileobj=filelike, mode='rb') as buf:
        assert magic == int(buf.read(4).hex(), 16)
        buf.read(4)
        array = np.frombuffer(buf.read(), dtype=np.uint8)
    return np.eye(10)[array]


if not LOCAL_PATH.exists():
    LOCAL_PATH.mkdir()

X_train = parse_image(maybe_download(LOCAL_PATH / TRAIN_IMAGE))
X_label = parse_label(maybe_download(LOCAL_PATH / TRAIN_LABEL))

mnist = tf.data.Dataset.from_tensor_slices((X_train, X_label))
mnist = mnist.map(lambda x, y: (tf.image.resize_images(x, [64, 64]), y))
# mean centering so (-1, 1)
mnist = mnist.map(lambda x, y: ((x / 255 - 0.5) / 0.5, tf.cast(y, tf.float32)))

mnist_batch = mnist.shuffle(1000).batch(BATCH_SIZE)
mnist_batch_iter = mnist_batch.make_initializable_iterator()
