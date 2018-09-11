"""
helpers

utility functions and controllers
"""

import numbers
from io import BytesIO
from uuid import uuid4
from functools import wraps

import numpy as np

from PIL import Image
from flask import send_file

from gan.graph import condition_matrice, gaussian_noise


def _try_cast_integer(input):
    try:
        return int(input)
    except (ValueError, TypeError) as e:
        return input


def _validate_integer(input):
    if not isinstance(input, numbers.Number):
        input = _try_cast_integer(input)

    try:
        input = int(round(input, 0))
    except TypeError:
        return None

    if input not in range(10):
        return None

    return input


def produce_inputs(i):
    noise = gaussian_noise(1).astype(np.float32)
    y_gz, y_dx = condition_matrice(np.eye(10)[i])
    return noise, y_dx, y_gz


def _process_image(image_filelike):
    buf = BytesIO(image_filelike.read())
    img = Image.open(buf).convert('L')
    resized = img.resize((64, 64), Image.ANTIALIAS)
    return np.array(resized).reshape(1, 64, 64, 1).astype(np.float32)


def _process_array(array):
    dim = int(len(array) ** .5)
    arr = np.asarray(array).reshape(dim, dim).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    larger = img.resize((dim * 5, dim * 5), Image.ANTIALIAS)
    return larger


def file_response(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        buf = BytesIO()
        array_image = func(*args, **kwargs)
        image = _process_array(array_image)
        image.save(buf, format='png')
        buf.seek(0)
        return send_file(buf,
                         as_attachment=False,
                         attachment_filename=f'{uuid4()}.png',
                         mimetype='image/png')
    return wrapper
