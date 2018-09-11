#!/usr/bin/env python
"""
app

flask app serving external client calls.
"""
from random import randint

from flask import Flask
from flask import request, abort

from helper import _validate_integer
from serving import feed_serving


app = Flask("mnist_gan")


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        digit = randint(0, 9)
    elif request.method == 'POST':
        raw = request.get_data()
        digit = _validate_integer(raw)

        if digit is None:
            abort(400)

    return feed_serving(digit)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
