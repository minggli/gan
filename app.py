#!/usr/bin/env python
"""
app

flask app serving external client calls.
"""
from random import randint

from flask import Flask
from flask import (request, abort, render_template, flash, redirect,
                   make_response)

from helper import _validate_integer
from serving import gRPC_generate, gRPC_predict


app = Flask("mnist_gan")
SECRET_KEY = 'e07a29ed46559202147d177680570f331fa4e1e3e0570d95f591d2cab9f6c49e'
app.secret_key = SECRET_KEY


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        digit = randint(0, 9)
    elif request.method == 'POST':
        raw = request.get_data()
        digit = _validate_integer(raw)

        if digit is None:
            abort(400)

    return gRPC_generate(digit)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('no image uploaded.')
            return redirect(request.url)
        f = request.files['file']
        prob, = gRPC_predict(f)
        return make_response(f'{prob:.4f}')
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
