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
from serving import grpc_generate, grpc_predict
from config import APP_CONFIG

app = Flask(__name__)
app.config.update(APP_CONFIG)


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        digit = randint(0, 9)
    elif request.method == 'POST':
        raw = request.get_data()
        digit = _validate_integer(raw)

        if digit is None:
            abort(400)

    return grpc_generate(digit)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('no image uploaded.')
            return redirect(request.url)
        f = request.files['file']
        prob = grpc_predict(f)
        print(prob)
        return make_response(f'{prob:.4f}')
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
