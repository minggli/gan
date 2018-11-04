#!/usr/bin/env python
"""
app

flask app serving external client calls.
"""
from random import randint

from flask import Flask
from flask import request, render_template, redirect

from helper import _validate_integer
from serving import grpc_generate, grpc_predict
from config import APP_CONFIG

app = Flask(__name__)
app.config.update(APP_CONFIG)


@app.route('/')
def index():
    return redirect('/generate/')


@app.route('/generate/', defaults={'digit': None}, methods=['GET'])
@app.route('/generate/<int:digit>', methods=['GET'])
def generate(digit):
    if _validate_integer(digit) is None:
        digit = randint(0, 9)
    return grpc_generate(digit)


# TODO Wasserstein loss doesn't discriminate real or fake image like original
# GAN loss function, more work needed to reuse Critic/Discriminator
# to classify generated image. Loss function must include supervise element.
# @app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        f = request.files['file']
        prob = grpc_predict(f)
        return render_template('predict.html', result=prob)
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
