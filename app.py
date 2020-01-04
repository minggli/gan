#!/usr/bin/env python
"""
app

flask app serving external client calls.
"""
from random import randint

from flask import Flask
from flask_restplus import Api, Namespace, Resource

from helper import _validate_integer
from serving import grpc_generate
from config import APP_CONFIG

application = Flask(__name__)
application.config.update(APP_CONFIG)

ns = Namespace('generate', description='Generate images.')


@ns.route('/<int:digit>', defaults={'digit': None})
class Generate(Resource):
    def get(self, digit):
        if _validate_integer(digit) is None:
            digit = randint(0, 9)
        return grpc_generate(digit)


# TODO Wasserstein loss doesn't discriminate real or fake image like original
# GAN loss function, more work needed to reuse Critic/Discriminator
# to classify generated image. Loss function must include supervise element.
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         f = request.files['file']
#         prob = grpc_predict(f)
#         return render_template('predict.html', result=prob)
#     return render_template('predict.html')

api = Api(
    title='Wasserstein GAN',
    version='1.0',
    description='Generate images using GANs')
api.add_namespace(ns, path="/generate")
api.init_app(application)


if __name__ == '__main__':
    application.run(host='0.0.0.0', debug=True)
