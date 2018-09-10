#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

import tensorflow as tf

from tensorflow.saved_model.builder import SavedModelBuilder
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model.signature_def_utils import is_valid_signature
from tensorflow.saved_model.tag_constants import SERVING

from core import Graph
from helper import train
from pipeline import mnist_batch_iter
from config import NNConfig, d_params, g_params

d_train_step, d_loss, g_train_step, g_loss, g_z, g_o, gz, dx, d_real_x, \
    image = Graph(NNConfig, d_params, g_params).build()

y_dx = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(dx.name))
y_gz = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(gz.name))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init_op = tf.global_variables_initializer()

sess.run(init_op)

train_ops = [sess, mnist_batch_iter, d_real_x, g_z, y_dx, y_gz, d_train_step,
             d_loss, g_train_step, g_loss]

train(*train_ops)

generative_signature = predict_signature_def(
        inputs={'noise': g_z, 'y_dx': y_dx, 'y_gz': y_gz},
        outputs={'image': image}
)

assert is_valid_signature(generative_signature)

builder = SavedModelBuilder('./model_binaries/generate_image/0')
builder.add_meta_graph_and_variables(
    sess,
    [SERVING],
    signature_def_map={'generate': generative_signature},
    strip_default_attrs=True)

builder.save()
