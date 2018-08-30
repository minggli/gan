#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

import tensorflow as tf

from tensorflow.saved_model.builder import SavedModelBuilder
from tensorflow.saved_model.utils import build_tensor_info
from tensorflow.saved_model.signature_def_utils import build_signature_def
from tensorflow.saved_model.signature_def_utils import is_valid_signature
from tensorflow.saved_model.signature_constants import PREDICT_METHOD_NAME

from core import Graph
from helper import train
from pipeline import mnist_batch_iter
from config import NNConfig, d_params, g_params

d_train_step, d_loss, g_train_step, g_loss, g_z, g_o, gz, dx, d_real_x, \
    image = Graph(NNConfig, d_params, g_params).build()

y_dx = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(dx.name))
y_gz = tf.get_default_graph().get_tensor_by_name('y_{0}:0'.format(gz.name))

global_saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init_op = tf.global_variables_initializer()

sess.run(init_op)

train_ops = [sess, mnist_batch_iter, d_real_x, g_z, y_dx, y_gz, d_train_step,
             d_loss, g_train_step, g_loss]

train(*train_ops)

tensor_info_g_z = build_tensor_info(g_z)
tensor_info_y_dx = build_tensor_info(y_dx)
tensor_info_y_gz = build_tensor_info(y_gz)
tensor_info_image = build_tensor_info(image)

default_signature = build_signature_def(
                        inputs={'noise': tensor_info_g_z,
                                'y_dx': tensor_info_y_dx,
                                'y_gz': tensor_info_y_gz},
                        outputs={'image': tensor_info_image},
                        method_name=PREDICT_METHOD_NAME)

assert is_valid_signature(default_signature)

builder = SavedModelBuilder('./model_binaries')
builder.add_meta_graph_and_variables(
    sess,
    'SERVING',
    signature_def_map={'generate_image':
                       default_signature},
    strip_default_attrs=True)

builder.save()
