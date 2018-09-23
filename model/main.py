import tensorflow as tf
import numpy as np 
import pandas as pd
import os
from tensorflow.python.ops import embedding_ops
import hparams
import utils

import tensorflow.contrib.slim as slim

def conv_net(inputs):
  """Builds the ConvNet from Kelz 2016."""
    with slim.arg_scope(
    [slim.conv2d, slim.fully_connected],
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
    factor=2.0, mode='FAN_AVG', uniform=True)):
    
        n_filters = hparams.n_filters
        Ksize = hparams.Ksize
        max_pool_ksize = hparams.max_pool_ksize
        max_pool_stride = hparams.max_pool_stride
        fc_size = hparams.fc_size


        net = slim.conv2d(inputs, n_filters[0], Ksize, scope='conv1', normalizer_fn=slim.batch_norm)

        net = slim.conv2d(net, n_filters[1], Ksize, scope='conv2', normalizer_fn=slim.batch_norm)
        net = slim.max_pool2d(net, max_pool_ksize, stride=max_pool_stride, scope='pool2')
        net = slim.dropout(net, 0.25, scope='dropout2')

        net = slim.conv2d(net, n_filters[2], Ksize, scope='conv3', normalizer_fn=slim.batch_norm)
        net = slim.max_pool2d(net, max_pool_ksize, stride=max_pool_stride, scope='pool3')
        net = slim.dropout(net, 0.25, scope='dropout3')

        # Flatten while preserving batch and time dimensions.
        dims = tf.shape(net)
        net = tf.reshape(net, (dims[0], dims[1],
                               net.shape[2].value * net.shape[3].value), 'flatten4')

        net = slim.fully_connected(net, fc_size[0], scope='fc5')
        net = slim.dropout(net, 0.5, scope='dropout5')

    return net    


def acoustic_model():
    """Acoustic model that handles all specs for a sequence in one window."""
