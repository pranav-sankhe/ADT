import tensorflow as tf
import numpy as np 
import pandas as pd
import os
from tensorflow.python.ops import embedding_ops
import hparams


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, hparams.DTYPE, tf.truncated_normal_initializer(stddev=hparams.init_std)) # function to intilaize weights for each layer

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, hparams.DTYPE, tf.constant_initializer(0.1, dtype=tf.float32)) # fucntion to intiliaze bias vector for each layer


def conv_layer(prev_layer, in_filters, out_filters, Ksize, stride, poolTrue, name_scope):

    
    with tf.variable_scope(name_scope) as scope: # name of the block  
        
        kernel = _weight_variable('weights', [Ksize, Ksize, in_filters, out_filters]) # (kernels = filters as defined in TF doc). kernel size = 5 (5*5*5) 
        conv = tf.nn.conv3d(prev_layer, kernel, [stride, stride, stride, stride], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases) # define biases for conv1 
        conv1 = tf.nn.relu(bias, name=scope.name) # define the activation for conv1 
        prev_layer = conv1                                                                              
    if poolTrue:    
        pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')        
        norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
        layer = norm1

    return layer    


def fully_connected(size, prev_layer, name_scope):
    with tf.variable_scope(name_scope) as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, size])
        biases = _bias_variable('biases', [size])
        output = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
    return output




def _get_infer_maximum_iterations(sequence_length):
    """Maximum decoding steps at inference time."""
      # TODO(thangluong): add decoding_length_factor flag
    decoding_length_factor = 2.0
    max_encoder_length = tf.reduce_max(sequence_length)
    maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations



def _compute_loss(target_output, logits, batch_size):
    """Compute optimization loss."""
    if hparams.TIME_MAJOR:
        target_output = tf.transpose(target_output)
    max_time = get_max_time(target_output)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
        batch_size, max_time, dtype=logits.dtype)
    if hparams.TIME_MAJOR:
        target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)
    return loss

def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary, gradient_norm








def batch_norm(inputs, training):
    return tf.layers.batch_normalization(
          inputs=inputs, axis=4,
          momentum=hparams._BATCH_NORM_DECAY, epsilon=hparams._BATCH_NORM_EPSILON, center=True,
          scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [ [0, 0], [0,0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0] ])

    return padded_inputs




def conv3d_fixed_padding(inputs, filters, kernel_size, strides):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    return tf.layers.conv3d(
                          inputs,
                          filters,
                          kernel_size,
                          strides=(1, 1, 1),
                          padding=('SAME' if strides == 1 else 'VALID'),
                          use_bias=False,
                          kernel_initializer=tf.variance_scaling_initializer(),
                          )

################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides):

    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training)

    inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1)
    inputs = batch_norm(inputs, training)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,strides):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training)

    inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1)
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    inputs = conv3d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
    inputs = batch_norm(inputs, training)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

def _building_block_v2(inputs, filters, training, projection_shortcut, strides):

    shortcut = inputs
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1)

    return inputs + shortcut


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,strides):
    shortcut = inputs
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1)

    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)
    inputs = conv3d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)

    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)
    inputs = conv3d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

    return inputs + shortcut



def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name):
# Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv3d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)
