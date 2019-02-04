from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np 
import tensorflow as tf


import tensorflow as tf
import tensorflow.contrib.slim as slim

from magenta.common import flatten_maybe_padded_sequences
from magenta.common import tf_utils





def conv_net_kelz(inputs):
  """Builds the ConvNet from Kelz 2016."""
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer(
          factor=2.0, mode='FAN_AVG', uniform=True)):
    net = slim.conv2d(
        inputs, 32, [3, 3], scope='conv1', normalizer_fn=slim.batch_norm)

    net = slim.conv2d(
        net, 32, [3, 3], scope='conv2', normalizer_fn=slim.batch_norm)
    net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool2')
    net = slim.dropout(net, 0.25, scope='dropout2')

    net = slim.conv2d(
        net, 64, [3, 3], scope='conv3', normalizer_fn=slim.batch_norm)
    net = slim.max_pool2d(net, [1, 2], stride=[1, 2], scope='pool3')
    net = slim.dropout(net, 0.25, scope='dropout3')

    # Flatten while preserving batch and time dimensions.
    dims = tf.shape(net)
    net = tf.reshape(net, (dims[0], dims[1],
                           net.shape[2].value * net.shape[3].value), 'flatten4')

    net = slim.fully_connected(net, 512, scope='fc5')
    net = slim.dropout(net, 0.5, scope='dropout5')

    return net



def acoustic_model(inputs, hparams, lstm_units, lengths):
  """Acoustic model that handles all specs for a sequence in one window."""
  conv_output = conv_net_kelz(inputs)

  if lstm_units:
    rnn_cell_fw = tf.contrib.rnn.LSTMBlockCell(lstm_units)
    if hparams.onset_bidirectional:
      rnn_cell_bw = tf.contrib.rnn.LSTMBlockCell(lstm_units)
      outputs, unused_output_states = tf.nn.bidirectional_dynamic_rnn(
          rnn_cell_fw,
          rnn_cell_bw,
          inputs=conv_output,
          sequence_length=lengths,
          dtype=tf.float32)
      combined_outputs = tf.concat(outputs, 2)
    else:
      combined_outputs, unused_output_states = tf.nn.dynamic_rnn(
          rnn_cell_fw,
          inputs=conv_output,
          sequence_length=lengths,
          dtype=tf.float32)

    return combined_outputs
  else:
    return conv_output

def onset_detector_model(spec, onset_labels):

  onset_outputs = acoustic_model(spec, hparams, lstm_units=hparams.onset_lstm_units, lengths=lengths)
  

  onset_probs = slim.fully_connected(
      onset_outputs,
      constants.num_drums,
      activation_fn=tf.sigmoid,
      scope='onset_probs')

  # onset_probs_flat is used during inference.
  onset_probs_flat = flatten_maybe_padded_sequences(onset_probs, lengths)
  onset_labels_flat = flatten_maybe_padded_sequences(onset_labels, lengths)  #onset_labels are the ground truth labels
  tf.identity(onset_probs_flat, name='onset_probs_flat')
  tf.identity(onset_labels_flat, name='onset_labels_flat')
  tf.identity(
      tf.cast(tf.greater_equal(onset_probs_flat, .5), tf.float32),
      name='onset_predictions_flat')

  onset_losses = tf_utils.log_loss(onset_labels_flat, onset_probs_flat)
  tf.losses.add_loss(tf.reduce_mean(onset_losses))
  
  return onset_losses 
  #losses['onset'] = onset_losses


def note_detector_model(spec, frame_labels, lengths, onset_probs ,onset_outputs):

  if not hparams.share_conv_features:
    # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
    activation_outputs = acoustic_model(
        spec, hparams, lstm_units=hparams.frame_lstm_units, lengths=lengths)
    activation_probs = slim.fully_connected(
        activation_outputs,
        constants.num_drums,
        activation_fn=tf.sigmoid,
        scope='activation_probs')
  else:
    activation_probs = slim.fully_connected(
        onset_outputs,
        constants.num_drums,
        activation_fn=tf.sigmoid,
        scope='activation_probs')

  combined_probs = tf.concat([
      tf.stop_gradient(onset_probs)
      if hparams.stop_onset_gradient else onset_probs,
      tf.stop_gradient(activation_probs)
      if hparams.stop_activation_gradient else activation_probs
  ], 2)

  if hparams.combined_lstm_units > 0:
    rnn_cell_fw = tf.contrib.rnn.LSTMBlockCell(hparams.combined_lstm_units)
    if hparams.frame_bidirectional:
      rnn_cell_bw = tf.contrib.rnn.LSTMBlockCell(
          hparams.combined_lstm_units)
      outputs, unused_output_states = tf.nn.bidirectional_dynamic_rnn(
          rnn_cell_fw, rnn_cell_bw, inputs=combined_probs, dtype=tf.float32)
      combined_outputs = tf.concat(outputs, 2)
    else:
      combined_outputs, unused_output_states = tf.nn.dynamic_rnn(
          rnn_cell_fw, inputs=combined_probs, dtype=tf.float32)
  else:
    combined_outputs = combined_probs

  frame_probs = slim.fully_connected(
      combined_outputs,
      constants.num_drums,
      activation_fn=tf.sigmoid,
      scope='frame_probs')


  frame_labels_flat = flatten_maybe_padded_sequences(frame_labels, lengths)
  frame_probs_flat = flatten_maybe_padded_sequences(frame_probs, lengths)
  tf.identity(frame_probs_flat, name='frame_probs_flat')
  frame_label_weights_flat = flatten_maybe_padded_sequences(
      frame_label_weights, lengths)
  frame_losses = tf_utils.log_loss(
      frame_labels_flat,
      frame_probs_flat,
      weights=frame_label_weights_flat
      if hparams.weight_frame_and_activation_loss else None)
  tf.losses.add_loss(tf.reduce_mean(frame_losses))
  losses['frame'] = frame_losses


  predictions_flat = tf.cast(tf.greater_equal(frame_probs_flat, .5), tf.float32)


