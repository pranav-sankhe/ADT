from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np 
import tensorflow as tf


import tensorflow as tf
import tensorflow.contrib.slim as slim
import hparams

def log_loss(labels, predictions, epsilon=1e-7, scope=None, weights=None):
  """Calculate log losses.

  Same as tf.losses.log_loss except that this returns the individual losses
  instead of passing them into compute_weighted_loss and returning their
  weighted mean. This is useful for eval jobs that report the mean loss. By
  returning individual losses, that mean loss can be the same regardless of
  batch size.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    weights: Weights to apply to labels.

  Returns:
    A `Tensor` representing the loss values.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels`.
  """
  with tf.name_scope(scope, "log_loss", (predictions, labels)) as scope:
    predictions = tf.to_float(predictions)
    labels = tf.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = -tf.multiply(labels, tf.log(predictions + epsilon)) - tf.multiply(
        (1 - labels), tf.log(1 - predictions + epsilon))
    if weights is not None:
      losses = tf.multiply(losses, weights)

    return losses


def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
  """Flattens the batch of sequences, removing padding (if applicable).

  Args:
    maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
        sized `[N, M, ...]` where M = max(lengths).
    lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
        padding.

  Returns:
     flatten_maybe_padded_sequences: The flattened sequence tensor, sized
         `[sum(lengths), ...]`.
  """
  def flatten_unpadded_sequences():
    # The sequences are equal length, so we should just flatten over the first
    # two dimensions.
    return tf.reshape(maybe_padded_sequences,
                      [-1] + maybe_padded_sequences.shape.as_list()[2:])

  if lengths is None:
    return flatten_unpadded_sequences()

  def flatten_padded_sequences():
    indices = tf.where(tf.sequence_mask(lengths))
    return tf.gather_nd(maybe_padded_sequences, indices)

  return tf.cond(
      tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
      flatten_unpadded_sequences,
      flatten_padded_sequences)



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

def f1_score(precision, recall):
  """Creates an op for calculating the F1 score.

  Args:
    precision: A tensor representing precision.
    recall: A tensor representing recall.

  Returns:
    A tensor with the result of the F1 calculation.
  """
  return tf.where(
      tf.greater(precision + recall, 0), 2 * (
          (precision * recall) / (precision + recall)), 0)

def accuracy_without_true_negatives(true_positives, false_positives,
                                    false_negatives):
  """Creates an op for calculating accuracy without true negatives.

  Args:
    true_positives: A tensor representing true_positives.
    false_positives: A tensor representing false_positives.
    false_negatives: A tensor representing false_negatives.

  Returns:
    A tensor with the result of the calculation.
  """
  return tf.where(
      tf.greater(true_positives + false_positives + false_negatives, 0),
      true_positives / (true_positives + false_positives + false_negatives), 0)


def frame_metrics(frame_labels, frame_predictions):
  """Calculate frame-based metrics."""
  frame_labels_bool = tf.cast(frame_labels, tf.bool)
  frame_predictions_bool = tf.cast(frame_predictions, tf.bool)

  frame_true_positives = tf.reduce_sum(tf.to_float(tf.logical_and(
      tf.equal(frame_labels_bool, True),
      tf.equal(frame_predictions_bool, True))))
  frame_false_positives = tf.reduce_sum(tf.to_float(tf.logical_and(
      tf.equal(frame_labels_bool, False),
      tf.equal(frame_predictions_bool, True))))
  frame_false_negatives = tf.reduce_sum(tf.to_float(tf.logical_and(
      tf.equal(frame_labels_bool, True),
      tf.equal(frame_predictions_bool, False))))
  frame_accuracy = tf.reduce_mean(tf.to_float(
      tf.equal(frame_labels_bool, frame_predictions_bool)))

  frame_precision = tf.where(
      tf.greater(frame_true_positives + frame_false_positives, 0),
      tf.div(frame_true_positives,
             frame_true_positives + frame_false_positives),
      0)
  frame_recall = tf.where(
      tf.greater(frame_true_positives + frame_false_negatives, 0),
      tf.div(frame_true_positives,
             frame_true_positives + frame_false_negatives),
      0)
  frame_f1_score = f1_score(frame_precision, frame_recall)
  frame_accuracy_without_true_negatives = accuracy_without_true_negatives(
      frame_true_positives, frame_false_positives, frame_false_negatives)

  return {
      'true_positives': frame_true_positives,
      'false_positives': frame_false_positives,
      'false_negatives': frame_false_negatives,
      'accuracy': frame_accuracy,
      'accuracy_without_true_negatives': frame_accuracy_without_true_negatives,
      'precision': frame_precision,
      'recall': frame_recall,
      'f1_score': frame_f1_score,
  }

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

# def onset_detector_model(spec, onset_labels, lengths):

#   onset_outputs = acoustic_model(spec, hparams, lstm_units=hparams.onset_lstm_units, lengths=lengths)
  

#   onset_probs = slim.fully_connected(
#       onset_outputs,
#       hparams.num_bols,
#       activation_fn=tf.sigmoid,
#       scope='onset_probs')

#   # onset_probs_flat is used during inference.
#   onset_probs_flat = flatten_maybe_padded_sequences(onset_probs, lengths)
#   onset_labels_flat = flatten_maybe_padded_sequences(onset_labels, lengths)  #onset_labels are the ground truth labels
#   tf.identity(onset_probs_flat, name='onset_probs_flat')
#   tf.identity(onset_labels_flat, name='onset_labels_flat')
#   tf.identity(
#       tf.cast(tf.greater_equal(onset_probs_flat, .5), tf.float32),
#       name='onset_predictions_flat')
  
#   onset_losses = log_loss(onset_labels_flat, onset_probs_flat)
#   tf.losses.add_loss(tf.reduce_mean(onset_losses))
  
#   return onset_outputs, onset_probs, onset_losses 
#   #losses['onset'] = onset_losses


# def note_detector_model(spec, frame_labels, lengths, onset_probs ,onset_outputs):

#   if not hparams.share_conv_features:
#     # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
#     activation_outputs = acoustic_model(
#         spec, hparams, lstm_units=hparams.frame_lstm_units, lengths=lengths)
#     activation_probs = slim.fully_connected(
#         activation_outputs,
#         constants.num_drums,
#         activation_fn=tf.sigmoid,
#         scope='activation_probs')
#   else:
#     activation_probs = slim.fully_connected(
#         onset_outputs,
#         constants.num_drums,
#         activation_fn=tf.sigmoid,
#         scope='activation_probs')

#   combined_probs = tf.concat([
#       tf.stop_gradient(onset_probs)
#       if hparams.stop_onset_gradient else onset_probs,
#       tf.stop_gradient(activation_probs)
#       if hparams.stop_activation_gradient else activation_probs
#   ], 2)

#   if hparams.combined_lstm_units > 0:
#     rnn_cell_fw = tf.contrib.rnn.LSTMBlockCell(hparams.combined_lstm_units)
#     if hparams.frame_bidirectional:
#       rnn_cell_bw = tf.contrib.rnn.LSTMBlockCell(
#           hparams.combined_lstm_units)
#       outputs, unused_output_states = tf.nn.bidirectional_dynamic_rnn(
#           rnn_cell_fw, rnn_cell_bw, inputs=combined_probs, dtype=tf.float32)
#       combined_outputs = tf.concat(outputs, 2)
#     else:
#       combined_outputs, unused_output_states = tf.nn.dynamic_rnn(
#           rnn_cell_fw, inputs=combined_probs, dtype=tf.float32)
#   else:
#     combined_outputs = combined_probs

#   frame_probs = slim.fully_connected(
#       combined_outputs,
#       constants.num_drums,
#       activation_fn=tf.sigmoid,
#       scope='frame_probs')


#   frame_labels_flat = flatten_maybe_padded_sequences(frame_labels, lengths)
#   frame_probs_flat = flatten_maybe_padded_sequences(frame_probs, lengths)
#   tf.identity(frame_probs_flat, name='frame_probs_flat')
#   frame_label_weights_flat = flatten_maybe_padded_sequences(
#       frame_label_weights, lengths)
#   frame_losses = log_loss(
#       frame_labels_flat,
#       frame_probs_flat,
#       weights=frame_label_weights_flat
#       if hparams.weight_frame_and_activation_loss else None)
#   tf.losses.add_loss(tf.reduce_mean(frame_losses))
#   losses['frame'] = frame_losses


#   predictions_flat = tf.cast(tf.greater_equal(frame_probs_flat, .5), tf.float32)



#   # Creates a pianoroll labels in red and probs in green [minibatch, 88]
#   images = {}
#   onset_pianorolls = tf.concat(
#       [
#           onset_labels[:, :, :, tf.newaxis], onset_probs[:, :, :, tf.newaxis],
#           tf.zeros(tf.shape(onset_labels))[:, :, :, tf.newaxis]
#       ],
#       axis=3)
#   images['OnsetPianorolls'] = onset_pianorolls
#   activation_pianorolls = tf.concat(
#       [
#           frame_labels[:, :, :, tf.newaxis], frame_probs[:, :, :, tf.newaxis],
#           tf.zeros(tf.shape(frame_labels))[:, :, :, tf.newaxis]
#       ],
#       axis=3)
#   images['ActivationPianorolls'] = activation_pianorolls

#   return frame_losses, predictions_flat, frame_labels_flat


def get_model(spec, onset_labels, frame_labels, lengths):

  if hparams.stop_activation_gradient and not hparams.activation_loss:
    raise ValueError(
        'If stop_activation_gradient is true, activation_loss must be true.')

  losses = {}
  with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=True):
    with tf.variable_scope('onsets'):
      onset_outputs = acoustic_model(
          spec, hparams, lstm_units=hparams.onset_lstm_units, lengths=lengths)
      onset_probs = slim.fully_connected(
          onset_outputs,
          hparams.num_bols,
          activation_fn=tf.sigmoid,
          scope='onset_probs')

      # onset_probs_flat is used during inference.
      
      
      onset_probs_flat = flatten_maybe_padded_sequences(onset_probs, lengths)
      onset_labels_flat = flatten_maybe_padded_sequences(onset_labels, lengths)

      tf.identity(onset_probs_flat, name='onset_probs_flat')
      tf.identity(onset_labels_flat, name='onset_labels_flat')
      tf.identity(
          tf.cast(tf.greater_equal(onset_probs_flat, .5), tf.float32),
          name='onset_predictions_flat')
      
      onset_losses = log_loss(onset_labels_flat, onset_probs_flat)
      tf.losses.add_loss(tf.reduce_mean(onset_losses))
      losses['onset'] = onset_losses

    # with tf.variable_scope('velocity'):
    #   # TODO(eriche): this is broken when hparams.velocity_lstm_units > 0
    #   velocity_outputs = acoustic_model(
    #       spec,
    #       hparams,
    #       lstm_units=hparams.velocity_lstm_units,
    #       lengths=lengths)
    #   velocity_values = slim.fully_connected(
    #       velocity_outputs,
    #       hparams.num_bols,
    #       activation_fn=None,
    #       scope='onset_velocities')

    #   velocity_values_flat = flatten_maybe_padded_sequences(
    #       velocity_values, lengths)
    #   tf.identity(velocity_values_flat, name='velocity_values_flat')
    #   velocity_labels_flat = flatten_maybe_padded_sequences(
    #       velocity_labels, lengths)
    #   velocity_loss = tf.reduce_sum(
    #       onset_labels_flat *
    #       tf.square(velocity_labels_flat - velocity_values_flat),
    #       axis=1)
    #   tf.losses.add_loss(tf.reduce_mean(velocity_loss))
    #   losses['velocity'] = velocity_loss

    with tf.variable_scope('frame'):
      if not hparams.share_conv_features:
        # TODO(eriche): this is broken when hparams.frame_lstm_units > 0
        activation_outputs = acoustic_model(
            spec, hparams, lstm_units=hparams.frame_lstm_units, lengths=lengths)
        activation_probs = slim.fully_connected(
            activation_outputs,
            hparams.num_bols,
            activation_fn=tf.sigmoid,
            scope='activation_probs')
      else:
        activation_probs = slim.fully_connected(
            onset_outputs,
            hparams.num_bols,
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
              rnn_cell_fw, rnn_cell_bw, inputs=combined_probs, dtype=tf.float32,scope='frame_bidirectional')
          combined_outputs = tf.concat(outputs, 2)
        else:
          combined_outputs, unused_output_states = tf.nn.dynamic_rnn(
              rnn_cell_fw, inputs=combined_probs, dtype=tf.float32)
      else:
        combined_outputs = combined_probs

      frame_probs = slim.fully_connected(
          combined_outputs,
          hparams.num_bols,
          activation_fn=tf.sigmoid,
          scope='frame_probs')

    frame_labels_flat = flatten_maybe_padded_sequences(frame_labels, lengths)
    frame_probs_flat = flatten_maybe_padded_sequences(frame_probs, lengths)
    tf.identity(frame_probs_flat, name='frame_probs_flat')
    # frame_label_weights_flat = flatten_maybe_padded_sequences(
    #     frame_label_weights, lengths)
    frame_losses = log_loss(
        frame_labels_flat,
        frame_probs_flat)
        # weights=frame_label_weights_flat
        # if hparams.weight_frame_and_activation_loss else None)
    tf.losses.add_loss(tf.reduce_mean(frame_losses))
    losses['frame'] = frame_losses

    if hparams.activation_loss:
      activation_losses = log_loss(
          frame_labels_flat,
          flatten_maybe_padded_sequences(activation_probs, lengths))
          # weights=frame_label_weights_flat
          # if hparams.weight_frame_and_activation_loss else None)
      tf.losses.add_loss(tf.reduce_mean(activation_losses))
      losses['activation'] = activation_losses

  predictions_flat = tf.cast(tf.greater_equal(frame_probs_flat, .1), tf.float32)

  # Creates a pianoroll labels in red and probs in green [minibatch, 88]
  images = {}
  onset_drumrolls = tf.concat(
      [
          onset_labels[:, :, :, tf.newaxis], onset_probs[:, :, :, tf.newaxis],
          tf.zeros(tf.shape(onset_labels))[:, :, :, tf.newaxis]
      ],
      axis=3)
  images['Onset_Drumrolls'] = onset_drumrolls
  activation_drumrolls = tf.concat(
      [
          frame_labels[:, :, :, tf.newaxis], frame_probs[:, :, :, tf.newaxis],
          tf.zeros(tf.shape(frame_labels))[:, :, :, tf.newaxis]
      ],
      axis=3)
  images['ActivationDrumrolls'] = activation_drumrolls
  
  return (tf.losses.get_total_loss(), losses, frame_labels_flat,
          predictions_flat, images)


  # return tf.losses.get_total_loss(), predictions_flat, frame_labels_flat