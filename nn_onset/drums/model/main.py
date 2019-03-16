from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np 
import tensorflow as tf

  

def core_model(spec, onset_labels, frame_labels, lengths):

    losses = {}

    onset_outputs, onset_probs, onset_losses = onset_detector_model(spec, onset_labels)
    tf.losses.add_loss(tf.reduce_mean(onset_losses))
    losses['onset'] = onset_losses

    frame_losses, predictions_flat, frame_labels_flat = note_detector_model(spec, frame_labels, lengths, onset_probs, onset_outputs)
    tf.losses.add_loss(tf.reduce_mean(frame_losses))
    losses['frame'] = frame_losses

    return tf.losses.get_total_loss(),losses, frame_labels_flat, predictions_flat


def train():

    spec = tf.placeholder(tf.float32, [hparams.batch_size, hparams.freq_length, hparams.time_len])
    onset_labels = tf.placeholder(tf.float32, [hparams.batch_size, hparams.num_bols])
    frame_labels = tf.placeholder(tf.float32, [hparams.batch_size, hparams.num_bols])
    # lengths = 

    keep_prob = tf.placeholder_with_default(1.0, shape=())

    onset_losses, frame_losses, predictions_flat = core_model(spec, onset_labels, frame_labels, lengths)



    # with tf.Graph().as_default():
    transcription_data = _get_data(examples_path, hparams, is_training=True)
    spec, onset_labels, frame_labels, lengths = transcription_data


    loss, losses, unused_labels, unused_predictions = core_model(spec, onset_labels, frame_labels, lengths)

    tf.summary.scalar('loss', loss)
    for label, loss_collection in losses.iteritems():
      loss_label = 'losses/' + label
      tf.summary.scalar(loss_label, tf.reduce_mean(loss_collection))

    global_step = tf.train.get_or_create_global_step()
    
    learning_rate = tf.train.exponential_decay(
        hparams.learning_rate,
        global_step,
        hparams.decay_steps,
        hparams.decay_rate,
        staircase=True)
    
    tf.summary.scalar('learning_rate', learning_rate)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        clip_gradient_norm=hparams.clip_norm,
        summarize_gradients=True)

    logging_dict = {'global_step': tf.train.get_global_step(), 'loss': loss}

    hooks = [tf.train.LoggingTensorHook(logging_dict, every_n_iter=100)]
    if num_steps:
      hooks.append(tf.train.StopAtStepHook(num_steps))

    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            max_to_keep=checkpoints_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

    tf.contrib.training.train(
        train_op=train_op,
        logdir=train_dir,
        scaffold=scaffold,
        hooks=hooks,
        save_checkpoint_secs=300)


def console_entry_point():
  tf.app.run(train)


if __name__ == '__main__':
  console_entry_point()