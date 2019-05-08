from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np 
import tensorflow as tf
import hparams
import utils

import sys
sys.path.insert(0, '../data')
import data_utils
  

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

  # keep_prob = tf.placeholder_with_default(1.0, shape=())

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


  with tf.name_scope("compute_gradients"):
    # compute_gradients` returns a list of (gradient, variable) pairs
    params = tf.trainable_variables()

    for var in params:
      tf.summary.histogram(var.name, var)
      
    grads = tf.gradients(xs=params, ys=loss, colocate_gradients_with_ops=True)    # optimizer.compute_gradients(loss)
    clipped_grads, grad_norm_summary, grad_norm = gradient_clip(grads, max_gradient_norm=hparams.max_gradient_norm)
    grad_and_vars = zip(clipped_grads, params)

  apply_gradient_op = optimizer.apply_gradients(grad_and_vars, global_step)    


  session_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True,
      )
  session_config.gpu_options.allow_growth = True
  
  sess = tf.InteractiveSession(config=session_config)

  initializer = tf.contrib.layers.xavier_initializer()


  with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
    # with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()

    for j in range(hparams.num_iterations): 
      print ("Training:: iteration: ", j)
      
      train_length = 3000 #3391
      num_epochs = (train_length)/hparams.batch_size         # number of epochs for training 


      for i in range(int(epochs)-1):
        print ("Training:: Epoch ", i)

        spec_list, onset_list, bols_list = data_utils.provide_batch(i)
        _, loss_val, summary = sess.run(
            [apply_gradient_op, loss, merged],
            feed_dict={
                spec: spec_list, 
                onset_labels: onset_list,
                frame_labels: bols_list
            }
        ) 
        train_writer.add_summary(summary, j)

        print (np.sum(loss_val)/np.size(loss_val))

  # logging_dict = {'global_step': tf.train.get_global_step(), 'loss': loss}

  # hooks = [tf.train.LoggingTensorHook(logging_dict, every_n_iter=100)]
  # if num_steps:
  #   hooks.append(tf.train.StopAtStepHook(num_steps))

  # scaffold = tf.train.Scaffold(
  #     saver=tf.train.Saver(
  #         max_to_keep=checkpoints_to_keep,
  #         keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))



train()