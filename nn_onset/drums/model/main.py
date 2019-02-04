from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np 
import tensorflow as tf

  

def core_model(spec, onset_labels, frame_labels, lengths):

  onset_probs, onset_outputs = onset_detector_model(spec, onset_labels)
  frame_probs = note_detector_model(spec, frame_labels, lengths, onset_probs, onset_outputs)

def train():
	spec = tf.placeholder(tf.float32, [hparams.batch_size, hparams.freq_length, hparams.time_len])
	onset_labels = 
	frame_labels = 
	lengths = 
