from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import librosa.display
from scipy.io.wavfile import write,read
import librosa
import os
import scipy
from scipy import signal
import params 
# from magenta.music import audio_io
import wave
import six
import pandas as pd
def wav_to_mel(y):
  """Transforms the contents of a wav file into a series of mel spec frames."""
  # y = librosa.util.normalize(librosa.core.load(filepath, sr=params.sample_rate)[0])

  mel = librosa.feature.melspectrogram(
      y,
      params.sample_rate,
      hop_length=params.spec_hop_length,
      n_fft = params.n_fft,
      fmin=params.spec_fmin,
      n_mels=params.spec_n_bins).astype(np.float32)

  # Transpose so that the data is in [frame, bins] format.
  mel = mel.T
  return mel

def read_anottFile(filepath):
  data = pd.read_csv(filepath, header=None)
  onset_times = data[2][1:].values
  onset_times = onset_times.astype(int)
  bols = data[1][1:].values
  
  bad_indices = []  
  for i in range(len(bols)):
    if bols[i] not in params.bols:
      bad_indices.append(i)


  bols = np.delete(bols, bad_indices)
  onset_times = np.delete(onset_times, bad_indices)
  
  for i in range(len(bols)):
    bols[i] = bols[i].replace(" ", "")
  return onset_times, bols

def wav_to_num_frames(filepath):
  """Transforms a wav-encoded audio string into number of frames."""
  frames_per_second = params.sample_rate / params.spec_hop_length
  wav_data = audio_io.samples_to_wav_data(
      librosa.util.normalize(librosa.core.load(
          filepath, sr=params.sample_rate)[0]), params.sample_rate)

  w = wave.open(six.BytesIO(wav_data))
  
  return np.int32(w.getnframes() / w.getframerate() * frames_per_second)


# mel = wav_to_mel('now.wav')
# wav_to_num_frames('now.wav')



def get_labels():
  output_matrix =  np.zeros(params.batch_size, length)

def bol_to_int(bol):
  unique_bols = params.bols
  integer = unique_bols.index(bol)
  return integer

def split(filename, unit_length):
  wav_filepath = params.train_wav_dir +  filename + '.wav'
  y_prime = librosa.util.normalize(librosa.core.load(wav_filepath, sr=params.sample_rate)[0])

  split_length = unit_length*params.sample_rate
  num_splits = len(y_prime)//split_length + 1
  
  y = np.zeros(split_length*num_splits)
  y[0:len(y_prime)] = y_prime

  output = np.split(y, num_splits)
  specs = []
  print("--------------------------------------------------")
  print("Computing Spectrograms...")
  print("--------------------------------------------------")
  for i in range(num_splits):
    specs.append(wav_to_mel(output[i]))
    print("computed spectrogram for split " + str(i))

  # frames_per_second = params.sample_rate / params.spec_hop_length
  # num_frames = np.int32((split_length/params.sample_rate) * frames_per_second)

  anott_filepath = params.train_anott_dir +  filename + '.csv'
  onset_times, bols = read_anottFile(anott_filepath)
  
  start = 0
  end = 20*params.sample_rate
  step = params.spec_hop_length
  t = np.arange(start=start, stop=end, step=step)
  num_frames = len(t)
  
  onset_labels = np.zeros((num_splits, num_frames))
  print("--------------------------------------------------")
  print("Computing Onset Labels ...")
  print("--------------------------------------------------")
  for onset_time in onset_times:
    split_index = onset_time//split_length
    onset_index = onset_time - split_length*split_index
    idx = np.abs(t - onset_index).argmin()
    # import pdb; pdb.set_trace()
    onset_labels[split_index, idx] = 1
  print("computed onset labels")  

  
  bol_labels = np.zeros((num_splits, num_frames))

  print("--------------------------------------------------")
  print("Computing Bol Labels ...")
  print("--------------------------------------------------")  
  for i in range(len(onset_times)):
    split_index = onset_times[i]//split_length
    onset_index = onset_times[i] - split_length*split_index
    idx = np.abs(t - onset_index).argmin()
    bol_labels[split_index, idx] = bol_to_int(bols[i])
  print("computed bol labels")  
  return specs, onset_labels, bol_labels


def store_as_npy():

  train_wav_dir = params.train_wav_dir
  train_anott_dir = params.train_anott_dir


  wav_files = os.listdir(train_wav_dir)
  filenames = []
  for i in range(len(wav_files)):
    filenames.append(wav_files[i].split('.')[0]) 
  # anott_files = os.listdir(train_anott_dir)
  # anott_files = np.sort(anott_files)

  if not os.path.exists(params.np_train_files_dir):
    os.makedirs(params.np_train_files_dir)

  if not os.path.exists(params.np_spec_dir):
    os.makedirs(params.np_spec_dir)

  if not os.path.exists(params.np_onset_dir):
    os.makedirs(params.np_onset_dir)

  if not os.path.exists(params.np_bols_dir):
    os.makedirs(params.np_bols_dir)

  np_spec_dir = params.np_spec_dir
  np_onset_dir = params.np_onset_dir
  np_bols_dir = params.np_bols_dir


  for filename in filenames:
    specs, onset_labels, bol_labels = split(filename, unit_length=20)
  
    num_splits = len(specs)

    for i in range(num_splits):
      np.save( np_spec_dir + filename + '_split' + str(i), specs[i])
      np.save( np_onset_dir + filename + '_split' + str(i), onset_labels[i])
      np.save( np_bols_dir + filename + '_split' + str(i), bol_labels[i])

store_as_npy()