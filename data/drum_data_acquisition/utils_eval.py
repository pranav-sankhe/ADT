import data_utils
import numpy as np 
import matplotlib.pyplot as plt
import librosa
from scipy import signal
# import tensorflow as tf 
import os
import data_params
import xml.etree.ElementTree as ET
from xml.dom import minidom
import librosa.display
from scipy.io import wavfile as wav


n_fft = data_params.n_fft
win_length = data_params.win_length
hop_length = 3*win_length/4

filepath = data_params.test_filepath
y, sr = librosa.load(filepath, sr=44100)
t, spec = data_utils.spectrogram(y, n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=True,save_flag=False)

import pdb; pdb.set_trace()

