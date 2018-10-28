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


filepath = data_params.test_filepath
y, sr = librosa.load('test_solo.wav', sr=data_params.sample_rate)
frames = data_utils.extractSvlAnnotRegionFile('test.svl')

import pdb; pdb.set_trace()

n_fft = data_params.n_fft
win_length = data_params.win_length
hop_length = 3*win_length/4
l = data_utils.create_gt_activations_svl('test.svl', 'test_solo.wav')