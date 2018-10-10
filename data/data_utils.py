import numpy as np 
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf 
import os


def spectrogram(y, sr, N, n_fft, hop_length, win_length  ):
	spec = librosa.stft(y, n_fft, hop_length, win_length, window='hann', center=True, pad_mode='reflect')
	mag = np.abs(spec)
	phase = np.angle(mag)
	return mag

def get_templates():
	






	
