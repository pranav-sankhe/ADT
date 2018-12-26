import numpy as np 
import matplotlib.pyplot as plt
import librosa.display
from scipy.io.wavfile import write,read
import librosa
import os
import scipy
from scipy import signal


def add_gaussian(filepath):
	y, sr = librosa.load(filepath, sr=44100) 
	y_hat = y + np.random.rand()



def pre_emphasis(input_signal):
    '''
    A pre-emphasis filter is useful in several ways: 
    - balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies,
    - avoid numerical problems during the Fourier transform operation and 
    - may also improve the Signal-to-Noise Ratio (SNR).
    Equation: y[t] = x[t] - alpha * x[t-1] 
    '''
    pre_emphasis_alpha = 0.95#params.pre_emphasis_alpha 
    pre_emphasized_signal = np.append(input_signal[0], input_signal[1:] - pre_emphasis_alpha * input_signal[:-1]) 
    return pre_emphasized_signal


def get_limiting_indices(y):
    y = y/np.max(y)
    energy_threshold = 0.3#params.energy_threshold
    window_len = 3500

    window = np.hamming(window_len)
    sig_energy = np.convolve(y**2,window**2,'same')
    
    
    sig_energy = sig_energy/max(sig_energy)     #Normalize energy
    sig_energy_thresh = (sig_energy > energy_threshold).astype('float')
    #convert the bar graph to impulses by subtracting signal from it's shifted version 
    indices = np.nonzero(abs(sig_energy_thresh[1:] - sig_energy_thresh[0:-1]))[0]         
    
    start_indices = [indices[2*i] for i in range(int(len(indices)/2))]
    end_indices   = [indices[2*i+1] for i in range(int(len(indices)/2))]

    return start_indices, end_indices


def segment(filepath):
	y, sr = librosa.load(filepath, sr=44100)
	start_indices, end_indices = get_limiting_indices(y)

	for p in range(len(end_indices)):
	    sig = y[start_indices[p] : end_indices[p]]
	    sig = pre_emphasis(sig)
	    filename = filepath.split('.')[0].split('/')[-1]
	    write( filename + '.wav', sr, sig)

