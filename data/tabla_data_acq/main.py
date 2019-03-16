import os
import pandas as pd 
import numpy as np 
import params
import librosa
from scipy import signal
import params

def get_spectrogram(filepath, n_fft, win_length):

    y, sr = librosa.load(filepath, sr=44100) 

    y = np.pad(y, (0, data_params.max_audio_length - len(y)), 'constant', constant_values=(0))
    f, t, spec = signal.stft(y, sr, nperseg=win_length,nfft=n_fft)
    mag = np.abs(spec)
    phase = np.angle(mag)
    return mag, f, t

def get_gtOnsets(filepath):
	data = pd.read_csv(filepath, header=None)
	time = data[0].values
	bol = data[1].values

	return time, bol

def concatenate_audio(audio_file_list):
    complete_list = []
    data_dir = params.audio_dir
    for audio_filename in audio_file_list:
        audio_filepath = data_dir + '/' + audio_filename + '.wav'
        y, sr = librosa.load(audio_filepath, sr=params.sample_rate)
        y = list(y)
        complete_list = complete_list + y
        print("Length of list" , len(complete_list))
    return complete_list



def spectrogram(y, n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=False,save_flag=False):
    print("Computing the spectrogram....")
    # write('../test_audio/fut.wav', sr, y)      #write file under test
    if flag_hp:
        y_harm, y_perc = librosa.effects.hpss(y)

        # D_harm = librosa.stft(y_harm, int(n_fft), int(hop_length), int(win_length), window='hann')
        D_perc = librosa.stft(y_perc, int(n_fft), int(hop_length), int(win_length), window='hann')

        sample_rate = params.sample_rate
        num_samples = len(y)
        time_len = num_samples/sample_rate
        
        start = 0
        end = time_len
        step = hop_length/sample_rate

        segmented_time = np.arange(start=start, stop=end, step=step)
        return segmented_time, np.absolute(D_perc)

    else:        
        D = librosa.stft(y, n_fft, hop_length, win_length, window='hann')
        return D


def create_gt_activations(csv_filepath, audio_filepath, n_fft, hop_length, win_length):
    print("Creating activations from file " + audio_filepath + ' ....')
    onset_times, bols = get_gtOnsets(csv_filepath)
    sample_rate = params.sample_rate
    num_timeStamps = len(bols)

    bol_list = params.bols
    # num_bols = len(bol_list)
    HH_gt_onset = []
    KD_gt_onset = []
    SD_gt_onset = []
    

    for i in range(num_timeStamps):
        bols[i] = bols[i].replace(" ", "")

    y, sr = librosa.load(audio_filepath, sr=params.sample_rate)
    t, spec = spectrogram(y, n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=True,save_flag=False)
    T = len(t)    

    annotation = dict() 
    for i in range(len(bol_list)):
        annotation[bol_list[i]] = []

    for i in range(num_timeStamps):
        annotation[bols[i]].append(onset_times[i]*sample_rate)
    
    activation = dict()    

    for i in range(len(bol_list)):
        activation[bol_list[i]] = np.zeros(T)

    for bol in bol_list:
        for i in annotation[bol]:
            idx = np.abs(t*params.sample_rate - i).argmin()
            activation[bol][idx] = 1

    return activation





onset_dir = params.onset_dir
anot_files = os.listdir(onset_dir)
anot_filepath = onset_dir + '/' + anot_files[0]


audio_dir = params.audio_dir
audio_filepath = audio_dir + '/' + anot_files[0].split('.')[0] + '.wav'
# audio_files = os.listdir(audio_dir)
# audio_filepath = audio_dir + '/' + audio_files[0]

sample_rate = params.sample_rate
n_fft = params.n_fft
win_length = params.win_length
hop_length = params.hop_length

create_gt_activations(anot_filepath, audio_filepath, n_fft, hop_length, win_length)