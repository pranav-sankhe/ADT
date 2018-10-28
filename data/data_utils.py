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

def spectrogram(y, n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=False,save_flag=False):

    # write('../test_audio/fut.wav', sr, y)      #write file under test
    if flag_hp:
        y_harm, y_perc = librosa.effects.hpss(y)

        D_harm = librosa.stft(y_harm, n_fft, hop_length, win_length, window='hann')
        D_perc = librosa.stft(y_perc, n_fft, hop_length, win_length, window='hann')

    
        plt.subplot(211)    
        librosa.display.specshow(librosa.amplitude_to_db(D_harm,
                                                       ref=np.max),
                               y_axis='log', x_axis='time')
        plt.title('Harmonic')    
        
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()            

        plt.subplot(212)
        librosa.display.specshow(librosa.amplitude_to_db(D_perc,
                                                       ref=np.max),
                               y_axis='log', x_axis='time')
        plt.title('Percussion')
        
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if plotFlag:
            plt.show()
        return D_perc        
    else:        
        D = librosa.stft(y, n_fft, hop_length, win_length, window='hann')
        librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')
        plt.title(':Power spectrogram: First ' + str(len(y)) + ' iterations' + ' with hopsize = ' + str(hop_length))
        plt.colorbar(format='%+2.0f dB')
        if save_flag:
            pylab.savefig('../results/' + str(len(y)) + 'i_' + 'spectogram.png')
        plt.tight_layout()
        if plotFlag:             
            plt.show()


def get_spectrogram(filepath, n_fft, win_length):

    #sr, y = wav.read(filepath)
    y, sr = librosa.load(filepath, sr=44100) 
    #sr, y = wav.read(filepath)

    y = np.pad(y, (0, data_params.max_audio_length - len(y)), 'constant', constant_values=(0))
    f, t, spec = signal.stft(y, sr, nperseg=win_length,nfft=n_fft)
    # plt.pcolormesh(t, f, np.abs(spec), vmin=0, vmax=2*np.sqrt(2)*np.abs(spec)[0][0])
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    mag = np.abs(spec)
    phase = np.angle(mag)
    return mag, f, t

def get_spec_dims(test_filepath, n_fft, win_length):
    
    #sr, y = wav.read(test_filepath)
    y, sr = librosa.load(test_filepath, sr=None) 
    y = np.pad(y, (0, data_params.max_audio_length - len(y)), 'constant', constant_values=(0))
    f, t, spec = signal.stft(y, sr, nperseg=win_length,nfft=n_fft)
    mag = np.abs(spec)
    phase = np.angle(mag)
    return mag.shape[0], mag.shape[1]

def pre_emphasis(input_signal):
    '''
    A pre-emphasis filter is useful in several ways: 
    - balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies,
    - avoid numerical problems during the Fourier transform operation and 
    - may also improve the Signal-to-Noise Ratio (SNR).
    Equation: y[t] = x[t] - alpha * x[t-1]
    '''

    pre_emphasis_alpha = hparams.pre_emphasis_alpha 
    pre_emphasized_signal = np.append(input_signal[0], input_signal[1:] - pre_emphasis_alpha * input_signal[:-1]) 
    return pre_emphasized_signal

def framing(input_signal, sample_rate):

    frame_size = hparams.frame_size
    frame_stride = hparams.frame_stride

    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate  
    signal_length = len(input_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(input_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]   
    frames = frames * np.hamming(frame_length)

    return frames


def frame_wise_fft(frames):
    fft_length = hparams.fft_length
    mag_frames = np.abs(np.fft.rfft(frames, fft_length))
    power_spectrum = np.square(mag_frames)/(fft_length)   # frame wise power spectrum

    return power_spectrum

def filter_banks(frames, sample_rate):
    '''
    We can convert between Hertz (f) and Mel (m) using the following equation
    m = 2595* log10(1 + f * 700)
    '''
    fft_length = hparams.fft_length
    num_filters = hparams.num_filters


    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((fft_length + 1) * hz_points / sample_rate)

    fbank = np.zeros((num_filters, int(np.floor(fft_length / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB


    return filter_banks
    

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]


def create_gt_activations_xml(xml_filepath, audio_filepath):
    drums, onset_times, offset_times = read_xml_file(xml_filepath)
    sample_rate = data_params.sample_rate
    num_drums = len(drums)
    HH_gt_onset = []
    KD_gt_onset = []
    SD_gt_onset = []
    for i in range(num_drums):
        if drums[i] == 'HH':
            HH_gt_onset.append(onset_times[i]*sample_rate)

        if drums[i] == 'KD':
            KD_gt_onset.append(onset_times[i]*sample_rate)

        if drums[i] == 'SD':
            SD_gt_onset.append(onset_times[i]*sample_rate)

    y, sr = librosa.load(audio_filepath, sr=data_params.sample_rate)        
    T = len(y)
    activation_HH = np.zeros(T)
    activation_SD = np.zeros(T)
    activation_KD = np.zeros(T)
        
    for i in HH_gt_onset:
        activation_HH[int(i)] = 1
    for i in KD_gt_onset:
        activation_KD[int(i)] = 1
    for i in SD_gt_onset:
        activation_SD[int(i)] = 1        
    
    return activation_HH, activation_KD, activation_SD

def create_gt_activations_svl(svl_filepath, audio_filepath):
    sample_rate = data_params.sample_rate
    frames = extractSvlAnnotRegionFile(svl_filepath)    
    y, sr = librosa.load(audio_filepath, sr=data_params.sample_rate)        
    T = len(y)
    activation = np.zeros(T)
    for i in frames:
        activation[int(i)] = 1

    return activation

            
def get_audio_files(data_dir, drum_type_index, gen_type_index):
        
    audio_dir = data_dir + '/audio'
    num_drums = data_params.num_drums 
    drums =  data_params.drums
    num_gen_type = data_params.num_gen_type
    gen_type =  data_params.gen_type

    # dummy_complete_file_list = os.listdir(audio_dir)
    # complete_file_list = [] 
    # length = len(dummy_complete_file_list)
    # for i in range(length):
    #     if dummy_complete_file_list[i].find('TechnoDrum02') == -1:
    #         if dummy_complete_file_list[i].find('WaveDrum02') == -1:     
    #             complete_file_list.append(dummy_complete_file_list[i])

    complete_file_list = os.listdir(audio_dir)            
    length = len(complete_file_list)
    list_drum_type = []
    if drum_type_index == 'all':
        list_drum_type = complete_file_list
    else:
        for i in range(length):
            if complete_file_list[i].find(drums[drum_type_index]) != -1:
                list_drum_type.append(complete_file_list[i])

    list_gen_type = []                     
    if gen_type_index == 'all':
        list_gen_type = complete_file_list
    else:
        if complete_file_list[i].find(gen_type[gen_type_index])!= -1:
            list_gen_type.append(complete_file_list[i])                

    drum_recording_list = np.intersect1d(list_drum_type, list_gen_type)

    return drum_recording_list


def get_xml_files(data_dir, drum_type_index, gen_type_index):
        
    audio_dir = data_dir + '/annotation_xml'
    num_drums = data_params.num_drums 
    drums =  data_params.drums
    num_gen_type = data_params.num_gen_type
    gen_type =  data_params.gen_type

    # dummy_complete_file_list = os.listdir(audio_dir)
    # complete_file_list = [] 
    # length = len(dummy_complete_file_list)
    # for i in range(length):
    #     if dummy_complete_file_list[i].find('TechnoDrum02') == -1:
    #         if dummy_complete_file_list[i].find('WaveDrum02') == -1:     
    #             complete_file_list.append(dummy_complete_file_list[i])

    complete_file_list = os.listdir(audio_dir)            
    length = len(complete_file_list)


    list_drum_type = []
    if drum_type_index == 'all':
        list_drum_type = complete_file_list
    
    else:
        for i in range(length):
            if complete_file_list[i].find(drums[drum_type_index]) != -1:
                
                list_drum_type.append(complete_file_list[i])

    list_gen_type = []                     
    if gen_type_index == 'all':
        list_gen_type = complete_file_list
    else:
        for i in range(length):
            if complete_file_list[i].find(gen_type[gen_type_index])!= -1:
                list_gen_type.append(complete_file_list[i])                

    drum_recording_list = np.intersect1d(list_drum_type, list_gen_type)             

    return drum_recording_list


# readSVLfiles.py


def read_xml_file(filepath):
    onset_times = []
    offset_times = []
    drums = []
    tree = ET.parse(filepath)
    root = tree.getroot()

    for onset_time in root.iter('onsetSec'):
        onset_times.append(float(onset_time.text))
    for offset_time in root.iter('offsetSec'):
        offset_times.append(float(offset_time.text))
    for drum in root.iter('instrument'):
        drums.append(drum.text)
    
    return drums, onset_times, offset_times


def get_svl_files(data_dir, drum_type_index, gen_type_index):
    
    audio_dir = data_dir + '/annotation_svl'
    num_drums = data_params.num_drums 
    drums =  data_params.drums
    num_gen_type = data_params.num_gen_type
    gen_type =  data_params.gen_type

    # dummy_complete_file_list = os.listdir(audio_dir)
    # complete_file_list = [] 
    # length = len(dummy_complete_file_list)
    # for i in range(length):
    #     if dummy_complete_file_list[i].find('TechnoDrum02') == -1:
    #         if dummy_complete_file_list[i].find('WaveDrum02') == -1:     
    #             complete_file_list.append(dummy_complete_file_list[i])

    complete_file_list = os.listdir(audio_dir)            
    length = len(complete_file_list)
    list_drum_type = []
    if drum_type_index == 'all':
        list_drum_type = complete_file_list
    else:
        for i in range(length):
            if complete_file_list[i].find(drums[drum_type_index]) != -1:
                list_drum_type.append(complete_file_list[i])

    list_gen_type = []                     
    if gen_type_index == 'all':
        list_gen_type = complete_file_list
    else:
        if complete_file_list[i].find(gen_type[gen_type_index])!= -1:
            list_gen_type.append(complete_file_list[i])                

    drum_recording_list = np.intersect1d(list_drum_type, list_gen_type)

    return drum_recording_list

def extractSvlAnnotRegionFile(filename):
    """
    extractSvlAnnotRegionFile(filename)
    
    Extracts the SVL files (sonic visualiser)
    in this function, we assume annotation files
    for regions (generated by Sonic Visualiser 1.7.2,
    Regions Layer)
    
    Returns the following objects:
        parameters: copy-paste of the "header" of the SVL file,
        frames    : a numpy array containing the time
                    stamps of each frame, at the beginning
                    of the frame,
        durations : a numpy array containing the duration
                    of each frame,
        labels    : a dictionary with keys equal to the frame
                    number, containing the labels,
        values    : a dictionary with keys equal to the frame
                    number, containing the values.
    
    Note that this code does not parse the end of the xml file.
    The 'display' field is therefore discarded here.
    
    Numpy and xml.dom should be imported in order to use this
    function.
        
    Jean-Louis Durrieu, 2010
    firstname DOT lastname AT epfl DOT ch
    """
    ## Load the XML structure:
    dom = minidom.parse(filename)
    
    ## Keep only the data-tagged field:
    ##    note that you could also keep any other
    ##    field here. 
    dataXML = dom.getElementsByTagName('model')
    # sample_rate = dataXML[0].attributes.keys()[6]
    # sample_rate = np.int(dataXML[0].getAttribute(sample_rate))    
    
    ## XML structure with all the points from datasetXML:
    pointsXML = dom.getElementsByTagName('point')
    
    nbPoints = len(pointsXML)
    
    ## number of attributes per point, not used here,
    ## but could be useful to check what type of SVL file it is?
    # nbAttributes = len(pointsXML[0].attributes.keys())
    
    ## Initialize the numpy arrays (frame time stamps and durations):
    frames = np.zeros([nbPoints], dtype=np.float)
    durations = np.zeros([nbPoints], dtype=np.float)
    ## Initialize the dictionaries (values and labels)
    values = {}
    labels = {}
    
    ## Iteration over the points:
    for node in range(nbPoints):
        ## converting sample to seconds for the time stamps and the durations:
        frames[node] = np.int(pointsXML[node].getAttribute('frame'))
        # durations[node] = np.int(pointsXML[node].getAttribute('duration')) / np.double(sample_rate)
        ## copy-paste for the values and the labels:
        # values[node] = pointsXML[node].getAttribute('value')
        # labels[node] = pointsXML[node].getAttribute('label')
        
    ## return the result:
    
    return frames#, durations, labels, values



def spectrogram_params():
    filepath = data_params.test_filepath

    n_fft = data_params.n_fft
    win_length = data_params.win_length
    
    y, sr = librosa.load(filepath, sr=44100) 
    
    y = np.pad(y, (0, data_params.max_audio_length - len(y)), 'constant', constant_values=(0))
    overlap = (3*win_length)/4
    f, t, spec = signal.stft(y, sr, nfft=n_fft)
    plt.pcolormesh(t, f, np.abs(spec), vmin=0, vmax=2*np.sqrt(2)*np.abs(spec)[0][0])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    mag = np.abs(spec)
    phase = np.angle(mag)
    return mag, f, t

    