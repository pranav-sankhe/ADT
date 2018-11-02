import numpy as np
import sys 
sys.path.insert(0, '../data')
import data_utils
import librosa
from sklearn.decomposition import NMF
import hparams
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt
import mir_eval


def concatenate_audio(audio_file_list):
    complete_list = []
    data_dir = hparams.data_dir
    for audio_filename in audio_file_list:
        audio_filepath = data_dir + '/audio/' + audio_filename + '.wav'
        y, sr = librosa.load(audio_filepath, sr=hparams.sample_rate)
        y = list(y)
        complete_list = complete_list + y
        print("Length of list" , len(complete_list))
    return complete_list


def concatenate_xml(xml_file_list, audio_file_list, n_fft, hop_length, win_length):
    complete_list = []
    HH = []
    KD = []
    SD = []
    data_dir = hparams.data_dir
    for audio_filename, xml_filename in zip(audio_file_list, xml_file_list):
        xml_filepath = data_dir + '/annotation_xml/' + xml_filename + '.xml'
        audio_filepath = data_dir + '/audio/' + audio_filename + '.wav'
        activation_HH, activation_KD, activation_SD = data_utils.create_gt_activations_xml(xml_filepath, audio_filepath, n_fft, hop_length, win_length)
        HH = list(HH) + list(activation_HH)
        KD = list(KD) + list(activation_KD)
        SD = list(SD) + list(activation_SD)
        print ("Length of HH activation = " , len(HH))
    return HH, KD, SD

def get_templates():
    gen_type = hparams.gen_type
    drum_type_index = 3         # select 'MIX' 
    data_dir = hparams.data_dir
    gen_type_index = 0
    n_fft = hparams.n_fft
    win_length = hparams.win_length
    hop_length = hparams.hop_length

    strip_dot = lambda x: x.split('.')[0]
    strip_hash = lambda x: x.split('#')[0] 

    audio_file_list = data_utils.get_audio_files(data_dir, drum_type_index, gen_type_index)
    
    audio_file_list = [x.split('.')[0] for x in audio_file_list]

    audio_file_list = audio_file_list[0:int(hparams.get_template_length*len(audio_file_list))]
    file_list_length = len(audio_file_list)

    xml_file_list = data_utils.get_xml_files(data_dir, drum_type_index, gen_type_index)
    xml_file_list = [x.split('.')[0] for x in xml_file_list]
    xml_file_list = np.intersect1d(xml_file_list, audio_file_list)

    audio_file_names = [x.split('#')[0] for x in audio_file_list]

    

    y = concatenate_audio(audio_file_list)
    HH, KD, SD = concatenate_xml(xml_file_list, audio_file_list, n_fft, hop_length, win_length)

    t, spec = data_utils.spectrogram(np.array(y), n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=True,save_flag=False)
    

    max_len = max(len(t), len(HH))
    V = np.zeros((spec.shape[0], max_len))
    V[:,0:spec.shape[1]] = spec

    activations = np.zeros((hparams.num_drums, max_len))
    activations[0, :] = HH
    activations[1, :] = KD
    activations[2, :] = SD
    
    print("Applying NMF ...")
    model = NMF(n_components=3, init='custom')
    template = model.fit_transform(V, W = np.random.rand(V.shape[0], hparams.num_drums) , H = activations)
        
    np.save('templates', template)                
    print("Templates saved in the file 'templates.npy' ")
    return template

def predict_activations(filename):

    gen_type = hparams.gen_type
    drum_type_index = 3         # select 'MIX' 
    data_dir = hparams.data_dir

    n_fft = hparams.n_fft
    win_length = hparams.win_length
    hop_length = hparams.hop_length

    
    
    filepath = data_dir + '/audio/' + filename + '.wav'
    y, sr = librosa.load(filepath, sr=hparams.sample_rate) 
    t, V = data_utils.spectrogram(np.array(y), n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=True,save_flag=False)
    T = V.shape[1]
    templates = np.load('templates.npy')
    # max_len = max(len(t), len(HH))
    # HH = np.pad(HH, (0, len(t) - len(HH)), 'constant', constant_values=(0))
    # KD = np.pad(KD, (0, len(t) - len(KD)), 'constant', constant_values=(0))
    # SD = np.pad(SD, (0, len(t) - len(SD)), 'constant', constant_values=(0))
    print("Applying NMF ...")
    model = NMF(n_components=3, init='custom')
    model.fit_transform(V, W = templates , H = np.random.rand(hparams.num_drums, T))
    H = model.components_
    print('Activations computed')
    return H

def eval(filename):
    data_dir = hparams.data_dir
    win_length = hparams.win_length
    n_fft = hparams.n_fft
    sample_rate = hparams.sample_rate
    hop_length = hparams.hop_length

    audio_filepath = data_dir + '/audio/' + filename + '.wav'
    xml_filepath = data_dir + '/annotation_xml/' + filename + '.xml'

    
    drums, onset_times, offset_times = data_utils.read_xml_file(xml_filepath)
    y, sr = librosa.load(audio_filepath, sr=hparams.sample_rate) 
    t, V = data_utils.spectrogram(np.array(y), n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=True,save_flag=False)
    

    length = len(drums)
    HH_gt_onset = []
    KD_gt_onset = []
    SD_gt_onset = []
    for i in range(length):
        if drums[i] == 'HH':
            HH_gt_onset.append(onset_times[i])

        if drums[i] == 'KD':
            KD_gt_onset.append(onset_times[i])

        if drums[i] == 'SD':
            SD_gt_onset.append(onset_times[i])

    for i in range(len(HH_gt_onset)):
        HH_gt_onset[i] = data_utils.find_nearest(t, HH_gt_onset[i])

    for i in range(len(KD_gt_onset)):
        KD_gt_onset[i] = data_utils.find_nearest(t, KD_gt_onset[i])

    for i in range(len(SD_gt_onset)):
        SD_gt_onset[i] = data_utils.find_nearest(t, SD_gt_onset[i])


    pred_activations = predict_activations(filename)
    
    HH_pred = pred_activations[0, :]
    KD_pred = pred_activations[1, :]
    SD_pred = pred_activations[2, :]
    
    HH_peaks, _ = find_peaks(HH_pred, width=1, prominence=3) 
    KD_peaks, _ = find_peaks(KD_pred, width=1, prominence=3) 
    SD_peaks, _ = find_peaks(SD_pred, width=1, prominence=3) 

    # plt.figure(); plt.plot(HH_peaks, HH_pred[HH_peaks], "ob"); plt.plot(HH_pred); plt.legend(['HH'])
    # plt.figure(); plt.plot(KD_peaks, KD_pred[KD_peaks], "ob"); plt.plot(KD_pred); plt.legend(['KD'])
    # plt.figure(); plt.plot(SD_peaks, SD_pred[SD_peaks], "ob"); plt.plot(SD_pred); plt.legend(['SD'])


    pred_time_HH = t[HH_peaks]
    pred_time_KD = t[KD_peaks]
    pred_time_SD = t[SD_peaks]

    
    f_measure_HH = mir_eval.beat.f_measure(np.array(HH_gt_onset), np.array(pred_time_HH))
    f_measure_SD = mir_eval.beat.f_measure(np.array(SD_gt_onset), np.array(pred_time_SD))
    f_measure_KD = mir_eval.beat.f_measure(np.array(KD_gt_onset), np.array(pred_time_KD))
        
    pred_time_all = np.append(pred_time_HH, pred_time_KD)
    pred_time_all = np.append(pred_time_all ,pred_time_SD)
    pred_time_all = np.sort(pred_time_all)

    gt_all = np.append(HH_gt_onset, KD_gt_onset) 
    gt_all = np.append(gt_all, SD_gt_onset)
    gt_all = np.sort(gt_all)
    f_measure = mir_eval.beat.f_measure(gt_all, pred_time_all)
    
    # plt.subplot(3, 1, 1)
    # HH_peaks, _ = find_peaks(HH_pred, width=1.2, prominence=2.7) 
    # plt.plot(HH_peaks, HH_pred[HH_peaks], "ob"); plt.plot(HH_pred)
    # plt.subplot(3, 1, 2)
    # KD_peaks, _ = find_peaks(KD_pred, width=1.2, prominence=2.7) 
    # plt.plot(KD_peaks, KD_pred[KD_peaks], "ob"); plt.plot(KD_pred)
    # plt.subplot(3, 1, 3)
    # SD_peaks, _ = find_peaks(SD_pred, width=1.2, prominence=2.7) 
    # plt.plot(SD_peaks, SD_pred[SD_peaks], "ob"); plt.plot(SD_pred)
    # plt.show()

    
    return f_measure
