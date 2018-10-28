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



def concatenate_audio(audio_file_list):
    complete_list = []
    data_dir = hparams.data_dir
    for audio_filename in audio_file_list:
        audio_filepath = data_dir + '/audio/' + audio_filename + '.wav'
        y, sr = librosa.load(audio_filepath, sr=hparams.sample_rate)
        y = list(y)
        complete_list = complete_list + y
    return complete_list


def concatenate_xml(xml_file_list, audio_file_list):
    complete_list = []
    HH = []
    KD = []
    SD = []
    data_dir = hparams.data_dir
    for audio_filename, xml_filename in zip(audio_file_list, xml_file_list):
        xml_filepath = data_dir + '/annotation_xml/' + xml_filename + '.xml'
        audio_filepath = data_dir + '/audio/' + audio_filename + '.wav'
        activation_HH, activation_KD, activation_SD = data_utils.create_gt_activations_xml(xml_filepath, audio_filepath)
        activation_HH = list(activation_HH)
        activation_SD = list(activation_SD)
        activation_KD = list(activation_KD)
        HH = HH + activation_HH
        KD = KD + activation_KD
        SD = SD + activation_SD

    return HH, KD, SD

def get_templates():
    gen_type = hparams.gen_type
    drum_type_index = 3         # select 'MIX' 
    data_dir = hparams.data_dir

    strip_dot = lambda x: x.split('.')[0]
    strip_hash = lambda x: x.split('#')[0] 

    audio_file_list = data_utils.get_audio_files(data_dir, drum_type_index, "all")
    
    audio_file_list = [x.split('.')[0] for x in audio_file_list]

    audio_file_list = audio_file_list[0:int(hparams.get_template_length*len(audio_file_list))]
    file_list_length = len(audio_file_list)

    xml_file_list = data_utils.get_xml_files(data_dir, drum_type_index, "all")
    xml_file_list = [x.split('.')[0] for x in xml_file_list]
    xml_file_list = np.intersect1d(xml_file_list, audio_file_list)

    audio_file_names = [x.split('#')[0] for x in audio_file_list]

    y = concatenate_audio(audio_file_list)
    HH, KD, SD = concatenate_xml(xml_file_list, audio_file_list)
    activations = np.zeros((hparams.num_drums, len(y)))
    activations[0, :] = HH
    activations[1, :] = KD
    activations[2, :] = SD
    n_fft = hparams.n_fft
    win_length = hparams.win_length
    hop_length = hparams.hop_length
    import pdb; pdb.set_trace()
    V = data_utils.spectrogram(y, n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=False,save_flag=False)
    
    model = NMF(n_components=3, init='custom')
    template = model.fit_transform(V, W = avg_template , H = activations)
        
    np.save('templates', avg_template)                
    return avg_template

        



def predict_activations(filename):

    gen_type = hparams.gen_type
    drum_type_index = 3         # select 'MIX' 
    data_dir = hparams.data_dir

    n_fft = hparams.n_fft
    win_length = hparams.win_length

    K, T = data_utils.get_spec_dims(hparams.test_filepath, n_fft, win_length)
    
    filepath = data_dir + '/audio/' + filename + '.wav'
    V, f, t = data_utils.get_spectrogram(filepath, n_fft, win_length)

    templates = np.load('templates.npy')

    model = NMF(n_components=3, init='custom')
    model.fit_transform(V, W = templates , H = np.random.rand(hparams.num_drums, T))
    H = model.components_

    return H

def eval(filename):
    data_dir = hparams.data_dir
    win_length = hparams.win_length
    n_fft = hparams.n_fft
    sample_rate = hparams.sample_rate

    audio_filepath = data_dir + '/audio/' + filename + '.wav'
    xml_filepath = data_dir + '/annotation_xml/' + filename + '.xml'
    
    K, T = data_utils.get_spec_dims(hparams.test_filepath, n_fft, win_length)
    
    drums, onset_times, offset_times = data_utils.read_xml_file(xml_filepath)
    V, f, t = data_utils.get_spectrogram(audio_filepath, n_fft, win_length)

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
    HH_pred = pred_activations[0,:]
    KD_pred = pred_activations[1,:]
    SD_pred = pred_activations[2,:]

    HH_peaks, _ = find_peaks(HH_pred, width=1.3, prominence=3.2) 
    KD_peaks, _ = find_peaks(KD_pred, width=1.3, prominence=3.2) 
    SD_peaks, _ = find_peaks(SD_pred, width=1.3, prominence=3.2) 

    # HH_peaks = np.divide(HH_peaks, sample_rate)
    # SD_peaks = np.divide(SD_peaks, sample_rate)
    # KD_peaks = np.divide(KD_peaks, sample_rate)
    plt.figure(); plt.plot(HH_peaks, HH_pred[HH_peaks], "ob"); plt.plot(HH_pred); plt.legend(['HH'])
    plt.figure(); plt.plot(KD_peaks, KD_pred[KD_peaks], "ob"); plt.plot(KD_pred); plt.legend(['KD'])
    plt.figure(); plt.plot(SD_peaks, SD_pred[SD_peaks], "ob"); plt.plot(SD_pred); plt.legend(['SD'])    
    plt.show()

    pred_time_HH = t[HH_peaks]
    pred_time_KD = t[KD_peaks]
    pred_time_SD = t[SD_peaks]
    
    # import pdb; pdb.set_trace()
    # f_measure = mir_eval.beat.f_measure(gt_activations, estimated_beats)




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

    
    # return f_measure

def tweak_eval_params(filename):
    data_dir = hparams.data_dir
    win_length = hparams.win_length
    n_fft = hparams.n_fft
    sample_rate = hparams.sample_rate

    audio_filepath = data_dir + '/audio/' + filename + '.wav'
    xml_filepath = data_dir + '/annotation_xml/' + filename + '.xml'
    
    K, T = data_utils.get_spec_dims(hparams.test_filepath, n_fft, win_length)
    # V, f, t = data_utils.get_spectrogram(data_dir + '/audio/' + audio_file_list[i] + '.wav', n_fft, win_length)
    gt_activations =  data_utils.read_xml_file(xml_filepath)
    

    pred_activations = predict_activations(filename)
    HH_pred = pred_activations[0,:]
    KD_pred = pred_activations[1,:]
    SD_pred = pred_activations[2,:]


    peaks, _ = find_peaks(HH_pred, distance=20)
    peaks2, _ = find_peaks(HH_pred, width=1.2, prominence=2.7)      # BEST!
    peaks3, _ = find_peaks(HH_pred, width=20)
    peaks4, _ = find_peaks(HH_pred, threshold=0.4)     # Required vertical distance to its direct neighbouring samples, pretty useless
    
    plt.subplot(2, 2, 1)
    plt.plot(peaks, HH_pred[peaks], "xr"); plt.plot(HH_pred); plt.legend(['distance'])
    plt.subplot(2, 2, 2)
    plt.plot(peaks2, HH_pred[peaks2], "ob"); plt.plot(HH_pred); plt.legend(['prominence'])
    plt.subplot(2, 2, 3)
    plt.plot(peaks3, HH_pred[peaks3], "vg"); plt.plot(HH_pred); plt.legend(['width'])
    plt.subplot(2, 2, 4)
    plt.plot(peaks4, HH_pred[peaks4], "xk"); plt.plot(HH_pred); plt.legend(['threshold'])
    plt.show()

    # for i in range(len(HH_gt_onset)):
    #     HH_gt_onset[i] = find_nearest(t, HH_gt_onset[i])

    # for i in range(len(KD_gt_onset)):
    #     KD_gt_onset[i] = find_nearest(t, KD_gt_onset[i])

    # for i in range(len(SD_gt_onset)):
    #     SD_gt_onset[i] = find_nearest(t, SD_gt_onset[i])
