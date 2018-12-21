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

def concatenate_svl(audio_file_list, n_fft, hop_length, win_length):
    complete_list = []
    flag = 0 
    data_dir = hparams.data_dir
    for audio_filename in audio_file_list:
        audio_filepath = data_dir + '/audio/' + audio_filename + '.wav'
        
        activation_HH, activation_KD, activation_SD = data_utils.create_gt_activations_svl(audio_filepath, n_fft, hop_length, win_length)
        
        if flag == 0:
            HH = activation_HH
            KD = activation_KD
            SD = activation_SD
            flag = 1
        else:
            HH = np.append(HH, activation_HH, axis=1)
            KD = np.append(HH, activation_KD, axis=1)
            SD = np.append(HH, activation_SD, axis=1)
        print ("Length of HH activation = " , max(HH.shape))
    return HH, KD, SD


def get_templates():
    gen_type = hparams.gen_type
    drum_type_index = 3         # select 'MIX'
    data_dir = hparams.data_dir
    gen_type_index = 2
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

    svl_file_list = data_utils.get_svl_files(data_dir, drum_type_index, gen_type_index)
    svl_file_list = [x.split('.')[0] for x in svl_file_list]
    svl_file_list = np.intersect1d(svl_file_list, audio_file_list)

    audio_file_names = [x.split('#')[0] for x in audio_file_list]

    y = concatenate_audio(audio_file_list)
    # HH, KD, SD = concatenate_xml(xml_file_list, audio_file_list, n_fft, hop_length, win_length)
    HH, KD, SD = concatenate_svl(audio_file_list, n_fft, hop_length, win_length)

    t, spec = data_utils.spectrogram(np.array(y), n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=True,save_flag=False)

    HH_Ncomponents = hparams.HH_Ncomponents
    KD_Ncomponents = hparams.KD_Ncomponents
    SD_Ncomponents = hparams.SD_Ncomponents
    nmf_Ncomponents = HH_Ncomponents + KD_Ncomponents + SD_Ncomponents
    
    max_len = max( max(HH.shape), max(KD.shape), max(SD.shape) )
    V = np.zeros((spec.shape[0], max_len))
    V[:,0:spec.shape[1]] = spec

    activations = np.zeros((nmf_Ncomponents, max_len))
    activations[0:HH_Ncomponents, 0:max(HH.shape)] = HH
    activations[HH_Ncomponents: HH_Ncomponents + KD_Ncomponents, 0:max(KD.shape)] = KD
    activations[-SD_Ncomponents:, 0:max(SD.shape)] = SD
    
    print("Applying NMF ...")

    model = NMF(n_components = nmf_Ncomponents, solver= hparams.solver, init='custom', max_iter=hparams.max_iter)
    template = model.fit_transform(V, W = np.random.rand(V.shape[0], nmf_Ncomponents) , H = activations)
        
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

    HH_Ncomponents = hparams.HH_Ncomponents
    KD_Ncomponents = hparams.KD_Ncomponents
    SD_Ncomponents = hparams.SD_Ncomponents
    nmf_Ncomponents = HH_Ncomponents + KD_Ncomponents + SD_Ncomponents
    
    
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
    model = NMF(n_components=nmf_Ncomponents, solver= hparams.solver, init='custom', max_iter=hparams.max_iter)
    model.fit_transform(V, W = templates , H = np.random.rand(nmf_Ncomponents, T))
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
    # xml_filepath = data_dir + '/annotation_xml/' + filename + '.xml'

    
    # drums, onset_times, offset_times = data_utils.read_xml_file(xml_filepath)
    y, sr = librosa.load(audio_filepath, sr=hparams.sample_rate) 
    t, V = data_utils.spectrogram(np.array(y), n_fft, hop_length, win_length, window='hann', plotFlag=True,flag_hp=True,save_flag=False)
    

    # length = len(drums)
    # HH_gt_onset = []
    # KD_gt_onset = []
    # SD_gt_onset = []
    # for i in range(length):
    #     if drums[i] == 'HH':
    #         HH_gt_onset.append(onset_times[i])

    #     if drums[i] == 'KD':
    #         KD_gt_onset.append(onset_times[i])

    #     if drums[i] == 'SD':
    #         SD_gt_onset.append(onset_times[i])

    # for i in range(len(HH_gt_onset)):
    #     HH_gt_onset[i] = data_utils.find_nearest(t, HH_gt_onset[i])

    # for i in range(len(KD_gt_onset)):
    #     KD_gt_onset[i] = data_utils.find_nearest(t, KD_gt_onset[i])

    # for i in range(len(SD_gt_onset)):
    #     SD_gt_onset[i] = data_utils.find_nearest(t, SD_gt_onset[i])



    HH_Ncomponents = hparams.HH_Ncomponents
    KD_Ncomponents = hparams.KD_Ncomponents
    SD_Ncomponents = hparams.SD_Ncomponents
    nmf_Ncomponents = HH_Ncomponents + KD_Ncomponents + SD_Ncomponents

    HH_gt_onset, KD_gt_onset, SD_gt_onset = data_utils.get_onsets_svl(audio_filepath)

    HH_gt_onset = np.divide(HH_gt_onset, sample_rate)
    KD_gt_onset = np.divide(KD_gt_onset, sample_rate)
    SD_gt_onset = np.divide(SD_gt_onset, sample_rate)

    for i in range(len(HH_gt_onset)):
        HH_gt_onset[i] = data_utils.find_nearest(t, HH_gt_onset[i])

    for i in range(len(KD_gt_onset)):
        KD_gt_onset[i] = data_utils.find_nearest(t, KD_gt_onset[i])

    for i in range(len(SD_gt_onset)):
        SD_gt_onset[i] = data_utils.find_nearest(t, SD_gt_onset[i])



    pred_activations = predict_activations(filename)
    # import pdb; pdb.set_trace()
    HH_pred = np.add(pred_activations[0, :], pred_activations[1, :])
    KD_pred = np.add(pred_activations[2, :], pred_activations[3, :])
    SD_pred = np.add(pred_activations[4, :], pred_activations[5, :])
    
    HH_peaks, _ = find_peaks(HH_pred, width=2.5, prominence=3)
    KD_peaks, _ = find_peaks(KD_pred, width=3, prominence=3)
    SD_peaks, _ = find_peaks(SD_pred, width=2.5, prominence=3)

    plt.figure(); plt.plot(HH_peaks, HH_pred[HH_peaks], "ob"); plt.plot(HH_pred); plt.legend(['HH'])
    plt.xlabel('Time in samples')
    plt.ylabel('Amplitude')
    plt.figure(); plt.plot(KD_peaks, KD_pred[KD_peaks], "ob"); plt.plot(KD_pred); plt.legend(['KD'])
    plt.xlabel('Time in samples')
    plt.ylabel('Amplitude')
    plt.figure(); plt.plot(SD_peaks, SD_pred[SD_peaks], "ob"); plt.plot(SD_pred); plt.legend(['SD'])
    plt.xlabel('Time in samples')
    plt.ylabel('Amplitude')


    pred_time_HH = t[HH_peaks]
    pred_time_KD = t[KD_peaks]
    pred_time_SD = t[SD_peaks]


    f_measure_HH = mir_eval.beat.f_measure(HH_gt_onset,pred_time_HH)
    f_measure_KD = mir_eval.beat.f_measure(KD_gt_onset,pred_time_KD)
    f_measure_SD = mir_eval.beat.f_measure(SD_gt_onset, pred_time_SD)

    # f_measure_HH = mir_eval.onset.f_measure(np.array(HH_gt_onset), np.array(pred_time_HH), window=0.05)
    # pred_time_all = np.append(pred_time_HH, pred_time_KD)
    # pred_time_all = np.append(pred_time_all ,pred_time_SD)
    # pred_time_all = np.sort(pred_time_all)

    # gt_all = np.append(HH_gt_onset, KD_gt_onset) 
    # gt_all = np.append(gt_all, SD_gt_onset)
    # gt_all = np.sort(gt_all)
    # print ('Length of predicted beats ', len(pred_time_all))
    # print ('Length of ground truth beats ', len(gt_all))
    # f_measure = mir_eval.beat.f_measure(gt_all, pred_time_all)
    # print('F-measure = ', f_measure)
        
    f_measure_avg = (f_measure_HH + f_measure_SD + f_measure_KD)/3.0




    print('length of predicted beats for HH = ',  len(pred_time_HH), 'gt beats = ', len(HH_gt_onset))
    print('F-measure for HH = ', f_measure_HH)
    print('length of predicted beats for KD = ',  len(pred_time_KD), 'gt beats = ', len(KD_gt_onset))
    print('F-measure for KD = ', f_measure_KD)
    print('length of predicted beats for SD = ',  len(pred_time_SD), 'gt beats = ', len(SD_gt_onset))
    print('F-measure for SD = ', f_measure_SD)
    print('')
    print('Average F-measure = ', f_measure_avg)
    import pdb; pdb.set_trace()
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

    
    return f_measure_avg
