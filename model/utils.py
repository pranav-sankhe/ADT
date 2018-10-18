import numpy as np
import sys 
sys.path.insert(0, '../data')
import data_utils
import nimfa
import hparams

def get_templates():
    ## ['HH', 'KD', 'SD', 'MIX']
    ## HH
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
    svl_file_list    = data_utils.get_svl_files(data_dir, 'all', "all")
    svl_file_list = [x.split('.')[0] for x in svl_file_list]

    # prepare a 3*104 size matrix of file names.  [HH, KD, SD] * [# of recordings] 
    
    svl_files = np.zeros(( file_list_length, 3 ))
    svl_files = svl_files.astype(str)
    for i in range(len(svl_file_list)):
        for l in range(file_list_length):
            
            if svl_file_list[i].find(audio_file_names[l]) != -1:
                if svl_file_list[i].find('HH') != -1:
                    svl_files[l][0] = svl_file_list[i]
                if svl_file_list[i].find('KD') != -1:
                    svl_files[l][1] = svl_file_list[i]
                if svl_file_list[i].find('SD') != -1:
                    svl_files[l][2] = svl_file_list[i]


    n_fft = hparams.n_fft
    hop_length =  hparams.hop_length
    win_length = hparams.win_length
    window = hparams.window
    
    
    flag = 0
    K, T = data_utils.get_spec_dims(hparams.test_filepath, n_fft, hop_length, win_length)

    prev_template = np.zeros((K, hparams.num_drums))
    sum_of_templates = []
    for i in range(file_list_length):
        V = data_utils.get_spectrogram(data_dir + '/audio/' + audio_file_list[i] + '.wav', n_fft, hop_length, win_length)
        activations = data_utils.create_gt_activations( data_dir + '/annotation_xml/' + xml_file_list[i] + '.xml', n_fft, hop_length, win_length, T)
        lsnmf = nimfa.Lsnmf(V, seed=None, H=activations)
        lsnmf_fit = lsnmf()
        template = lsnmf_fit.basis()
        sum_of_templates = np.add(prev_template, template)
        prev_templates = sum_of_templates
    avg_template = np.divide(prev_templates, file_list_length)

    return avg_template

        



def nmf(V, init_W=None, init_H=None):

    if init_W == None:
        lsnmf = nimfa.Lsnmf(V, H=init_H, max_iter=10, rank=3)
        lsnmf_fit = lsnmf()
        H = lsnmf_fit.coef()
        # print('Template:\n%s' % H)

        # print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
        return H



    # init_W = np.random.rand(30, 4)
    # init_H = np.random.rand(4, 20)

    # # Fixed initialization of latent matrices
    # nmf = nimfa.Nmf(V, seed="fixed", W=init_W, H=init_H, rank=4)
    # nmf_fit = nmf()

    # # print("Euclidean distance: %5.3f" % nmf_fit.distance(metric="euclidean"))
    # # print('Initialization type: %s' % nmf_fit.seeding)
    # # print('Iterations: %d' % nmf_fit.n_iter)

    # W = nmf_fit.basis()
    # print('Basis matrix:\n%s' % W)

    # H = nmf.coef()
    # print('Mixture matrix:\n%s' % H)  

get_templates()

