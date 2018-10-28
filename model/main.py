import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, '../data')
import data_utils
import hparams
import utils

gen_type = hparams.gen_type
drum_type_index = 3         # select 'MIX' 
data_dir = hparams.data_dir

audio_file_list = data_utils.get_audio_files(data_dir, drum_type_index, "all")

audio_file_list = [x.split('.')[0] for x in audio_file_list]

audio_file_list = audio_file_list[0:int(hparams.get_template_length*len(audio_file_list))]
file_list_length = len(audio_file_list)

xml_file_list = data_utils.get_xml_files(data_dir, drum_type_index, "all")
xml_file_list = [x.split('.')[0] for x in xml_file_list]
xml_file_list = np.intersect1d(xml_file_list, audio_file_list)

utils.get_templates()
# pred_activations = utils.predict_activations(filename)
# plt.subplot(311)
# plt.plot(pred_activations[0,:])
# plt.subplot(312)
# plt.plot(pred_activations[1,:])
# plt.subplot(313)
# plt.plot(pred_activations[2,:])
# filename = audio_file_list[0]
# utils.eval(filename)

# templates = np.load('templates.npy')
# HH_templates = templates[:,0]
# KD_templates = templates[:,1]
# SD_templates = templates[:,2]
# plt.figure()
# plt.plot(HH_templates); plt.legend(['HH'])
# plt.figure()
# plt.plot(KD_templates); plt.legend(['KD'])
# plt.figure()
# plt.plot(SD_templates); plt.legend(['SD'])
# plt.show()
