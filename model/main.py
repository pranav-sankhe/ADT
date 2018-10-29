import numpy as np 
# import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, '../data')
import data_utils
import hparams
import utils


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
