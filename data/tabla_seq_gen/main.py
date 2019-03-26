import os
import pandas as pd 
import numpy as np 
import params
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import utils


def gen_source_sep_data():
    transcription_dir = params.onset_bol_dir
    transcription_files = os.listdir(transcription_dir)
    
    score_filepath = []
    bpm = []

    if not os.path.exists(params.source_sep_data):
        os.makedirs(params.source_sep_data)

    for file in transcription_files:
        file = transcription_dir + file
        filename = file.split('/')[-1]
        print('Generating tabla sequence for ', filename)
        output_wav = utils.gen_from_transp(file)
        store_path = params.source_sep_data + filename.split('.')[0] + '.wav'
        librosa.output.write_wav(store_path, output_wav, params.sample_rate)
        print("Stored the wav file at ", store_path)



def gen_train_data(bpm = 70):
    transcription_file = []
    score_dir = params.score_dir
    score_files = os.listdir(score_dir)

    if not os.path.exists(params.train_data):
        os.makedirs(params.train_data)


    for file in score_files:
        file = score_dir + file
        filename = file.split('/')[-1]
        print('Generating tabla sequence for ', filename)
    
        output_wav = utils.gen_random_seq(file, bpm)
        
        store_path = params.train_data + filename.split('.')[0] + '.wav'
        librosa.output.write_wav(store_path, output_wav, params.sample_rate)
        print("Stored the wav file at ", store_path)


# gen_source_sep_data()
gen_train_data(bpm = 70)