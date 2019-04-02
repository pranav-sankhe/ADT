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

    train_data_wav = params.train_data_wav
    train_data_trans = params.train_data_trans

    if not os.path.exists(train_data_wav):
        os.makedirs(train_data_wav)

    if not os.path.exists(train_data_trans):
        os.makedirs(train_data_trans)


    utils.strategy_1(bpm)

    for i in range(len(score_files)):
        for j in range(len(score_files)):
            if i != j:
                print(score_files[i], score_files[j])
                utils.strategy_2(bpm, score_files[i], score_files[j])


# gen_source_sep_data()
gen_train_data(bpm = 70)