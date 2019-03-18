import os
import pandas as pd 
import numpy as np 
import params
import librosa
from scipy import signal
import matplotlib.pyplot as plt


unique_bols = params.unique_bols
ge_dirs = params.ge_dirs
ke_dirs = params.ke_dirs
sample_rate = params.sample_rate

def read_score_file(filepath):
    f = open(filepath, "r")
    flag = 0
    tabla_seq = []
    beats = []

    for line in f: 
        if '!S' in line:
            flag = 1
        if flag == 1:
            tabla_seq.append(line)
    
    for seq in tabla_seq:
        val = []
        for element in seq:
            val.append(element)
            if element == ';':
                beats.append(val)
                val = []
    return beats


def read_wave_file(filepath):
    y, sr = librosa.load(filepath, sr=None)
    return y, sr

def get_gtOnsets(filepath):
    file_type = filepath.split('.')[-1]
    if file_type == 'csv':
        data = pd.read_csv(filepath, header=None)
        time = data[0].values
        bols = data[1].values
        for i in range(len(bols)):
            bols[i] = bols[i].replace(" ", "")
        return time, bols
    
    if file_type == 'txt':
        f = open(filepath, "r")
        time = []
        bols = []
        for line in f:
            time.append(float(line.split(',')[0]))
            bols.append(line.split(',')[1].strip())
        return time, bols    

def get_onset_time(filepath, onset_time, length):
    window, sr = read_wave_file(filepath)

    y_len = 1024
    y = np.hamming(y_len)
    sig_energy = np.convolve(window**2,y**2,'same')
    max_index = np.argmax(sig_energy)
    

    start = int(onset_time*sample_rate - max_index)
    end = int(len(window) - max_index + onset_time*sample_rate)
    
    if start < 0:
        start = 0
        start_index = int(max_index - onset_time*sample_rate)
        window = window[start_index:]

    if end > length:
        end = length
        end_index = int( (len(window) - max_index ) - (length - onset_time))
        window = window[0: len(window) - end_index]

    window = window[0: end-start]
    return window, start, end


        
def gen(transcription_file, score_file, bpm, transcription=False, scores=True):
    if transcription:
        onset_times, bol_seq = get_gtOnsets(transcription_file)
        length = int(np.max(onset_times)*params.sample_rate + 1000)
        output_wav = np.zeros(length)

        for i in range(len(bol_seq)):
            
            if bol_seq[i].lower() == ge_dirs[0].lower():
                print("1 :Ghe deteceted")
                filenames = os.listdir(params.isolated_drums_dir + ge_dirs[0])
                filepath = params.isolated_drums_dir + ge_dirs[0] + '/' +  filenames[-1]
                
                window, start, end = get_onset_time(filepath, onset_times[i], length)
                output_wav[start:end] = np.add(output_wav[start:end] ,window) 


            if bol_seq[i].lower() == ke_dirs[0].lower():    
                print("2 : Ke deteceted")
                filenames = os.listdir(params.isolated_drums_dir + ke_dirs[0])
                filepath = params.isolated_drums_dir + ke_dirs[0] + '/' +  filenames[-1]


                window, start, end = get_onset_time(filepath, onset_times[i], length)
                output_wav[start:end] = np.add(output_wav[start:end] ,window) 

            
            else:
                for unique_bol in unique_bols: 
                    if bol_seq[i].lower() in unique_bol.lower():
                        print("3: ", bol_seq[i].lower() + " deteceted")
                        filenames = os.listdir(params.isolated_drums_dir + unique_bol)
                        filepath = params.isolated_drums_dir + unique_bol + '/' +  filenames[-1]

                        window, start, end = get_onset_time(filepath, onset_times[i], length)
                        output_wav[start:end] = np.add(output_wav[start:end] ,window) 


        return output_wav

    if scores:
        beats = read_score_file(score_filepath)            
        get_time_intervals(bpm, beats)


def get_time_intervals(bpm, beats):
    bols = []
    time_intervals = []
    
    time_per_beat = (bpm/60.0)*sample_rate
    
    segment_by_space = []
    space = ' '
    for i in range(len(beats)):
        val_1 = []
        val = []
        for element in beats[i]:
            val.append(element)
            if space == element:
                val_1.append(val)
                val = []
        segment_by_space.append(val_1)
        val_1 = []
    # for i in range(len(segment_by_space)): 
    #     segment_by_space[i] = ''.join(segment_by_space[i]).strip()
    for i in segment_by_space:
        print i

    import pdb; pdb.set_trace()            




# transcription_file = params.onset_bol_dir + 'dli_3.csv'    
transcription_file = params.onset_bol_dir_ste + 'dli_3.txt'

score_file = []
bpm = []
output_wav = gen(transcription_file, score_file, bpm, transcription=True, scores=False)
librosa.output.write_wav('output.wav', output_wav, params.sample_rate)

# score_dir = params.score_dir
# score_files = os.listdir(score_dir)
# score_filepath = score_dir + '/' + score_files[0]
# bpm = 100
# # gen(transcription_file, transcription=False, scores=True, score_filepath, bpm)
# gen(transcription_file, score_filepath, bpm, transcription=False, scores=True)

# filepath = params.isolated_drums_dir + '/Da/' + 'Da_1_l_1.wav' 
# get_onset_time(filepath)
