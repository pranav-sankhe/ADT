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
bol_map_file = params.bol_map_file

def read_wave_file(filepath):
    y, sr = librosa.load(filepath, sr=None)
    return y, sr

def get_gtOnsets(filepath):
    data = pd.read_csv(filepath, header=None)
    time = data[0].values
    bols = data[1].values
    for i in range(len(bols)):
        bols[i] = bols[i].replace(" ", "")
    return time, bols

##stEnergy function added by Rohit
def stEnergy(signal,winsize,hopsize,wintype,mode):
    hopi=0; ste=np.array([])
    while hopi+winsize<=len(signal):
        ste=np.append(ste,sum((signal[hopi:hopi+winsize]*eval('np.%s(%d)'%(wintype,winsize)))**2))
        hopi+=hopsize
    if mode=='log': return 10*np.log10(ste/max(ste))
    else: return ste/max(ste)
##

##bolMap function added by Rohit
def bolMap(bol_seq):
    bolmap=np.loadtxt(bol_map_file,delimiter=',',dtype=str)
    
    for ind in range(len(bol_seq)):
        if bol_seq[ind] == '-':
            pass
        else:
            bol_seq[ind]=bolmap[np.where(bolmap==bol_seq[ind].title())[0][0]][0]
    return bol_seq
##

def get_onset_time(filepath):
    y, sr = read_wave_file(filepath)
    max_index = np.argmax(stEnergy(y,int(sr*0.02),int(sr*0.01),'hamming','linear'))
    max_index *= 0.01
    max_index = (max_index + 0.005)*sample_rate
    y[0:1000] = np.multiply(y[0:1000], np.hamming(2000)[0:1000])
    return max_index ,y

def get_silent_part(filepath):
    y, sr = read_wave_file(filepath)
    energy = stEnergy(y,int(sr*0.02),int(sr*0.01),'hamming','linear')
    max_index = np.argmax(energy)
    energy = energy[max_index:]
    start_index = np.where(energy<1e-4)[0][0]
    start_index *= 0.01
    start_index = (start_index + 0.005)*sample_rate
    window = y[int(start_index):]
    
    return len(window)/2, window

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

    for i in range(len(beats)):
        beats[i] = ''.join(beats[i]).strip()
        beats[i] = ''.join(beats[i]).split(' ')
    

    return beats



def get_times_strokes(beats, bpm):

    samples_per_beat = int((bpm/60.0)*sample_rate)
    num_beats = len(beats)
    onset_times = []
    strokes = []
    time = 0

    for i in range(num_beats):
        samples_per_elem = samples_per_beat/len(beats[i])

        for elem in beats[i]:
            elem = elem.split(',')
            
            for e in elem:
                strokes.append(e)
                onset_times.append(time)
                time = time + samples_per_beat/len(beats[i]*len(elem))

    return onset_times, strokes


def refine_strokes(strokes):
    for i in range(len(strokes)):
        strokes[i] = strokes[i].replace(";", "")
    strokes=bolMap(strokes)
    return strokes


def get_aligned_window(filepath, onset_time, length):
    max_index, window = get_onset_time(filepath)
                        
    start = int(onset_time*sample_rate - max_index)
    end = int(len(window) - max_index + onset_time*sample_rate)

    if start < 0:
        start = 0
        start_index = int(max_index - onset_time*sample_rate)
        window = window[start_index:]

    if end > length:
        end = length
        end_index = int( (len(window) - max_index ) - (length - onset_time*sample_rate))
        window = window[0: len(window) - end_index]

    window = window[0: end-start]

    return start, end, window


def gen_from_transp(transcription_file):
    onset_times, bol_seq = get_gtOnsets(transcription_file)
    length = int(np.max(onset_times)*params.sample_rate + 50000)
    output_wav = np.zeros(length)
    
    bol_seq=bolMap(bol_seq)

    for i in range(len(bol_seq)):
        if bol_seq[i].lower() == ge_dirs[0].lower():
            # print("1 :Ghe deteceted")
            filenames = os.listdir(params.isolated_drums_dir + ge_dirs[0])
            filepath = params.isolated_drums_dir + ge_dirs[0] + '/' +  filenames[0]
            start, end, window = get_aligned_window(filepath, onset_times[i], length)
            output_wav[start:end] = np.add(output_wav[start:end] ,window) 

        elif bol_seq[i].lower() == ke_dirs[0].lower():    
            # print("2 : Ke deteceted")
            filenames = os.listdir(params.isolated_drums_dir + ke_dirs[0])
            filepath = params.isolated_drums_dir + ke_dirs[0] + '/' +  filenames[0]
            start, end, window = get_aligned_window(filepath, onset_times[i], length)
            output_wav[start:end] = np.add(output_wav[start:end] ,window) 

        
        else:
            for unique_bol in unique_bols: 
                if bol_seq[i].lower() == unique_bol.lower():
                    # print("3: ", bol_seq[i].lower() + " deteceted")
                    filenames = os.listdir(params.isolated_drums_dir + unique_bol)
                    filepath = params.isolated_drums_dir + unique_bol + '/' +  filenames[-1]
                    max_index, window = get_onset_time(filepath)
                                        
                    start, end, window = get_aligned_window(filepath, onset_times[i], length)
                    output_wav[start:end] = np.add(output_wav[start:end] ,window) 
    return output_wav


def gen_random_seq(output_wav, bpm, length, onset_times, bol_seq):            

    for i in range(len(bol_seq)):
        if bol_seq[i].lower() == ge_dirs[0].lower():
            # print("1 :Ghe deteceted")
            filenames = os.listdir(params.isolated_drums_dir + ge_dirs[0])
            filepath = params.isolated_drums_dir + ge_dirs[0] + '/' +  filenames[0]
            max_index, window = get_onset_time(filepath)                                
            start, end, window = get_aligned_window(filepath, onset_times[i], length)
            output_wav[start:end] = np.add(output_wav[start:end] ,window) 

        elif bol_seq[i].lower() == ke_dirs[0].lower():    
            # print("2 : Ke deteceted")
            filenames = os.listdir(params.isolated_drums_dir + ke_dirs[0])
            filepath = params.isolated_drums_dir + ke_dirs[0] + '/' +  filenames[0]
            max_index, window = get_onset_time(filepath)
            start, end, window = get_aligned_window(filepath, onset_times[i], length)
            output_wav[start:end] = np.add(output_wav[start:end] ,window) 


        elif bol_seq[i].lower() == '-':
            # print("3 : - deteceted")
            max_index, window = get_silent_part(filepath)
            start = int(onset_times[i] - max_index)
            end = int(len(window) - max_index + onset_times[i] )

            if start < 0:                    
                start = 0
                start_index = int(max_index - onset_times[i] )
                window = window[start_index:]

            if end > length:
                end = length
                end_index = int( (len(window) - max_index ) - (length - onset_times[i]))
                window = window[0: len(window) - end_index]

            window = window[0: end-start]
            output_wav[start:end] = np.add(output_wav[start:end] ,window) 
        
        else:
            for unique_bol in unique_bols: 
                if bol_seq[i].lower() == unique_bol.lower():
                    # print("4: ", bol_seq[i].lower() + " deteceted")
                    filenames = os.listdir(params.isolated_drums_dir + unique_bol)
                    filepath = params.isolated_drums_dir + unique_bol + '/' +  filenames[-1]
                    max_index, window = get_onset_time(filepath)                    
                    start, end, window = get_aligned_window(filepath, onset_times[i], length)
                    output_wav[start:end] = np.add(output_wav[start:end] ,window) 



    return output_wav




'''
strategy 1: Available files with varying BPM: filename code: score_filename + strat_1 +  bpm + .wav
strategy 2: Mixing up two score_files: filename code: score_filename1 + score_filename2 + strat_2 + bpm + .wav
'''

def write_trans_file(bol_seq, onset_times, filepath):
    columns = ['bols', 'onsTime']
    dataframe = pd.DataFrame(columns=columns)
    dataframe[columns[0]] = bol_seq
    dataframe[columns[1]] = onset_times

    dataframe.to_csv(filepath)


def strategy_1(bpm):

    sample_rate = params.sample_rate
    score_dir = params.score_dir
    score_files = os.listdir(score_dir)
    
    for file in score_files:
        file = score_dir + file
        filename = file.split('/')[-1]
        print('Generating tabla sequence for ', filename)

        beats = read_score_file(file)
        num_beats = len(beats)
        samples_per_beat = int((bpm/60.0)*sample_rate)
        length = num_beats*samples_per_beat + 100000
        output_wav = np.zeros(length)
        onset_times, strokes = get_times_strokes(beats, bpm)
        bol_seq = refine_strokes(strokes)        

        output_wav = gen_random_seq(output_wav, bpm, length, onset_times, bol_seq)
        import pdb; pdb.set_trace()
        store_path = params.train_data_wav + filename.split('.')[0] + '_strat_1_' + str(bpm) + '.wav'
        librosa.output.write_wav(store_path, output_wav, sample_rate)
        trans_filepath = params.train_data_trans + filename.split('.')[0] + '_strat_1_' + str(bpm) + '.csv'
        write_trans_file(bol_seq, onset_times, trans_filepath)
        
        print("Stored the wav file at ", store_path, ' and labels at ', trans_filepath)



def strategy_2(bpm, filepath_1, filepath_2):

    beats_1 = read_score_file(params.score_dir + filepath_1)
    beats_2 = read_score_file(params.score_dir + filepath_2)

    beats = []
    for i in range(min(len(beats_1), len(beats_2))):
        if i%2 == 0:
            beats.append(beats_1[i])
        if i%2 == 1:
            beats.append(beats_2[i])

    beats = beats_1        
    num_beats = len(beats)
    samples_per_beat = int((bpm/60.0)*sample_rate)

    length = num_beats*samples_per_beat + 100000
    output_wav = np.zeros(length)
    onset_times, strokes = get_times_strokes(beats, bpm)
    bol_seq = refine_strokes(strokes)
    
    output_wav = gen_random_seq(output_wav, bpm, length, onset_times, bol_seq)
    import pdb; pdb.set_trace()

    filename_1 = filepath_1.split('/')[-1]
    filename_2 = filepath_2.split('/')[-1]
    store_path = params.train_data_wav + filename_1.split('.')[0] + '_' + filename_2.split('.')[0] + '_strat_2_' + str(bpm) + '.wav'

    librosa.output.write_wav(store_path, output_wav, sample_rate)
    
    trans_filepath = params.train_data_trans + filename_1.split('.')[0] + '_' + filename_2.split('.')[0] + '_strat_2_' + str(bpm) + '.csv'
    write_trans_file(bol_seq, onset_times, trans_filepath)
    
    print("Stored the wav file at ", store_path, ' and labels at ', trans_filepath)
