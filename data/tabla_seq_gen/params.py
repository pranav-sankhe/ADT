isolated_drums_dir = '../datasets/tabla/isolated/'
onset_bol_dir = '../datasets/tabla/onsNoMap_ste/'
bol_map_file = '../datasets/tabla/bol_mapping.txt'
score_dir = '../datasets/tabla/scores/'
source_sep_data = '../datasets/tabla/source_sep/'
train_data = '../datasets/tabla/train_data/'

train_data_wav = '../datasets/tabla/train_data/wav/'
train_data_trans = '../datasets/tabla/train_data/trans/'

energy_threshold = 0.1
sample_rate = 44100

unique_bols = [ 'Da', 'Dha', 'Dhere', 'Dhi', 'Dhin', 'Dhit', 'Din', 'Kda', 'Na', 'Re', 
                    'Ta', 'Tak', 'Te', 'Tere', 'Ti', 'Tin', 'Tit', 'Traka', 'Trkt', 'Tun']

ge_dirs = ['Ge', 'Ge_1', 'Ge_2', 'Ge_3', 'Ge_4', 'Ge_5']                     
ke_dirs = ['Ke', 'Ke_1', 'Ke_2', 'Ke_3']

def rand_gen_strategy(strategy_id):
    # if strategy_id == 1: # permute between all the files within the isolated dirs



    if strategy_id == 2: # Exchange beats between two compositions


        beats = read_score_file(score_filepath)            
        num_beats = len(beats)
        samples_per_beat = int((bpm/60.0)*sample_rate)

        length = num_beats*samples_per_beat + 50000
        output_wav = np.zeros(length)
        onset_times, strokes = get_times_strokes(beats, bpm)