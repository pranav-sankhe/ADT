dataset_dir = '../datasets/tabla_solo_1.0'
onset_dir = '../datasets/tabla_solo_1.0/onsMap'
audio_dir = '../datasets/tabla_solo_1.0/wav'
score_dir = '../datasets/tabla_solo_1.0/score'
sample_rate = 44100.0

n_fft = 4096.0
win_length = 2048.0

hop_length = int(win_length/16.0)

bols = ['DA', 'KI', 'GE', 'TA', 'NA', 'DIN', 'KDA', 'TIT', 'DHA', 'DHE', 'DHET', 'DHI', 'DHIN', 'RE', 'TE', 'TII', 'TIN', 'TRA']
